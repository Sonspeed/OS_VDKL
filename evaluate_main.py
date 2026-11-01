from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
import os
import nltk
for pkg in ['punkt',]:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, all_gather

from rouge_metric import compute_metrics, metric_max_over_ground_truths, rouge

torch.set_num_threads(4)


def prepare_dataset_main(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data


def run_model(args, tokenizer, model, dataset: PromptDataset, epoch, device, decode_mode=None, force_branch=None, compute_lm=True):
    
    collate_fn = dataset.collate
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    if decode_mode is not None:
        from os_vdkl.multi_lora import unwrap_model
        core = unwrap_model(model)
        if hasattr(core, "os_decode_mode"):
            core.os_decode_mode = decode_mode
            core.os_force_branch = force_branch
            core.os_branch_ids = None
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses = []
    
    generation_config = GenerationConfig (
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)

            if compute_lm:
                all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
                input_ids = all_ids[:, :-1]
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                label_ids = all_ids[:, 1:]
                label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
                label_ids[:, :model_batch["input_ids"].size(1)-1] = -100
                if args.model_type in ["gpt2"]:
                    position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                    out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
                else:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = out.logits
                loss_mask = (label_ids != -100).float()
                loss_func = nn.CrossEntropyLoss(reduction="none")
                lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
                all_lm_losses.append(lm_loss)

            query_ids = model_batch["input_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            
            query_ids = F.pad(query_ids, (args.max_prompt_length-query_ids.size(1), 0, 0, 0), value=tokenizer.pad_token_id)
            response_ids = F.pad(response_ids, (0, args.max_length-args.max_prompt_length-response_ids.size(1), 0, 0), value=tokenizer.pad_token_id)
            
            all_query_ids.append(query_ids)
            all_response_ids.append(response_ids)

    if compute_lm:
        all_lm_losses = torch.cat(all_lm_losses)
        mean_lm_loss = all_lm_losses.mean()
        dist.all_reduce(mean_lm_loss, dist.ReduceOp.SUM, group=dp_group)
        mean_lm_loss = mean_lm_loss.item() / dp_world_size
    else:
        mean_lm_loss = 0.0
        
    all_query_ids = torch.cat(all_query_ids)
    all_query_ids = all_gather(all_query_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_query_ids = all_query_ids.view(-1, all_query_ids.size(-1))
    all_query_ids = all_query_ids[:len(dataset)]
    
    all_response_ids = torch.cat(all_response_ids)
    all_response_ids = all_gather(all_response_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
    all_response_ids = all_response_ids[:len(dataset)]
        
    return (
        mean_lm_loss,
        all_query_ids,
        all_response_ids)


def evaluate_main(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    if _uses_os_branches(args, model):
        _evaluate_os_branches(args, tokenizer, model, dataset, split, epoch, device)
        return

    lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, epoch, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    torch.save(all_preds, os.path.join(args.save, "preds.pt"))

    all_responses = []
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:
        for p in all_preds[0]:
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip()
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())
    
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    log_str = f"{split} | name: {args.data_names} | {gen_res} | lm_loss {round(lm_loss, 4)} | avg. gen lenth: {mean_gen_length}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))


def _uses_os_branches(args, model):
    if not getattr(args, "os_vdkl", False):
        return False
    if getattr(args, "os_eval_mode", "all") != "all":
        return False
    from os_vdkl.multi_lora import unwrap_model
    return hasattr(unwrap_model(model), "os_lora_A")


def _clean_response(query, response):
    text = (query + response)[len(query):]
    end = text.find("<|endoftext|>")
    if end >= 0:
        text = text[:end]
    return text.replace("<n>", "\n").strip()


def _per_example_rouge(predictions, references):
    scores = []
    for pred, gold in zip(predictions, references):
        if not isinstance(gold, list):
            gold = [gold]
        scores.append(100.0 * metric_max_over_ground_truths(rouge, pred, gold))
    return scores


def _evaluate_os_branches(args, tokenizer, model, dataset, split, epoch, device):
    """Four decoding passes: each LoRA branch, then the log-mixture."""
    from os_vdkl.metrics import cross_branch_self_bleu, distinct_n
    from os_vdkl.multi_lora import unwrap_model

    core = unwrap_model(model)
    previous = (core.os_decode_mode, core.os_branch_ids, core.os_force_branch)
    num_branches = int(core.os_lora_A.size(0))
    try:
        branch_texts = []
        query_strs = None
        for branch in range(num_branches):
            _, query_ids, response_ids = run_model(
                args, tokenizer, model, dataset, epoch, device,
                decode_mode="branch", force_branch=branch, compute_lm=False,
            )
            queries = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            branch_texts.append([_clean_response(q, r) for q, r in zip(queries, responses)])
            if query_strs is None:
                query_strs = queries

        lm_loss, query_ids, response_ids = run_model(
            args, tokenizer, model, dataset, epoch, device,
            decode_mode="mixture", compute_lm=True,
        )
        query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
        response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        mixture_texts = [_clean_response(q, r) for q, r in zip(query_strs, response_strs)]
    finally:
        core.os_decode_mode, core.os_branch_ids, core.os_force_branch = previous

    if dist.get_rank() != 0:
        return

    references = list(dataset.answers)
    n = min(len(mixture_texts), len(references), *(len(texts) for texts in branch_texts))
    query_strs = query_strs[:n]
    response_strs = response_strs[:n]
    mixture_texts = mixture_texts[:n]
    branch_texts = [texts[:n] for texts in branch_texts]
    references = references[:n]
    gen_res = compute_metrics(mixture_texts, references)
    branch_means = []
    branch_example_scores = []
    for texts in branch_texts:
        branch_means.append(compute_metrics(texts, references)["rougeL"])
        branch_example_scores.append(_per_example_rouge(texts, references))
    mixture_example = _per_example_rouge(mixture_texts, references)
    oracle_scores = []
    oracle_branch = []
    for i in range(len(mixture_texts)):
        scores_i = [branch_example_scores[k][i] for k in range(num_branches)]
        best = max(range(num_branches), key=lambda k: scores_i[k])
        oracle_scores.append(scores_i[best])
        oracle_branch.append(best)
    oracle = float(np.mean(oracle_scores)) if oracle_scores else 0.0
    mean_branch = float(np.mean(branch_means)) if branch_means else 0.0
    self_bleu = cross_branch_self_bleu(branch_texts)
    distinct2 = distinct_n(branch_texts, n=2)

    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")
    all_preds = [[(q, q + r) for q, r in zip(query_strs, response_strs)]]
    torch.save(all_preds, os.path.join(args.save, "preds.pt"))
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:
        for text in mixture_texts:
            f.write(json.dumps({"text": text}) + "\n")

    phase2_path = os.path.join(args.save, f"phase2_candidates_{args.data_names}.jsonl")
    with open(phase2_path, "w") as f:
        for i, prompt in enumerate(query_strs):
            gold = references[i] if i < len(references) else ""
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            record = {
                "id": i,
                "prompt": prompt,
                "gold": gold,
                "branch_responses": [branch_texts[k][i] for k in range(num_branches)],
                "branch_rougeL": [round(branch_example_scores[k][i], 4) for k in range(num_branches)],
                "oracle_best_branch": int(oracle_branch[i]),
                "mixture_response": mixture_texts[i],
                "mixture_rougeL": round(mixture_example[i], 4),
            }
            f.write(json.dumps(record) + "\n")

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])
    # gen_res is first so existing log parsers keep reading mixture ROUGE-L.
    log_str = (
        f"{split} | name: {args.data_names} | {gen_res} | lm_loss {round(lm_loss, 4)} | "
        f"avg. gen lenth: {mean_gen_length} | "
        f"Oracle_Best_of_K_ROUGE-L: {oracle:.4f} | Mean_Branch_ROUGE-L: {mean_branch:.4f} | "
        f"Cross_Branch_Self_BLEU: {self_bleu:.4f} | Distinct_2: {distinct2:.4f} | "
        f"branch_rougeL: {[round(x, 4) for x in branch_means]}"
    )
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
