"""Sanity checks for OS-VDKL.

Usage (from the repository root): python tools/test_os_vdkl.py

1. The M x K log-domain Sinkhorn solver preserves both marginals to 1e-5
   in the default 5 iterations (a feasible plan is a fixed point, and a
   mild cost reaches the constraints in 5 steps at eps=0.05).
2. At orthogonal initialization the K branch shifts differ, and the full
   loss gives a non-zero gradient to every branch that is not a copy of
   the others.
3. L_local does not send gradient into h0 or W_head. L_global does.
4. Factored logit shifts match the explicit hidden residual, chunked loss
   matches the single-chunk loss, and a tiny GPT-2 forward/generate/save
   round-trip keeps the branch weights.
5. Wall-clock of the M x K solver against WASD's sparse Sinkhorn.
"""
import importlib.util
import math
import os
import sys
import tempfile
import time
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")


def _load(name, path):
    # Load the module file directly so this check does not import DeepSpeed via distillm/__init__.py.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_losses = _load("os_vdkl_losses_impl", os.path.join("distillm", "losses.py"))
_sampler = _load("os_vdkl_sampler_impl", os.path.join("distillm", "sampler.py"))
log_domain_sinkhorn = _losses.log_domain_sinkhorn
os_vdkl_loss = _losses.os_vdkl_loss
os_vdkl_loss_from_hidden = _losses.os_vdkl_loss_from_hidden
sparse_sinkhorn_sequence = _losses.sparse_sinkhorn_sequence
SampleGenerator = _sampler.SampleGenerator

from os_vdkl.metrics import cross_branch_self_bleu, distinct_n  # noqa: E402
from os_vdkl.multi_lora import (  # noqa: E402
    attach_os_multi_lora,
    factored_logit_shift,
    hidden_logit_shift,
    init_os_lora_parameters,
    mixture_log_probs,
    os_decode_context,
)


def _args(**overrides):
    base = dict(
        os_vdkl=True,
        os_num_branches=3,
        os_top_m=16,
        os_lora_rank=4,
        os_lora_alpha=16.0,
        os_init_scale=0.02,
        os_sinkhorn_eps=0.05,
        os_sinkhorn_iters=5,
        os_lambda_local=0.5,
        os_mu_ortho=0.1,
        os_eta_sem=0.05,
        os_sgo_mode="random_branch",
        os_eval_mode="all",
        student_temp=1.0,
        os_chunk_tokens=64,
        model_path=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _marginal_errors(gamma, teacher_hat):
    num_branches = gamma.size(-1)
    row = (gamma.sum(dim=-1) - 1).abs().max().item()
    col = ((teacher_hat.unsqueeze(-1) * gamma).sum(dim=1) - 1.0 / num_branches).abs().max().item()
    return row, col


def test_sinkhorn_marginals():
    torch.manual_seed(0)
    num_tokens, support, num_branches = 32, 64, 3
    eps, iters = 0.05, 5
    teacher = torch.softmax(torch.randn(num_tokens, support), dim=-1)
    log_teacher = torch.log(teacher.clamp_min(1e-12))

    raw = torch.rand(num_tokens, support, num_branches) + 0.05
    gamma = raw / raw.sum(dim=-1, keepdim=True)
    for _ in range(80):
        column = (teacher.unsqueeze(-1) * gamma).sum(dim=1, keepdim=True).clamp_min(1e-12)
        gamma = gamma * ((1.0 / num_branches) / column)
        gamma = gamma / gamma.sum(dim=-1, keepdim=True)
    built_row, built_col = _marginal_errors(gamma, teacher)
    cost = -eps * gamma.clamp_min(1e-12).log()
    solved = log_domain_sinkhorn(cost, log_teacher, eps, iters)
    row, col = _marginal_errors(solved, teacher)
    drift = (solved - gamma).abs().max().item()
    print(f"[1] feasible plan before solver     : row {built_row:.2e} | col {built_col:.2e}")
    print(f"[1] preserved after {iters} iters       : row {row:.2e} | col {col:.2e} | drift {drift:.2e}")

    mild = torch.rand(num_tokens, support, num_branches) * eps
    mild_gamma = log_domain_sinkhorn(mild, log_teacher, eps, iters)
    mild_row, mild_col = _marginal_errors(mild_gamma, teacher)
    print(f"[1] mild cost after {iters} iters        : row {mild_row:.2e} | col {mild_col:.2e}")

    sharp = -torch.log_softmax(torch.randn(num_tokens, support, num_branches), dim=1)
    sharp_gamma = log_domain_sinkhorn(sharp, log_teacher, eps, iters)
    sharp_row, sharp_col = _marginal_errors(sharp_gamma, teacher)
    print(f"[1] energy cost after {iters} iters      : row {sharp_row:.2e} | col {sharp_col:.2e} (sharp kernel, info)")
    ok = row < 1e-5 and col < 1e-5 and drift < 1e-5 and mild_row < 1e-5 and mild_col < 1e-5 and sharp_row < 1e-5
    return ok


def _branch_pack(hidden, vocab, rank=4, num_branches=3):
    torch.manual_seed(0)
    lora_A, lora_B = init_os_lora_parameters(num_branches, rank, hidden, 0.02)
    lora_A = lora_A.detach().requires_grad_(True)
    lora_B = lora_B.detach().requires_grad_(True)
    weight = torch.randn(vocab, hidden, requires_grad=True)
    return lora_A, lora_B, weight


def test_projection_and_asymmetry():
    torch.manual_seed(0)
    num_tokens, hidden, vocab = 8, 32, 48
    h0 = torch.randn(num_tokens, hidden, requires_grad=True)
    lora_A, lora_B, weight = _branch_pack(hidden, vocab)
    scale = 16.0 / 4.0

    off = 0.0
    eye = (0.02 ** 2) * torch.eye(lora_B.size(-1))
    for i in range(lora_B.size(0)):
        gram = lora_B[i].detach().float().T @ lora_B[i].detach().float()
        off = max(off, (gram - eye).abs().max().item())
        for j in range(i + 1, lora_B.size(0)):
            cross = lora_B[i].detach().float().T @ lora_B[j].detach().float()
            off = max(off, cross.abs().max().item())
    print(f"[2] orthogonal B gram error         : {off:.2e}")

    factored = factored_logit_shift(h0, lora_A, lora_B, weight, scale)
    explicit = hidden_logit_shift(h0, lora_A, lora_B, weight, scale)
    proj_err = (factored - explicit).abs().max().item()
    spread = (factored[:, 0] - factored[:, 1]).abs().mean().item()
    print(f"[2] factored vs hidden shift error  : {proj_err:.2e}")
    print(f"[2] mean |delta z_0 - delta z_1|    : {spread:.2e}")

    teacher = torch.randn(num_tokens, vocab) * 3
    args = _args()
    parts = os_vdkl_loss_from_hidden(h0, teacher, lora_A, lora_B, weight, args)
    grads = torch.autograd.grad(parts["loss"], [lora_A, lora_B, h0, weight], retain_graph=True)
    gA, gB, gH, gW = grads
    pair = 0.0
    for i in range(gB.size(0)):
        assert gB[i].norm().item() > 1e-8, gB[i].norm().item()
        assert gA[i].norm().item() > 1e-8, gA[i].norm().item()
        for j in range(i + 1, gB.size(0)):
            pair = max(pair, (gB[i] - gB[j]).abs().max().item())
    print(f"[2] full-loss |gB_i - gB_j|         : {pair:.2e} | |g h0| {gH.norm().item():.2e} | |g W| {gW.norm().item():.2e}")

    local_grads = torch.autograd.grad(
        parts["l_local"], [h0, weight, lora_A, lora_B], retain_graph=True, allow_unused=True,
    )
    gH_l, gW_l, gA_l, gB_l = local_grads
    h_norm = 0.0 if gH_l is None else gH_l.norm().item()
    w_norm = 0.0 if gW_l is None else gW_l.norm().item()
    print(f"[3] L_local |g h0| {h_norm:.2e} | |g W| {w_norm:.2e} | |g A| {gA_l.norm().item():.2e} | |g B| {gB_l.norm().item():.2e}")

    global_h = torch.autograd.grad(parts["l_global"], h0, retain_graph=True)[0]
    print(f"[3] L_global |g h0|                 : {global_h.norm().item():.2e}")

    args_chunk = _args(os_chunk_tokens=3)
    # A second graph on the same leaves. Compare values and gradients.
    h1 = h0.detach().requires_grad_(True)
    a1 = lora_A.detach().requires_grad_(True)
    b1 = lora_B.detach().requires_grad_(True)
    w1 = weight.detach().requires_grad_(True)
    full = os_vdkl_loss_from_hidden(h1, teacher, a1, b1, w1, _args(os_chunk_tokens=1000))
    chunked = os_vdkl_loss_from_hidden(h1, teacher, a1, b1, w1, args_chunk)
    value_gap = (full["loss"] - chunked["loss"]).abs().item()
    g_full = torch.autograd.grad(full["loss"], [h1, a1, b1, w1], allow_unused=True)
    g_chunk = torch.autograd.grad(chunked["loss"], [h1, a1, b1, w1], allow_unused=True)
    grad_gap = max((x - y).abs().max().item() for x, y in zip(g_full, g_chunk))
    print(f"[2] chunked vs full loss / grad gap : {value_gap:.2e} / {grad_gap:.2e}")

    ok = (
        off < 1e-4 and proj_err < 1e-4 and spread > 1e-4 and pair > 1e-6
        and h_norm < 1e-8 and w_norm < 1e-8 and gA_l.norm().item() > 1e-8 and gB_l.norm().item() > 1e-8
        and global_h.norm().item() > 1e-8 and value_gap < 1e-4 and grad_gap < 1e-4
        and math.isfinite(parts["loss"].item())
    )
    return ok


def test_student_wrapper():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2, n_head=4, n_embd=32, vocab_size=64, n_positions=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.train()
    args = _args(os_top_m=8, os_chunk_tokens=4, student_temp=2.0)
    attach_os_multi_lora(model, args)

    tokens = torch.randint(0, config.vocab_size, (2, 8))
    with os_decode_context(model, "base"):
        base = model(input_ids=tokens)
    base_ref = F.linear(base.os_h0, model.lm_head.weight)
    base_err = (base.logits - base_ref).abs().max().item()

    with os_decode_context(model, "mixture"):
        mixed = model(input_ids=tokens)
    prob_err = (mixed.logits.float().exp().sum(-1) - 1).abs().max().item()

    with os_decode_context(model, "branch", force_branch=0):
        branch0 = model(input_ids=tokens).logits
    with os_decode_context(model, "branch", force_branch=1):
        branch1 = model(input_ids=tokens).logits
    branch_gap = (branch0 - branch1).abs().mean().item()

    ids = torch.tensor([0, 1])
    with os_decode_context(model, "random_branch", branch_ids=ids):
        picked = model(input_ids=tokens).logits
    pick_err = max(
        (picked[0] - branch0[0]).abs().max().item(),
        (picked[1] - branch1[1]).abs().max().item(),
    )

    z0 = torch.randn(2, 5, config.vocab_size)
    h0 = torch.randn(2, 5, config.n_embd)
    manual = mixture_log_probs(z0, h0, model.os_lora_A, model.os_lora_B, model.lm_head.weight, model.os_scale)
    branch_log = []
    delta = factored_logit_shift(h0, model.os_lora_A, model.os_lora_B, model.lm_head.weight, model.os_scale)
    for k in range(3):
        branch_log.append(torch.log_softmax(z0 + delta[:, :, k], dim=-1))
    manual_ref = torch.logsumexp(torch.stack(branch_log, dim=2), dim=2) - math.log(3)
    mix_err = (manual - manual_ref).abs().max().item()

    labels = tokens.clone()
    labels[:, :1] = -100
    with os_decode_context(model, "base"):
        outputs = model(input_ids=tokens)
    teacher = torch.randn_like(outputs.logits)
    loss = os_vdkl_loss(args, outputs, teacher, {"label": labels}, model)
    loss.backward()
    lora_grad = model.os_lora_A.grad.norm().item()
    backbone_grad = model.transformer.ln_f.weight.grad.norm().item()
    print(f"[4] base logit error {base_err:.2e} | mixture sum-1 {prob_err:.2e} | branch gap {branch_gap:.2e}")
    print(f"[4] random-branch match {pick_err:.2e} | mixture formula {mix_err:.2e}")
    print(f"[4] backward |g A| {lora_grad:.2e} | |g ln_f| {backbone_grad:.2e} | loss {loss.item():.4f}")

    model.os_lora_A.grad = None
    model.os_lora_B.grad = None
    with os_decode_context(model, "mixture"):
        generated = model.generate(tokens[:, :4], max_new_tokens=2, do_sample=False, pad_token_id=0)
    gen_ok = tuple(generated.shape) == (2, 6)

    class _Tok:
        pad_token_id = 0
        eos_token_id = 0

    sample_args = _args(
        max_length=12, max_prompt_length=6, do_sample=False, gen_top_p=1.0, top_k=0,
        temperature=1.0, repetition_penalty=1.0, os_sgo_mode="random_branch",
    )
    before = model.os_decode_mode
    sampled = SampleGenerator(sample_args, _Tok()).run_sample(
        model, {"input_ids": tokens[:, :4], "attention_mask": torch.ones(2, 4, dtype=torch.long)},
    )
    restored = model.os_decode_mode == before and model.os_branch_ids is None
    print(f"[4] generate shape {tuple(generated.shape)} | sample ids {tuple(sampled['input_ids'].shape)} | mode restored {restored}")

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=False)
        from os_vdkl.multi_lora import save_os_multi_lora
        save_os_multi_lora(model, tmp, args)
        fresh = GPT2LMHeadModel.from_pretrained(tmp)
        load_args = _args(model_path=tmp, os_vdkl=False, os_num_branches=1, os_lora_rank=1, os_lora_alpha=1.0)
        attach_os_multi_lora(fresh, load_args)
        weight_err = max(
            (fresh.os_lora_A.cpu() - model.os_lora_A.detach().cpu()).abs().max().item(),
            (fresh.os_lora_B.cpu() - model.os_lora_B.detach().cpu()).abs().max().item(),
        )
        loaded_k = fresh.os_lora_A.shape[0]
        loaded_r = fresh.os_lora_A.shape[1]
    print(f"[4] sidecar reload error {weight_err:.2e} | K {loaded_k} r {loaded_r} alpha {load_args.os_lora_alpha}")

    identical = ["the cat sat on the mat"] * 3
    diverse = ["alpha beta gamma delta", "river stone quiet lamp", "orbit cedar violet mink"]
    bleu_same = cross_branch_self_bleu([identical, identical, identical])
    bleu_diff = cross_branch_self_bleu([[diverse[0]], [diverse[1]], [diverse[2]]])
    dist_same = distinct_n([identical, identical, identical], n=2)
    dist_diff = distinct_n([[diverse[0]], [diverse[1]], [diverse[2]]], n=2)
    print(f"[4] self-BLEU same/diff {bleu_same:.3f}/{bleu_diff:.3f} | distinct-2 same/diff {dist_same:.3f}/{dist_diff:.3f}")

    ok = (
        base_err < 1e-5 and prob_err < 1e-4 and branch_gap > 1e-5 and pick_err < 1e-5
        and mix_err < 1e-4 and lora_grad > 1e-8 and backbone_grad > 1e-8
        and math.isfinite(loss.item()) and gen_ok and restored and sampled["input_ids"].shape[-1] == 12
        and weight_err < 1e-5 and loaded_k == 3 and loaded_r == 4 and abs(load_args.os_lora_alpha - 16.0) < 1e-6
        and bleu_same > bleu_diff and dist_diff > dist_same
    )
    return ok


def _wasd_kernel(vocab, eps, knn, device):
    emb = F.normalize(torch.randn(vocab, 16, device=device), dim=1)
    cost = 1 - emb @ emb.T
    cost.fill_diagonal_(0.0)
    _, idx = torch.topk(cost, knn + 1, largest=False)
    rows = torch.arange(vocab, device=device).repeat_interleave(knn)
    cols = idx[:, 1:].reshape(-1)
    rows_s = torch.cat([rows, cols, torch.arange(vocab, device=device)])
    cols_s = torch.cat([cols, rows, torch.arange(vocab, device=device)])
    dense = torch.zeros(vocab, vocab, device=device)
    dense[rows_s, cols_s] = torch.exp(-cost[rows_s, cols_s] / eps)
    kernel = dense.to_sparse_coo().coalesce()
    return kernel, kernel.transpose(0, 1).coalesce()


def test_speed():
    torch.manual_seed(0)
    device = "cpu"
    num_tokens, support, num_branches = 128, 64, 3
    cost = torch.randn(num_tokens, support, num_branches)
    log_teacher = torch.log_softmax(torch.randn(num_tokens, support), dim=-1)
    for _ in range(2):
        log_domain_sinkhorn(cost, log_teacher, 0.05, 5)
    start = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        log_domain_sinkhorn(cost, log_teacher, 0.05, 5)
    os_ms = (time.perf_counter() - start) * 1000 / repeats

    # Vocab-scale WASD (a few thousand types; full GPT-2 vocab is larger still).
    vocab, batch, length = 2048, 2, 16
    kernel, kernel_t = _wasd_kernel(vocab, 0.5, 8, device)
    teacher = torch.softmax(torch.randn(batch, length, vocab), dim=-1)
    student = torch.softmax(torch.randn(batch, length, vocab), dim=-1)
    starts = torch.zeros(batch, dtype=torch.long)
    ends = torch.full((batch,), length, dtype=torch.long)
    sparse_sinkhorn_sequence(teacher, student, kernel, kernel_t, starts, ends, epsilon=0.5, max_iter=5, tol=0.0)
    start = time.perf_counter()
    wasd_repeats = 2
    for _ in range(wasd_repeats):
        sparse_sinkhorn_sequence(teacher, student, kernel, kernel_t, starts, ends, epsilon=0.5, max_iter=5, tol=0.0)
    wasd_ms = (time.perf_counter() - start) * 1000 / wasd_repeats
    print(
        f"[5] OS-VDKL Sinkhorn N={num_tokens} M={support} K={num_branches} {os_ms:.2f} ms/step | "
        f"WASD Sinkhorn V={vocab} positions={batch * length} {wasd_ms:.2f} ms/step | "
        f"speedup {wasd_ms / max(os_ms, 1e-9):.1f}x"
    )
    return os_ms < wasd_ms


def main():
    checks = [
        ("sinkhorn", test_sinkhorn_marginals),
        ("gradients", test_projection_and_asymmetry),
        ("wrapper", test_student_wrapper),
        ("speed", test_speed),
    ]
    ok = True
    for name, fn in checks:
        passed = fn()
        print(f"[{name}] {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
