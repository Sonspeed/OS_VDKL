"""Multi-LoRA student branches for OS-VDKL.

K low-rank adapters sit on the final hidden state, just before the shared
output embedding. Branch logit shifts use the factored form

    a = A h,  U = W_head B,  delta z = (alpha / r) a U

so the vocabulary matmul does not have to be repeated from a full d-dimensional
activation for every branch. B is orthogonally initialized (not zero) so the
K branches are not identical at step 0.
"""

import math
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def unwrap_model(model):
    seen = set()
    while hasattr(model, "module") and model.module is not model and id(model) not in seen:
        seen.add(id(model))
        model = model.module
    return model


def head_weight(model):
    model = unwrap_model(model)
    if hasattr(model, "lm_head") and getattr(model.lm_head, "weight", None) is not None:
        return model.lm_head.weight
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if hasattr(base, "lm_head") and getattr(base.lm_head, "weight", None) is not None:
            return base.lm_head.weight
    if hasattr(model, "embed_out") and getattr(model.embed_out, "weight", None) is not None:
        return model.embed_out.weight
    raise AttributeError("OS-VDKL requires the student output embedding (lm_head.weight).")


def init_os_lora_parameters(num_branches, rank, hidden_size, init_scale, device=None, dtype=None):
    """Orthogonal B (across branches) and Kaiming A.

    ``B_cat = [B_1, ..., B_K]`` is drawn with ``orthogonal_(gain=init_scale)``
    and then split, so ``B_i^T B_j = 0`` for ``i != j`` whenever ``d >= K r``.
    """
    if dtype is None:
        dtype = torch.float32
    lora_A = torch.empty(num_branches, rank, hidden_size, device=device, dtype=torch.float32)
    for k in range(num_branches):
        nn.init.kaiming_uniform_(lora_A[k], a=math.sqrt(5))

    b_cat = torch.empty(hidden_size, num_branches * rank, device=device, dtype=torch.float32)
    nn.init.orthogonal_(b_cat, gain=init_scale)
    lora_B = b_cat.view(hidden_size, num_branches, rank).permute(1, 0, 2).contiguous()
    return lora_A.to(dtype=dtype), lora_B.to(dtype=dtype)


def factored_logit_shift(h0, lora_A, lora_B, weight, scale):
    """Fast factored full-vocab shift. ``h0`` is ``[..., d]``; returns ``[..., K, V]``."""
    activation = torch.einsum("...d,krd->...kr", h0.float(), lora_A.float())
    factors = torch.einsum("vd,kdr->krv", weight.float(), lora_B.float())
    return scale * torch.einsum("...kr,krv->...kv", activation, factors)


def hidden_logit_shift(h0, lora_A, lora_B, weight, scale):
    """Same shift via an explicit hidden residual. Used to check the factored form."""
    activation = torch.einsum("...d,krd->...kr", h0.float(), lora_A.float())
    delta_h = scale * torch.einsum("...kr,kdr->...kd", activation, lora_B.float())
    return torch.einsum("...kd,vd->...kv", delta_h, weight.float())


def single_branch_logits(z0, h0, lora_A, lora_B, weight, scale, branch_ids):
    """Raw branch logits ``z_k = z_0 + delta z_k``.

    ``branch_ids`` is an int (every row uses that branch) or a LongTensor of
    shape ``[batch]`` (one branch per sequence).
    """
    if isinstance(branch_ids, int):
        activation = torch.einsum("bld,rd->blr", h0.float(), lora_A[branch_ids].float())
        delta_h = scale * torch.einsum("blr,dr->bld", activation, lora_B[branch_ids].float())
    else:
        ids = branch_ids.to(device=h0.device, dtype=torch.long).view(-1)
        if ids.numel() != h0.size(0):
            raise RuntimeError(
                f"OS-VDKL branch ids have length {ids.numel()} but the batch has {h0.size(0)} rows."
            )
        activation = torch.einsum("bld,krd->blkr", h0.float(), lora_A.float())
        rows = torch.arange(h0.size(0), device=h0.device)
        activation = activation[rows, :, ids, :]
        delta_h = scale * torch.einsum("blr,bdr->bld", activation, lora_B.float()[ids])
    return z0.float() + F.linear(delta_h, weight.float())


def mixture_log_probs(z0, h0, lora_A, lora_B, weight, scale, chunk_tokens=16):
    """Log-mixture ``log Q_bar = logsumexp_k(log_softmax(z_k)) - log K``.

    The vocabulary factors ``U`` are computed once. Time is checkpointed so a
    training-time mixture CE does not keep every token's branch logits.
    """
    num_branches = lora_A.size(0)
    factors = torch.einsum("vd,kdr->krv", weight.float(), lora_B.float())
    activation = torch.einsum("bld,krd->blkr", h0.float(), lora_A.float())
    length = z0.size(1)
    pieces = []

    def _chunk(z_chunk, a_chunk, factors_all):
        mixed = None
        for k in range(num_branches):
            delta = scale * torch.einsum("blr,rv->blv", a_chunk[:, :, k], factors_all[k])
            log_q = torch.log_softmax(z_chunk.float() + delta, dim=-1)
            mixed = log_q if mixed is None else torch.logaddexp(mixed, log_q)
        return mixed - math.log(num_branches)

    grad_enabled = torch.is_grad_enabled() and (
        h0.requires_grad or lora_A.requires_grad or lora_B.requires_grad or weight.requires_grad
    )
    for start in range(0, length, chunk_tokens):
        end = min(length, start + chunk_tokens)
        z_chunk = z0[:, start:end]
        a_chunk = activation[:, start:end]
        if grad_enabled:
            pieces.append(checkpoint(_chunk, z_chunk, a_chunk, factors, use_reentrant=False))
        else:
            pieces.append(_chunk(z_chunk, a_chunk, factors))
    return torch.cat(pieces, dim=1)


def apply_os_logits(model, h0, z0):
    mode = getattr(model, "os_decode_mode", "mixture")
    if mode in (None, "base"):
        return z0
    weight = head_weight(model)
    scale = model.os_scale
    if mode == "branch":
        branch = 0 if model.os_force_branch is None else int(model.os_force_branch)
        logits = single_branch_logits(z0, h0, model.os_lora_A, model.os_lora_B, weight, scale, branch)
    elif mode == "random_branch":
        ids = model.os_branch_ids
        if ids is None:
            ids = torch.randint(0, model.os_lora_A.size(0), (h0.size(0),), device=h0.device)
            model.os_branch_ids = ids
        logits = single_branch_logits(z0, h0, model.os_lora_A, model.os_lora_B, weight, scale, ids)
    else:
        logits = mixture_log_probs(z0, h0, model.os_lora_A, model.os_lora_B, weight, scale)
    return logits.to(dtype=z0.dtype)


def _final_norm(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "transformer") and hasattr(model.transformer, "norm_f"):
        return model.transformer.norm_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise RuntimeError("OS-VDKL could not find the final LayerNorm (transformer.ln_f).")


def _patch_forward(model):
    if getattr(model, "_os_forward_patched", False):
        return
    orig = model.forward
    norm = _final_norm(model)

    def _capture_h0(module, inputs, output):
        model._os_h0 = output[0] if isinstance(output, tuple) else output

    norm.register_forward_hook(_capture_h0)

    def forward(*args, **kwargs):
        return_dict = kwargs.get("return_dict", model.config.use_return_dict)
        kwargs["return_dict"] = True
        model._os_h0 = None
        outputs = orig(*args, **kwargs)
        h0 = model._os_h0
        if h0 is not None:
            outputs.os_h0 = h0
            outputs.logits = apply_os_logits(model, h0, outputs.logits)
        if not return_dict:
            return outputs.to_tuple()
        return outputs

    model.forward = forward
    model._os_forward_patched = True


def _sidecar_path(model_path):
    if not model_path:
        return None
    path = os.path.join(model_path, "os_multi_lora.pt")
    return path if os.path.exists(path) else None


def _load_sidecar(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def os_config_dict(args):
    return {
        "num_branches": int(args.os_num_branches),
        "rank": int(args.os_lora_rank),
        "alpha": float(args.os_lora_alpha),
        "init_scale": float(args.os_init_scale),
        "top_m": int(args.os_top_m),
        "sinkhorn_eps": float(args.os_sinkhorn_eps),
        "sinkhorn_iters": int(args.os_sinkhorn_iters),
        "lambda_local": float(args.os_lambda_local),
        "mu_ortho": float(args.os_mu_ortho),
        "eta_sem": float(args.os_eta_sem),
        "sgo_mode": args.os_sgo_mode,
        "eval_mode": args.os_eval_mode,
    }


def save_os_multi_lora(model, save_dir, args):
    model = unwrap_model(model)
    if not hasattr(model, "os_lora_A"):
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "os_multi_lora.pt")
    blob = {
        "lora_A": model.os_lora_A.detach().float().cpu(),
        "lora_B": model.os_lora_B.detach().float().cpu(),
        "config": os_config_dict(args),
    }
    torch.save(blob, path)
    return path


def attach_os_multi_lora(model, args):
    """Register branch parameters and patch ``forward`` when OS-VDKL is on.

    A checkpoint that already contains ``os_multi_lora.pt`` turns the flag on
    and supplies ``K``, rank, and alpha so evaluation matches training.
    """
    if not hasattr(args, "os_vdkl"):
        return model

    sidecar = _sidecar_path(getattr(args, "model_path", None))
    blob = _load_sidecar(sidecar) if sidecar is not None else None
    if blob is not None:
        args.os_vdkl = True
        cfg = blob.get("config") or {}
        weight_a = blob.get("lora_A")
        if weight_a is not None:
            args.os_num_branches = int(weight_a.shape[0])
            args.os_lora_rank = int(weight_a.shape[1])
        if "alpha" in cfg:
            args.os_lora_alpha = float(cfg["alpha"])

    if not args.os_vdkl:
        return model
    if hasattr(model, "os_lora_A"):
        return model

    ref = next(model.parameters())
    hidden = head_weight(model).shape[1]
    num_branches = int(args.os_num_branches)
    rank = int(args.os_lora_rank)
    lora_A, lora_B = init_os_lora_parameters(
        num_branches, rank, hidden, float(args.os_init_scale), device=ref.device, dtype=ref.dtype,
    )
    model.register_parameter("os_lora_A", nn.Parameter(lora_A))
    model.register_parameter("os_lora_B", nn.Parameter(lora_B))
    if blob is not None:
        model.os_lora_A.data.copy_(blob["lora_A"].to(device=ref.device, dtype=ref.dtype))
        model.os_lora_B.data.copy_(blob["lora_B"].to(device=ref.device, dtype=ref.dtype))

    model.os_scale = float(args.os_lora_alpha) / float(rank)
    model.os_decode_mode = "mixture"
    model.os_branch_ids = None
    model.os_force_branch = None
    _patch_forward(model)
    return model


@contextmanager
def os_decode_context(model, mode, branch_ids=None, force_branch=None):
    model = unwrap_model(model)
    if not hasattr(model, "os_decode_mode"):
        yield model
        return
    previous = (model.os_decode_mode, model.os_branch_ids, model.os_force_branch)
    model.os_decode_mode = mode
    model.os_branch_ids = branch_ids
    model.os_force_branch = force_branch
    try:
        yield model
    finally:
        model.os_decode_mode, model.os_branch_ids, model.os_force_branch = previous
