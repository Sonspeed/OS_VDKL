import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def ab_div(logits, teacher_logits, no_model_batch, alpha, beta):
    """Calculate D^{(alpha, beta)} divergence."""
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    eps = 1e-8

    if abs(alpha) < eps and abs(beta) < eps:
        # Case 1: alpha = 0, beta = 0
        divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)

    elif abs(alpha) < eps:
        # Case 2: alpha = 0, beta != 0
        safe_log_ratio_div_beta = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(torch.exp(beta * log_q) * (beta * safe_log_ratio_div_beta - 1) + torch.exp(beta * log_p), dim=-1) / (beta ** 2)

    elif abs(beta) < eps:
        # Case 3: beta = 0, alpha != 0
        safe_log_ratio_div_alpha = torch.where(torch.isfinite(log_p - log_q), log_p - log_q, 0.0)
        divergence = torch.sum(torch.exp(alpha * log_p) * (alpha * safe_log_ratio_div_alpha - 1) + torch.exp(alpha * log_q), dim=-1) / (alpha ** 2)

    elif abs(alpha + beta) < eps:
        # Case 4: alpha + beta = 0
        safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1, dim=-1) / (alpha ** 2)

    else:
        # General case
        apb = alpha + beta
        term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
        term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
        term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
        divergence = - (term1 - term2 - term3) / (alpha * beta)
        
    mask = (no_model_batch["label"] != -100).float()
    
    # Replace any NaN/inf values in the divergence with 0.0 for safe masking
    safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
    
    # Apply the mask and compute the final mean loss
    masked_sum = (safe_divergence * mask).sum()
    mask_sum = mask.sum()
    
    # Avoid division by zero if the mask is empty
    loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
    
    return loss



def bdkd(logits, teacher_logits, no_model_batch):
    def entropy(logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        return -torch.sum(probs * log_probs, dim=-1)

    entropy_student = entropy(logits)  # (B, S)
    entropy_teacher = entropy(teacher_logits)  # (B, S)

    weight_student = torch.where(entropy_student > entropy_teacher, 2.0, 1.0)
    weight_teacher = torch.where(entropy_teacher > entropy_student, 2.0, 1.0)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss1 = -torch.sum(x * mask.view(-1) * weight_teacher.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss2 = -torch.sum(x * mask.view(-1) * weight_student.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss1 + distil_loss2


def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def get_ratio(teacher_logits, logits, mu=0.5):
    # [B, L, V]
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)  # B,L,V
    mask = cum_sum > mu
    mask[:, :, 0] = False

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)

    return s1 / (s1 + s2), s2 / (s1 + s2)


def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    # ratio: [B,L]
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)  # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x - x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum((teacher_x - x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1),
                                                                                                     dim=0)
    return distil_loss


def AKL(teacher_logits, logits, no_model_batch):
    inf_mask = torch.isinf(logits)  # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int()  # [batch, seq]

    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(logits, teacher_logits, inf_mask,
                                                                                   mask, l_ratio)
    return distil_loss




def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1 - lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss


def wsd(logits, teacher_logits, no_model_batch, lam=0.5):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1 - lam) * for_kl + lam * rev_kl

    return distil_loss


def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    # assert isinstance(alpha, float)
    inf_mask = torch.isinf(q_logits) | torch.isinf(p_logits)
    q_logits = torch.masked_fill(q_logits, inf_mask, 0)
    p_logits = torch.masked_fill(p_logits, inf_mask, 0)
    q_prob = torch.nn.functional.softmax(q_logits, dim=-1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=-1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=-1)  # gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=-1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=-1)
    return loss, grad_loss


def alphanet(logits, teacher_logits, no_model_batch, alpha, beta):
    loss1 = ab_div(logits, teacher_logits, no_model_batch, alpha, 1 - alpha)
    loss2 = ab_div(logits, teacher_logits, no_model_batch, beta, 1 - beta)
    if loss1 > loss2:
        return loss1
    return loss2


@torch.no_grad()
def sparse_sinkhorn_sequence(
        pi_r,          # (B, T, N)
        pi_theta,      # (B, T, N)
        K,             # (N, N) sparse kernel exp(-C / epsilon)
        Kt,            # transpose of K
        starts: torch.Tensor,
        ends: torch.Tensor,
        epsilon: float = 1.0,
        max_iter: int = 10,
        tol: float = 1e-6,
        chunk_rows: int = 1024,
):
    B, T, N = pi_r.shape
    M = B * T
    device = pi_r.device
    dtype = K.dtype

    tiny = torch.finfo(dtype).tiny
    eps_damp = 1e-12

    pi_r_flat = pi_r.reshape(M, N).to(dtype=dtype)
    pi_theta_flat = pi_theta.reshape(M, N).to(dtype=dtype)

    u = pi_theta_flat.clone()
    v = pi_r_flat.clone()

    active_mask = torch.zeros(M, dtype=torch.bool, device=device)
    for b in range(B):
        s = int(starts[b].item())
        e = min(T, int(ends[b].item()))
        if s < e:
            active_mask[b * T + s:b * T + e] = True

    for _ in range(max_iter):
        active_all = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
        if active_all.numel() == 0:
            break
        for active in active_all.split(chunk_rows):
            u_act = u[active]
            v_act = v[active]

            pi_theta_act = pi_theta_flat[active]
            pi_r_act = pi_r_flat[active]

            support_theta = pi_theta_act > 0
            support_r = pi_r_act > 0

            Kv = torch.sparse.mm(K, v_act.t()).t().clamp_min(tiny)
            u_new = torch.where(support_theta, pi_theta_act / Kv, torch.zeros_like(u_act))

            Ktu = torch.sparse.mm(Kt, u_new.t()).t().clamp_min(tiny)
            v_new = torch.where(support_r, pi_r_act / Ktu, torch.zeros_like(v_act))

            rel_u = (u_new - u_act).abs() / torch.maximum(u_new.abs(), u_act.abs()).clamp_min(eps_damp)
            rel_v = (v_new - v_act).abs() / torch.maximum(v_new.abs(), v_act.abs()).clamp_min(eps_damp)
            delta = torch.maximum(rel_u.max(dim=1).values, rel_v.max(dim=1).values)

            u[active] = u_new
            v[active] = v_new

            active_mask[active[delta <= tol]] = False

    f = u.clamp_min_(tiny).log_().mul_(epsilon)
    f.masked_fill_(pi_theta_flat <= 0, 0.0)
    f_labels = f.reshape(B, T, N)

    torch.cuda.empty_cache()
    return f_labels


@torch.no_grad()
def sparse_sinkhorn_self_sequence(
        p,             # (B, T, N)
        K,             # (N, N) sparse kernel exp(-C / epsilon), symmetric
        starts: torch.Tensor,
        ends: torch.Tensor,
        epsilon: float = 1.0,
        max_iter: int = 10,
        tol: float = 1e-6,
        chunk_rows: int = 1024,
):
    eta = 0.5
    B, T, N = p.shape
    M = B * T
    device = p.device
    tiny = 1e-16

    p_flat = p.view(M, N)

    a = torch.ones(M, N, device=device)

    active_list = [
        torch.arange(starts[b].item(), min(T, ends[b].item()), device=device) + b * T
        for b in range(B)
    ]
    active = torch.cat(active_list, dim=0) if len(active_list) > 0 else torch.empty(0, dtype=torch.long, device=device)

    for _ in range(max_iter):
        if active.numel() == 0:
            break
        keep = []
        for act in active.split(chunk_rows):
            a_act = a[act]
            s = torch.sparse.mm(K, (p_flat[act] * a_act).t()).t()
            a_prop = 1.0 / (s + tiny)
            a_new = torch.exp(
                (1 - eta) * torch.log(a_act + tiny) +
                eta * torch.log(a_prop + tiny)
            )
            delta = ((a_new - a_act).abs() / torch.maximum(a_new.abs(), a_act.abs()).clamp_min(tiny)).max(dim=1).values
            a[act] = a_new
            keep.append(act[delta > tol])
            del a_act, s, a_prop, a_new, delta
        active = torch.cat(keep) if keep else active[:0]

    f_sub = epsilon * torch.log(a + tiny)
    f_labels = f_sub.view(B, T, N)

    del a, f_sub
    torch.cuda.empty_cache()

    return f_labels


def amid(logits, teacher_logits, no_model_batch, args, mat_K=None, mat_Kt=None, student_temp=1.0, **kwargs):

    # Teacher and Studnet distribution
    logits = logits / student_temp

    p = F.softmax(teacher_logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    logp = F.log_softmax(teacher_logits, dim=-1)
    logq = F.log_softmax(logits, dim=-1)


    # Basic
    alpha = args.amid_alpha
    lam = args.amid_lam
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)


    # Assistant distribution
    if lam <= 0.0:
        r = q
        logr = F.log_softmax(logits, dim=-1)
    elif lam >= 1.0:
        r = p
        logr = F.log_softmax(teacher_logits, dim=-1)
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
        else:
            t1 = math.log(lam)       + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq

            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
    del alpha, lam


    # Divergence
    div_name = args.amid_div_name
    div_order = args.amid_div_order

    if div_name == "fkl":
        if div_order == "pr":
            prod_probs = torch.masked_fill(p * (logp - logr), inf_mask, 0)
        elif div_order == "qr":
            prod_probs = torch.masked_fill(q * (logq - logr), inf_mask, 0)
        elif div_order == "rp":
            prod_probs = torch.masked_fill(r * (logr - logp), inf_mask, 0)
        elif div_order == "rq":
            prod_probs = torch.masked_fill(r * (logr - logq), inf_mask, 0)
        else:
            raise ValueError(f"Unknown amid-div-order: {div_order}")
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    elif div_name == "ab":
        # ab_alpha, ab_beta = args.ab_alpha, args.ab_beta
        ab_alpha, ab_beta = 0.2, 0.7
        apb = ab_alpha + ab_beta

        if div_order == "pr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logp + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "qr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logq + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "rp":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logp, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
        elif div_order == "rq":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logq, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
        else:
            raise ValueError(f"Unknown amid-div-order: {div_order}")
        divergence = - (term1 - term2 - term3) / (ab_alpha * ab_beta)

        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)

        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
        return loss

    elif div_name in ("wass", "sink"):
        # wass: entropy-regularized Wasserstein distance (ablation)
        # sink: Sinkhorn divergence (WASD)
        if div_order != "pr":
            raise NotImplementedError(f"amid-div-name '{div_name}' supports amid-div-order 'pr' only")
        if mat_K is None or mat_Kt is None:
            raise ValueError("Sparse cost kernel (mat_K, mat_Kt) is required for 'wass'/'sink'")

        starts = mask.float().argmax(dim=1)
        ends = mask.size(1) - mask.flip(dims=[1]).float().argmax(dim=1)

        f_pr = sparse_sinkhorn_sequence(
            p, r.detach(), mat_K, mat_Kt, starts, ends,
            epsilon=args.wd_epsilon, max_iter=args.wd_max_iter, tol=args.wd_tol,
            chunk_rows=args.wd_sinkhorn_chunk,
        )
        if div_name == "sink":
            f_rr = sparse_sinkhorn_self_sequence(
                r.detach(), mat_K, starts, ends,
                epsilon=args.wd_epsilon, max_iter=args.wd_max_iter, tol=args.wd_tol,
                chunk_rows=args.wd_sinkhorn_chunk,
            )
            divergence = r * (f_pr - f_rr)
        else:
            divergence = r * f_pr

        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
        safe_divergence = safe_divergence.sum(dim=-1)
        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
        return args.wd_lambda * loss

    else:
        raise ValueError(f"Unknown amid-div-name: {div_name}")


_OS_VDKL_METRICS = {}


def last_os_vdkl_metrics():
    """Most recent OS-VDKL diagnostics (python floats), empty if none yet."""
    return dict(_OS_VDKL_METRICS)


def log_domain_sinkhorn(cost, log_target, eps, iters):
    """Log-domain Sinkhorn on an ``M x K`` cost.

    ``cost`` is ``[N, M, K]`` and ``log_target`` is ``log P_hat`` with shape
    ``[N, M]`` or ``[N, M, 1]``. Returns the row-stochastic plan ``gamma``
    (``sum_k gamma = 1``). The column update enforces
    ``sum_y P(y) gamma_k(y) = 1/K`` at the fixed point.
    """
    if log_target.dim() == 2:
        log_target = log_target.unsqueeze(-1)
    phi = -cost / eps
    log_k = math.log(cost.size(-1))
    for _ in range(int(iters)):
        phi = phi - torch.logsumexp(phi, dim=2, keepdim=True)
        phi = phi - torch.logsumexp(phi + log_target, dim=1, keepdim=True) - log_k
    return torch.softmax(phi, dim=2)


def subspace_ortho_penalty(lora_B):
    """``sum_{i < j} || normalize(B_i)^T normalize(B_j) ||_F^2``."""
    if lora_B.size(0) < 2:
        return lora_B.float().sum() * 0.0
    normalized = F.normalize(lora_B.float(), p=2, dim=1)
    gram = torch.einsum("idr,jds->ijrs", normalized, normalized)
    pairs = torch.triu_indices(lora_B.size(0), lora_B.size(0), offset=1, device=lora_B.device)
    return (gram[pairs[0], pairs[1]] ** 2).sum()


def _dist_mean(value):
    tensor = value.detach().float().reshape(())
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


def _store_os_metrics(metrics):
    global _OS_VDKL_METRICS
    stored = {}
    for key, value in metrics.items():
        stored[key] = float(_dist_mean(value).item())
    _OS_VDKL_METRICS = stored


def _os_token_losses(h0, teacher_logits, lora_A, lora_B, weight, factors, scale, temp, top_m, sink_eps, sink_iters, eta_sem):
    """Global FKL plus decoupled local RKL on one chunk of active tokens."""
    hidden = h0.float()
    teacher = teacher_logits.detach().float()
    branches_a = lora_A.float()
    branches_b = lora_B.float()
    head = weight.float()
    factors = factors.float()
    num_tokens = hidden.size(0)
    num_branches = branches_a.size(0)
    vocab = teacher.size(-1)
    temperature = temp if temp > 0 else 1.0
    support = max(1, min(int(top_m), vocab))

    teacher_probs = torch.softmax(teacher / temperature, dim=-1)
    base_logits = F.linear(hidden, head)
    activation = torch.einsum("nd,krd->nkr", hidden, branches_a)
    branch_logits = base_logits.unsqueeze(1) + scale * torch.einsum("nkr,krv->nkv", activation, factors)
    log_branch = torch.log_softmax(branch_logits / temperature, dim=-1)
    log_mixture = torch.logsumexp(log_branch, dim=1) - math.log(num_branches)
    global_sum = -(teacher_probs * log_mixture).sum(dim=-1).sum()

    with torch.no_grad():
        entropy_branch = -(log_branch.exp() * log_branch).sum(dim=-1)
        entropy_mixture = -(log_mixture.exp() * log_mixture).sum(dim=-1)
        gain_sum = (entropy_mixture - entropy_branch.mean(dim=-1)).sum()

    top_vals, top_idx = torch.topk(teacher_probs, k=support, dim=-1)
    teacher_hat = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    activation_det = torch.einsum("nd,krd->nkr", hidden.detach(), branches_a)
    delta_h = scale * torch.einsum("nkr,kdr->nkd", activation_det, branches_b)
    head_support = head.detach()[top_idx]
    base_support = torch.gather(base_logits.detach(), 1, top_idx)
    local_logits = base_support.unsqueeze(2) + torch.einsum("nmd,nkd->nmk", head_support, delta_h)
    log_local = torch.log_softmax(local_logits / temperature, dim=1)
    local_probs = log_local.exp()

    with torch.no_grad():
        cost = -log_local
        if eta_sem > 0:
            head_norm = F.normalize(head_support.float(), p=2, dim=-1)
            prototype = torch.einsum("nmk,nmd->nkd", local_probs, head_norm)
            prototype = F.normalize(prototype, p=2, dim=-1)
            cost = cost + eta_sem * (1.0 - torch.einsum("nmd,nkd->nmk", head_norm, prototype))
        log_teacher_hat = torch.log(teacher_hat.clamp_min(1e-12))
        gamma = log_domain_sinkhorn(cost, log_teacher_hat, sink_eps, sink_iters)
        joint = gamma * teacher_hat.unsqueeze(-1)
        local_target = joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if num_branches < 2:
            tv_sum = local_probs.new_zeros(())
        else:
            tv_acc = local_probs.new_zeros(())
            pairs = 0
            for i in range(num_branches):
                for j in range(i + 1, num_branches):
                    tv_acc = tv_acc + (0.5 * (local_probs[..., i] - local_probs[..., j]).abs().sum(dim=-1)).sum()
                    pairs += 1
            tv_sum = tv_acc / pairs

    local_sum = (local_probs * (log_local - torch.log(local_target.clamp_min(1e-12)))).sum()
    return global_sum, local_sum, gain_sum, tv_sum


def os_vdkl_loss_from_hidden(h0, teacher_logits, lora_A, lora_B, weight, args):
    """OS-VDKL on active tokens.

    ``h0`` is ``[N, d]`` and ``teacher_logits`` is ``[N, V]``. ``L_local`` is
    computed from ``h0.detach()`` and ``W_head.detach()``, so it updates only
    the branch factors.
    """
    num_tokens = h0.size(0)
    if num_tokens == 0:
        zero = lora_A.float().sum() * 0.0
        _store_os_metrics({
            "l_global": zero, "l_local": zero, "r_ortho": zero,
            "entropy_gain": zero, "mean_tv": zero,
        })
        return {"loss": zero, "l_global": zero, "l_local": zero, "r_ortho": zero,
                "entropy_gain": zero, "mean_tv": zero}

    scale = float(args.os_lora_alpha) / float(args.os_lora_rank)
    temp = float(getattr(args, "student_temp", 1.0))
    chunk_tokens = max(1, int(getattr(args, "os_chunk_tokens", 32)))
    factors = torch.einsum("vd,kdr->krv", weight.float(), lora_B.float())

    global_sum = torch.zeros((), device=h0.device, dtype=torch.float32)
    local_sum = torch.zeros((), device=h0.device, dtype=torch.float32)
    gain_sum = torch.zeros((), device=h0.device, dtype=torch.float32)
    tv_sum = torch.zeros((), device=h0.device, dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled() and (
        h0.requires_grad or lora_A.requires_grad or lora_B.requires_grad or weight.requires_grad
    )

    def _run(h_chunk, t_chunk, a_param, b_param, w_param, u_param):
        return _os_token_losses(
            h_chunk, t_chunk, a_param, b_param, w_param, u_param, scale, temp,
            args.os_top_m, args.os_sinkhorn_eps, args.os_sinkhorn_iters, args.os_eta_sem,
        )

    for start in range(0, num_tokens, chunk_tokens):
        end = min(num_tokens, start + chunk_tokens)
        inputs = (h0[start:end], teacher_logits[start:end], lora_A, lora_B, weight, factors)
        if use_ckpt and (end - start) < num_tokens:
            g_sum, l_sum, e_sum, t_sum = checkpoint(_run, *inputs, use_reentrant=False)
        else:
            g_sum, l_sum, e_sum, t_sum = _run(*inputs)
        global_sum = global_sum + g_sum
        local_sum = local_sum + l_sum
        gain_sum = gain_sum + e_sum
        tv_sum = tv_sum + t_sum

    l_global = global_sum / num_tokens
    l_local = local_sum / (num_tokens * lora_A.size(0))
    r_ortho = subspace_ortho_penalty(lora_B)
    loss = l_global + float(args.os_lambda_local) * l_local + float(args.os_mu_ortho) * r_ortho
    entropy_gain = gain_sum / num_tokens
    mean_tv = tv_sum / num_tokens
    parts = {
        "loss": loss,
        "l_global": l_global,
        "l_local": l_local,
        "r_ortho": r_ortho,
        "entropy_gain": entropy_gain,
        "mean_tv": mean_tv,
    }
    _store_os_metrics({k: parts[k] for k in ("l_global", "l_local", "r_ortho", "entropy_gain", "mean_tv")})
    return parts


def os_vdkl_loss(args, logits_or_outputs, teacher_logits, no_model_batch, student_model=None):
    """Full-vocabulary mixture FKL + top-M dual-KL local RKL + Stiefel penalty."""
    from os_vdkl.multi_lora import head_weight, unwrap_model

    model = unwrap_model(student_model)
    if torch.is_tensor(logits_or_outputs):
        hidden = getattr(model, "_os_h0", None) if model is not None else None
        if hidden is None:
            raise RuntimeError("OS-VDKL needs the student final hidden state h0.")
    else:
        hidden = getattr(logits_or_outputs, "os_h0", None)
        if hidden is None and getattr(logits_or_outputs, "hidden_states", None):
            hidden = logits_or_outputs.hidden_states[-1]
        if hidden is None and model is not None:
            hidden = getattr(model, "_os_h0", None)
        if hidden is None:
            raise RuntimeError("OS-VDKL needs the student final hidden state h0.")

    labels = no_model_batch["label"]
    length = min(hidden.size(1), teacher_logits.size(1), labels.size(1))
    hidden = hidden[:, :length]
    teacher = teacher_logits[:, :length]
    mask = labels[:, :length] != -100
    parts = os_vdkl_loss_from_hidden(
        hidden[mask], teacher[mask].float(), model.os_lora_A, model.os_lora_B, head_weight(model), args,
    )
    return parts["loss"]
