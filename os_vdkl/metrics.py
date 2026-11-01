"""Lexical diversity metrics for multi-branch OS-VDKL evaluation."""

import math
from collections import Counter


def _tokens(text):
    return text.split()


def _ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _sentence_bleu(hypothesis, references):
    """BLEU-4 with add-one smoothing and a brevity penalty. Score in ``[0, 1]``."""
    if not hypothesis:
        return 0.0
    precisions = []
    for n in range(1, 5):
        hyp = _ngrams(hypothesis, n)
        if not hyp:
            precisions.append(1.0 / (1.0 + n))
            continue
        counts = Counter(hyp)
        ceiling = Counter()
        for ref in references:
            for gram, count in Counter(_ngrams(ref, n)).items():
                if count > ceiling[gram]:
                    ceiling[gram] = count
        overlap = sum(min(count, ceiling[gram]) for gram, count in counts.items())
        precisions.append((overlap + 1.0) / (len(hyp) + 1.0))
    closest = min((len(ref) for ref in references), key=lambda length: (abs(length - len(hypothesis)), length))
    if len(hypothesis) > closest:
        brevity = 1.0
    else:
        brevity = math.exp(1.0 - closest / max(len(hypothesis), 1))
    return brevity * math.exp(sum(math.log(p) for p in precisions) / 4.0)


def cross_branch_self_bleu(branch_texts):
    """Mean sentence BLEU of each branch against the other branches.

    ``branch_texts`` is a list of K sequences of strings, one per branch,
    aligned across the dataset. Returns a score in ``[0, 1]``.
    """
    if not branch_texts or len(branch_texts) < 2:
        return 0.0
    n_ex = len(branch_texts[0])
    scores = []
    for i in range(n_ex):
        toks = [_tokens(branch_texts[k][i]) for k in range(len(branch_texts))]
        for k, hyp in enumerate(toks):
            refs = [toks[j] for j in range(len(toks)) if j != k and len(toks[j]) > 0]
            if len(hyp) == 0 or not refs:
                continue
            scores.append(_sentence_bleu(hyp, refs))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def distinct_n(branch_texts, n=2):
    """Mean per-prompt distinct-n over the K branch responses of that prompt.

    Identical branches repeat the same n-grams, so the score falls as the
    branches collapse. Returns a score in ``[0, 1]``.
    """
    if not branch_texts:
        return 0.0
    n_ex = len(branch_texts[0])
    scores = []
    for i in range(n_ex):
        grams = []
        for texts in branch_texts:
            toks = _tokens(texts[i])
            if len(toks) < n:
                continue
            grams.extend(tuple(toks[j:j + n]) for j in range(len(toks) - n + 1))
        if not grams:
            continue
        scores.append(len(set(grams)) / len(grams))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
