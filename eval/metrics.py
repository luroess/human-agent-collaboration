from __future__ import annotations

import re
from typing import Iterable, Optional

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def exact_match(pred: str, ref: Optional[str]) -> float:
    if ref is None:
        return 0.0
    return 1.0 if pred.strip() == ref.strip() else 0.0


def token_f1(pred: str, ref: Optional[str]) -> float:
    if ref is None:
        return 0.0
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token in a:
        prev = 0
        for j, b_token in enumerate(b, start=1):
            temp = dp[j]
            if token == b_token:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l(pred: str, ref: Optional[str]) -> float:
    if ref is None:
        return 0.0
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    beta = 1.2
    denom = recall + (beta * beta * precision)
    if denom == 0:
        return 0.0
    return ((1 + beta * beta) * precision * recall) / denom


_BERTSCORE = None
_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def bertscore_f1(pred: str, ref: Optional[str], model_type: str = "bert-base-uncased") -> float:
    if ref is None:
        return 0.0
    # Offline-safe fallback: return 0 if BERTScore isn't available.
    try:
        import evaluate  # noqa: WPS433
    except Exception:
        return 0.0
    global _BERTSCORE
    if _BERTSCORE is None:
        _BERTSCORE = evaluate.load("bertscore")
    scores = _BERTSCORE.compute(predictions=[pred], references=[ref], model_type=model_type)
    return float(scores["f1"][0])


def evidence_coverage(pred: str, evidence: Optional[Iterable[str]]) -> float:
    if not evidence:
        return 0.0
    evidence_list = list(evidence)
    pred_lower = pred.lower()
    covered = sum(1 for item in evidence_list if str(item).lower() in pred_lower)
    return covered / len(evidence_list)


def semantic_similarity(
    pred: str,
    ref: Optional[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    if ref is None:
        return 0.0
    embedder = _EMBEDDER_CACHE.get(model_name)
    if embedder is None:
        try:
            embedder = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            return 0.0
        _EMBEDDER_CACHE[model_name] = embedder
    vectors = embedder.encode([pred, ref], normalize_embeddings=True)
    return float(cos_sim(vectors[0], vectors[1]))


def constraint_adherence(pred: str, constraints: Iterable[str]) -> float:
    constraints_list = list(constraints)
    if not constraints_list:
        return 0.0
    matched = 0
    for constraint in constraints_list:
        if constraint.startswith("Use exactly"):
            match = re.search(r"Use exactly (\d+) bullet", constraint)
            if match:
                expected = int(match.group(1))
                bullets = len(re.findall(r"^\s*[-*]\s+", pred, re.MULTILINE))
                if bullets == expected:
                    matched += 1
        elif constraint.startswith("Mention the keyword"):
            keyword = constraint.split("'")[1]
            if keyword in pred:
                matched += 1
        elif constraint.startswith("Avoid the word"):
            word = constraint.split("'")[1]
            if word not in pred:
                matched += 1
    return matched / len(constraints_list)
