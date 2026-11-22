# -*- coding: utf-8 -*-
"""Text-conditioned prior encoding for PSR.

This module keeps the old rule-based parser as a fallback, but adds a BERT-based
text encoder that produces a compact semantic vector and a set of control
parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import BertModel, BertTokenizer


DEFAULT_BERT_PATH = Path(r"F:\pcat\bert-base-uncased")


@dataclass
class TextPrior:
    threshold_bias: float = 0.0
    allow_multi_component: bool = True
    max_component_num: int = 3
    min_component_area: int = 40
    global_confidence_scale: float = 1.0
    side_mode: str = "bilateral"
    semantic_vector: Optional[np.ndarray] = None
    text_confidence: float = 1.0


def _rule_based_prior(text: str) -> TextPrior:
    t = (text or "").strip().lower()
    if not t:
        return TextPrior()

    threshold_bias = 0.0
    allow_multi_component = True
    max_component_num = 3
    min_component_area = 40
    global_confidence_scale = 1.0
    side_mode = "bilateral"

    if re.search(r"\b(no obvious lesion|negative|normal|clear|unremarkable)\b", t):
        threshold_bias += 0.08
        max_component_num = 1
        min_component_area = 80
        global_confidence_scale = 0.95

    if re.search(r"\b(severe|extensive|diffuse|multifocal)\b", t):
        threshold_bias -= 0.05
        allow_multi_component = True
        max_component_num = 5
        min_component_area = 20
        global_confidence_scale = 1.05

    if re.search(r"\bbilateral\b", t):
        allow_multi_component = True
        max_component_num = max(max_component_num, 4)
        side_mode = "bilateral"

    if re.search(r"\bunilateral\b", t):
        allow_multi_component = False
        max_component_num = 1
        side_mode = "left_or_right"

    if re.search(r"\b(left)\b", t):
        side_mode = "left"
    if re.search(r"\b(right)\b", t):
        side_mode = "right"

    if re.search(r"\b(focal|small|tiny)\b", t):
        threshold_bias -= 0.03
        min_component_area = 15

    if re.search(r"\b(mild|minimal|slight)\b", t):
        threshold_bias += 0.03
        min_component_area = max(min_component_area, 50)
        global_confidence_scale = 0.98

    return TextPrior(
        threshold_bias=float(threshold_bias),
        allow_multi_component=bool(allow_multi_component),
        max_component_num=int(max_component_num),
        min_component_area=int(min_component_area),
        global_confidence_scale=float(global_confidence_scale),
        side_mode=side_mode,
        semantic_vector=None,
        text_confidence=1.0,
    )


@lru_cache(maxsize=1)
def _load_bert(model_path: str):
    path = Path(model_path)
    tokenizer = BertTokenizer.from_pretrained(str(path))
    model = BertModel.from_pretrained(str(path))
    model.eval()
    return tokenizer, model


def encode_text_with_bert(text: str, model_path: str | Path = DEFAULT_BERT_PATH) -> tuple[np.ndarray, float]:
    """Encode text into a pooled semantic vector using local BERT.

    Returns
    -------
    semantic_vector : np.ndarray
        A 768-d float32 vector from the pooled [CLS] representation.
    confidence : float
        A lightweight confidence score derived from non-empty text and token count.
    """
    t = (text or "").strip()
    if not t:
        return np.zeros((768,), dtype=np.float32), 0.0

    tokenizer, model = _load_bert(str(model_path))
    inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=128, padding=False)
    with torch.no_grad():
        outputs = model(**inputs)
    # pooled_output is the [CLS]-based pooled embedding for BERT.
    pooled = outputs.pooler_output
    if pooled is None:
        pooled = outputs.last_hidden_state[:, 0, :]
    vec = pooled[0].detach().cpu().numpy().astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-6)
    token_count = int(inputs["input_ids"].shape[1])
    confidence = float(np.clip(token_count / 32.0, 0.2, 1.0))
    return vec, confidence


def parse_text_prior(text: str, use_bert: bool = True, model_path: str | Path = DEFAULT_BERT_PATH) -> Dict[str, float | int | bool | str | np.ndarray]:
    """Parse a report into control parameters and an optional semantic vector."""
    prior = _rule_based_prior(text)

    if use_bert:
        semantic_vector, confidence = encode_text_with_bert(text, model_path=model_path)
        prior.semantic_vector = semantic_vector
        prior.text_confidence = float(confidence)

        # Use the semantic confidence to mildly modulate global confidence scale.
        prior.global_confidence_scale = float(np.clip(prior.global_confidence_scale * (0.85 + 0.15 * confidence), 0.7, 1.2))

    return {
        "threshold_bias": prior.threshold_bias,
        "allow_multi_component": prior.allow_multi_component,
        "max_component_num": prior.max_component_num,
        "min_component_area": prior.min_component_area,
        "global_confidence_scale": prior.global_confidence_scale,
        "side_mode": prior.side_mode,
        "semantic_vector": prior.semantic_vector,
        "text_confidence": prior.text_confidence,
    }
