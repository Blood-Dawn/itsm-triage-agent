"""
models/finetune/predict.py
──────────────────────────
Inference wrapper for the fine-tuned DualHeadDistilBERT model (M3).

This is the counterpart to models/baseline/predict.py (M1). Both files
expose the same interface pattern — a predict() function that takes ticket
text and returns a structured result — so the M5 eval harness can call
both backends identically and compare them head-to-head.

WHAT M3 ADDS OVER M1:

    M1 (LLM baseline): sends the ticket to Claude via API, gets back
    natural-language reasoning + classification. ~$0.001/ticket, ~2s latency,
    internet required.

    M3 (fine-tuned): runs the ticket through a local neural network,
    gets back logits + predicted class. ~$0.000/ticket (no API cost),
    ~5ms latency on GPU, runs offline.

    The fine-tuned model doesn't generate reasoning text or next_action
    prose (that's a text generation capability — our classifier just
    picks categories). For next_action, we use a simple lookup table
    from the category, matching what the LLM baseline produces in spirit.

KEY CONCEPT: logits → probabilities → predicted class

    The model's forward pass returns raw logits (unnormalized scores).
    To get probabilities we apply softmax:

        probabilities = softmax(logits, dim=-1)
        predicted_class = argmax(probabilities)
        confidence = probabilities[predicted_class]

    Confidence is the probability assigned to the winning class.
    A high-confidence prediction (0.95+) means the model is sure.
    A low-confidence prediction (0.40 for 8 classes) means it's uncertain
    and you might want to flag that ticket for human review.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import DistilBertTokenizerFast

from models.finetune.dataset import (
    ID_TO_CATEGORY,
    ID_TO_PRIORITY,
    MAX_LENGTH,
)
from models.finetune.model import DualHeadDistilBERT

# ─── NEXT-ACTION LOOKUP ───────────────────────────────────────────────────────
#
# WHY DOES THE FINE-TUNED MODEL NEED A LOOKUP TABLE FOR NEXT ACTIONS?
#
# The classifier learns to predict category and priority, but it doesn't
# generate text. Next-action guidance requires language generation, which is
# the job of an LLM (M1) or a separate generation model.
#
# For M3, we use a simple category → default next_action mapping. This
# gives downstream consumers (like the FastAPI API in M4) something useful
# even from the local model. In production you'd either:
#   a) Chain the classifier with an LLM for next_action generation
#   b) Store a lookup table of approved SOPs per category in a database
#
# We keep one representative action per category — concise, actionable.

CATEGORY_NEXT_ACTIONS: dict[str, str] = {
    "hardware":  "Run diagnostics and schedule hardware inspection or replacement.",
    "software":  "Collect version info and error logs; escalate to software support.",
    "network":   "Check connectivity and escalate to network team with diagnostics.",
    "security":  "Isolate affected system immediately; escalate to security team.",
    "access":    "Verify identity and submit access provisioning request.",
    "email":     "Check mail server logs and escalate to messaging team.",
    "printer":   "Check printer status and dispatch technician if needed.",
    "other":     "Gather additional details and route to appropriate support team.",
}


# ─── RESULT DATACLASS ─────────────────────────────────────────────────────────
#
# WHY A PARALLEL STRUCTURE TO TriageResult IN models/baseline/predict.py?
#
# The M5 eval harness needs to compare M1 and M3 side by side. By giving
# both result types the same field names for the key outputs (category,
# priority, next_action, latency_ms, success, error), the eval code can
# treat them identically — no special-casing per model type.
#
# Fields unique to M3 (cat_confidence, pri_confidence) give richer insight
# into the neural model's certainty, which the LLM baseline can't provide.
# Fields from M1 not present here (cost_usd, reasoning, model) are either
# zero or not applicable for a local model.

@dataclass
class FinetuneResult:
    """
    Structured output from a single fine-tuned model inference call.

    Fields
    ------
    category : str
        Predicted ticket category (e.g., "hardware", "software").
    priority : str
        Predicted ticket priority (e.g., "P1", "P3").
    next_action : str
        Recommended next action (looked up from CATEGORY_NEXT_ACTIONS).
    cat_confidence : float
        Softmax probability for the predicted category (0.0–1.0).
        High values = model is sure. Low values = uncertain, worth reviewing.
    pri_confidence : float
        Softmax probability for the predicted priority (0.0–1.0).
    latency_ms : float
        End-to-end inference time in milliseconds (tokenize + forward pass).
    success : bool
        True if inference completed without error.
    error : str
        Error message if success=False, else empty string.
    """
    category:        str   = ""
    priority:        str   = ""
    next_action:     str   = ""
    cat_confidence:  float = 0.0
    pri_confidence:  float = 0.0
    latency_ms:      float = 0.0
    success:         bool  = False
    error:           str   = ""


# ─── MODULE-LEVEL STATE (lazy loading) ────────────────────────────────────────
#
# WHY LAZY LOADING AT MODULE LEVEL?
#
# Loading a model (DistilBERT + LoRA weights) takes ~1-2 seconds. If we
# loaded it inside predict() on every call, a batch of 200 tickets would
# waste 200×1.5s = 5 minutes just loading the model repeatedly.
#
# Instead, we load once the first time predict() is called and cache the
# model in module-level variables (_model, _tokenizer). On subsequent
# calls, the check `if _model is None` is False and we skip the load.
#
# This pattern is called "lazy initialization" or "memoized initialization".
# It's the same pattern used in models/baseline/predict.py for the
# Anthropic client, and it's the right approach for expensive resources.
#
# The alternative — loading at import time — is wrong because it would
# crash during import if the adapter directory doesn't exist yet (e.g.,
# before training has been run).

_model:     Optional[DualHeadDistilBERT]      = None
_tokenizer: Optional[DistilBertTokenizerFast] = None
_device:    Optional[torch.device]            = None
_adapter_dir_loaded: Optional[Path]           = None  # track which adapter is loaded


def _load_model(adapter_dir: Path) -> None:
    """
    Load the fine-tuned model into module-level cache.

    Called automatically on the first predict() call.
    Safe to call multiple times — only reloads if adapter_dir changed.

    Parameters
    ----------
    adapter_dir : Path
        Directory produced by the M2 training run, containing
        adapter weights and tokenizer files.
    """
    global _model, _tokenizer, _device, _adapter_dir_loaded

    # If already loaded from the same directory, skip
    if _model is not None and _adapter_dir_loaded == adapter_dir:
        return

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}\n"
            f"Run training first:\n"
            f"  python -m models.finetune.train --epochs 3 --batch-size 16"
        )

    logger.info(f"Loading fine-tuned model from {adapter_dir}")
    t0 = time.perf_counter()

    # Device selection: prefer GPU if available
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (LoRA encoder + classification heads)
    _model = DualHeadDistilBERT.from_pretrained(adapter_dir)
    _model = _model.to(_device)

    # Load tokenizer from the adapter directory
    # (we saved a copy there during training for self-contained deployment)
    _tokenizer = DistilBertTokenizerFast.from_pretrained(str(adapter_dir))

    _adapter_dir_loaded = adapter_dir

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Model loaded in {elapsed:.0f}ms on {_device}")


# ─── PREDICT ──────────────────────────────────────────────────────────────────

def predict(
    ticket_text: str,
    adapter_dir: Path,
) -> FinetuneResult:
    """
    Run inference on a single ticket using the fine-tuned model.

    Parameters
    ----------
    ticket_text : str
        Raw ticket text (same format as the training data).
    adapter_dir : Path
        Path to the saved adapter directory.

    Returns
    -------
    FinetuneResult
        Structured result with prediction, confidence, and latency.
        On error, result.success=False and result.error has details.
    """
    # Ensure model is loaded (lazy initialization)
    try:
        _load_model(adapter_dir)
    except Exception as e:
        return FinetuneResult(success=False, error=str(e))

    t_start = time.perf_counter()

    try:
        # ── TOKENIZE ──────────────────────────────────────────────────────────
        #
        # Same tokenization as training (same max_length, same special tokens).
        # return_tensors="pt" gives PyTorch tensors.
        # We call .to(_device) to move the token tensors to GPU if available.

        encoding = _tokenizer(
            ticket_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(_device)
        attention_mask = encoding["attention_mask"].to(_device)

        # ── FORWARD PASS ──────────────────────────────────────────────────────
        #
        # torch.no_grad(): disables gradient tracking.
        #
        # WHY? During training, PyTorch builds a computational graph as you
        # call forward(). This graph is used by loss.backward() to compute
        # gradients. At inference time, we'll never call backward(), so
        # building the graph wastes memory and compute.
        #
        # torch.no_grad() tells PyTorch: "don't build the graph for this
        # block." It's the inference equivalent of model.eval() — both are
        # required for correct, efficient inference.

        with torch.no_grad():
            outputs = _model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # No labels — inference mode, no loss computed
            )

        # ── DECODE PREDICTIONS ─────────────────────────────────────────────────
        #
        # outputs.cat_logits: [1, 8]  (batch size 1 for single ticket)
        # outputs.pri_logits: [1, 4]
        #
        # softmax converts logits → probabilities that sum to 1.0
        # argmax finds the index of the highest probability = predicted class
        # .item() converts a single-element tensor to a Python float/int

        cat_probs = F.softmax(outputs.cat_logits, dim=-1)   # [1, 8]
        pri_probs = F.softmax(outputs.pri_logits, dim=-1)   # [1, 4]

        cat_id = cat_probs.argmax(dim=-1).item()            # int 0-7
        pri_id = pri_probs.argmax(dim=-1).item()            # int 0-3

        cat_conf = cat_probs[0, cat_id].item()              # float 0-1
        pri_conf = pri_probs[0, pri_id].item()              # float 0-1

        # Convert integer IDs back to string labels using our reverse maps
        category = ID_TO_CATEGORY[cat_id]
        priority = ID_TO_PRIORITY[pri_id]
        next_action = CATEGORY_NEXT_ACTIONS[category]

        latency_ms = (time.perf_counter() - t_start) * 1000

        logger.debug(
            f"Predicted: {category}/{priority}  "
            f"conf={cat_conf:.2f}/{pri_conf:.2f}  "
            f"latency={latency_ms:.1f}ms"
        )

        return FinetuneResult(
            category=category,
            priority=priority,
            next_action=next_action,
            cat_confidence=round(cat_conf, 4),
            pri_confidence=round(pri_conf, 4),
            latency_ms=round(latency_ms, 1),
            success=True,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.error(f"Inference failed: {e}")
        return FinetuneResult(success=False, error=str(e), latency_ms=latency_ms)


def predict_batch(
    ticket_texts: list[str],
    adapter_dir: Path,
    show_progress: bool = True,
) -> list[FinetuneResult]:
    """
    Run inference on a list of tickets.

    Uses batched inference (all tickets tokenized together) for efficiency.
    This is faster than calling predict() in a loop because GPU matrix
    multiplications are highly parallelizable — processing 16 tickets at
    once takes roughly the same time as processing 1.

    WHY BATCHED INFERENCE?

        Single-ticket inference: ~5ms × 200 = 1000ms total
        Batched inference (batch=32): ~12ms × 7 batches = 84ms total

        The GPU is "embarrassingly parallel" — sending more work per call
        is almost free up to the VRAM limit. This is why batch size
        matters so much in ML.

    Parameters
    ----------
    ticket_texts : list[str]
        List of raw ticket texts.
    adapter_dir : Path
        Path to the saved adapter directory.
    show_progress : bool
        Print progress every 50 tickets.

    Returns
    -------
    list[FinetuneResult]
        One result per input ticket, in the same order.
    """
    try:
        _load_model(adapter_dir)
    except Exception as e:
        return [FinetuneResult(success=False, error=str(e))] * len(ticket_texts)

    results: list[FinetuneResult] = []
    BATCH_SIZE = 32   # Number of tickets to process per GPU call

    t_total_start = time.perf_counter()

    for batch_start in range(0, len(ticket_texts), BATCH_SIZE):
        batch_texts = ticket_texts[batch_start : batch_start + BATCH_SIZE]

        t_batch = time.perf_counter()

        # Tokenize the entire batch at once
        # padding=True: pad each sequence to the longest in THIS batch
        # (more efficient than padding to max_length when batches are short)
        encoding = _tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(_device)
        attention_mask = encoding["attention_mask"].to(_device)

        with torch.no_grad():
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask)

        cat_probs = F.softmax(outputs.cat_logits, dim=-1)   # [batch, 8]
        pri_probs = F.softmax(outputs.pri_logits, dim=-1)   # [batch, 4]

        cat_ids = cat_probs.argmax(dim=-1).tolist()          # list[int]
        pri_ids = pri_probs.argmax(dim=-1).tolist()          # list[int]

        batch_latency_ms = (time.perf_counter() - t_batch) * 1000
        per_ticket_ms    = batch_latency_ms / len(batch_texts)

        for i, (cat_id, pri_id) in enumerate(zip(cat_ids, pri_ids)):
            category    = ID_TO_CATEGORY[cat_id]
            priority    = ID_TO_PRIORITY[pri_id]
            next_action = CATEGORY_NEXT_ACTIONS[category]
            cat_conf    = cat_probs[i, cat_id].item()
            pri_conf    = pri_probs[i, pri_id].item()

            results.append(FinetuneResult(
                category=category,
                priority=priority,
                next_action=next_action,
                cat_confidence=round(cat_conf, 4),
                pri_confidence=round(pri_conf, 4),
                latency_ms=round(per_ticket_ms, 1),
                success=True,
            ))

        if show_progress and (batch_start + BATCH_SIZE) % 50 == 0:
            done = min(batch_start + BATCH_SIZE, len(ticket_texts))
            logger.info(f"  Processed {done}/{len(ticket_texts)} tickets...")

    total_ms = (time.perf_counter() - t_total_start) * 1000
    successes = sum(r.success for r in results)
    logger.info(
        f"Batch inference done: {successes}/{len(results)} succeeded  "
        f"| Total: {total_ms:.0f}ms  "
        f"| Avg: {total_ms/len(results):.1f}ms/ticket"
    )

    return results
