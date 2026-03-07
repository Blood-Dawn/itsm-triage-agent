"""
models/finetune/dataset.py
──────────────────────────
Dataset loading, tokenization, and label encoding for the M2 fine-tuning
pipeline.

WHY THIS FILE EXISTS (the big picture):
    Neural networks don't understand text or string labels. They work on
    numbers — specifically, tensors of integers (for token IDs and class
    labels) and floats (for model weights and activations).

    This file is the bridge between our JSONL ticket files and the format
    PyTorch/HuggingFace expects:

        JSONL file
        { "text": "My laptop won't boot...", "category": "hardware", ... }
                        ↓  tokenizer
        { input_ids: [101, 2026, 14924, ...], attention_mask: [1, 1, ...] }
                        ↓  label maps
        { category_labels: 0, priority_labels: 2 }

KEY CONCEPTS:
    - PyTorch Dataset: the standard interface for feeding data into models
    - Tokenizer: converts text → integer token IDs (vocab lookup + special tokens)
    - Label encoding: converts string class names → integer IDs
    - Padding & truncation: makes all sequences the same length for batching
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

from data.schema.ticket import Category, Priority

# ─── LABEL MAPS ───────────────────────────────────────────────────────────────
#
# WHY BUILD LABEL MAPS FROM THE SCHEMA?
#
# We could hard-code {"hardware": 0, "software": 1, ...}, but then if
# someone adds a new Category to the schema, the label maps would be wrong
# and the model would silently train with mismatched IDs — a very hard
# bug to track down.
#
# By deriving the maps from Category and Priority at import time, we get
# a single source of truth: the schema. Any change to the schema
# automatically propagates here. This is the DRY principle in action.
#
# WHY enumerate()?
# enumerate() gives us (index, value) pairs. We use the index as the
# integer ID. The order of items in a Python Enum is the order they were
# defined, which is stable and deterministic — same order every run.
#
# CRITICAL: Once you train a model with these maps, you CANNOT change the
# order of categories/priorities in the schema without retraining. The
# integer IDs are "burned into" the model weights.

CATEGORY_TO_ID: dict[str, int] = {c.value: i for i, c in enumerate(Category)}
ID_TO_CATEGORY: dict[int, str] = {i: c for c, i in CATEGORY_TO_ID.items()}

PRIORITY_TO_ID: dict[str, int] = {p.value: i for i, p in enumerate(Priority)}
ID_TO_PRIORITY: dict[int, str] = {i: p for p, i in PRIORITY_TO_ID.items()}

# These constants let other modules know the output sizes without
# importing the full label maps. Used in model.py to size the linear heads.
NUM_CATEGORIES: int = len(CATEGORY_TO_ID)   # 8
NUM_PRIORITIES:  int = len(PRIORITY_TO_ID)  # 4

# ─── TOKENIZER CONFIG ─────────────────────────────────────────────────────────
#
# WHY distilbert-base-uncased?
#
# DistilBERT is a compressed version of BERT that keeps 97% of BERT's
# performance while being 40% smaller and 60% faster. "base" = 6 layers,
# 768 hidden dims, 66M parameters. "uncased" = all text lowercased before
# tokenization (simpler vocabulary, fine for IT helpdesk text).
#
# Alternative: bert-base-uncased (bigger, slower), roberta-base (better
# on some tasks but larger). DistilBERT is the right trade-off for a
# portfolio project that needs to train in reasonable time.

BASE_MODEL_NAME = "distilbert-base-uncased"

# WHY 128 tokens?
#
# Most IT helpdesk tickets are short — a subject + 1-3 sentence description.
# We measured our generated tickets: 95th percentile is under 100 tokens.
# 128 gives comfortable headroom without wasting compute on padding.
#
# Compare: BERT's absolute maximum is 512 tokens. At 128, our batches
# are 4x cheaper to process than at 512. For classification tasks where
# the key signal is usually in the first sentence, 128 is sufficient.

MAX_LENGTH = 128


# ─── DATASET ──────────────────────────────────────────────────────────────────

class ITSMDataset(Dataset):
    """
    PyTorch Dataset for ITSM ticket dual-label classification.

    Loads all tickets from a JSONL file into memory, tokenizes the text,
    and returns tensors for each example on demand.

    WHY PYTORCH Dataset (not HuggingFace datasets.Dataset)?

        HuggingFace's `datasets` library is excellent for large corpora
        that don't fit in RAM — it memory-maps files from disk. For our
        ~2000 tickets that fit easily in memory, the simpler PyTorch
        Dataset is more transparent and easier to debug.

        We also need fine-grained control over `__getitem__` to return
        two separate label tensors (category_labels and priority_labels).
        The HuggingFace datasets library can do this, but requires more
        boilerplate.

    WHY LOAD EVERYTHING IN __init__?

        Reading lines from disk inside __getitem__ is slow — you'd be
        making a disk read for every single training sample. Loading all
        examples once in __init__ means __getitem__ is pure in-memory
        dict lookups, which is much faster.

        2000 tickets × ~200 bytes = ~400KB. No memory pressure at all.

    Parameters
    ----------
    jsonl_path : Path
        Path to the JSONL file (train.jsonl, val.jsonl, or test.jsonl).
    tokenizer : DistilBertTokenizerFast
        Pre-loaded tokenizer instance. We accept it as a parameter
        (dependency injection) rather than loading inside __init__
        so the caller can share one tokenizer across train/val/test.
    max_length : int
        Token sequence length to pad/truncate to. Default: 128.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict] = []

        # Load all examples upfront.
        # strip() removes the trailing newline on each JSONL line.
        # We skip empty lines (can happen at end of file).
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        """Total number of examples. PyTorch uses this to size the DataLoader."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return one tokenized example as a dict of tensors.

        WHY A DICT (not a tuple)?

            HuggingFace Trainer uses a DataCollatorWithPadding that
            expects a list of dicts, one per example. It collates them
            by key — stacking all "input_ids" tensors, all "attention_mask"
            tensors, etc. The keys in the dict here become keyword arguments
            in the model's forward() call.

            So if you rename "category_labels" here, you must rename
            the parameter in DualHeadDistilBERT.forward() to match.

        WHY squeeze(0)?

            tokenizer() with return_tensors="pt" returns tensors shaped
            [1, max_length] — a batch of size 1. Since __getitem__ returns
            ONE example (not a batch), we squeeze the batch dimension off
            to get shape [max_length]. The DataLoader will re-add the batch
            dimension when it collates N examples.

        WHY padding="max_length" here instead of dynamic padding?

            Dynamic padding (padding to the longest sequence in a batch)
            is slightly more efficient but requires a custom collate function.
            Padding to max_length is simpler, and at 128 tokens the wasted
            compute is negligible for a 2000-ticket dataset.
        """
        ex = self.examples[idx]

        # ── TOKENIZATION ──────────────────────────────────────────────────────
        #
        # The tokenizer does three things:
        #
        # 1. WordPiece tokenization:
        #    "running" → ["running"] (common word, stays whole)
        #    "helpdesk" → ["help", "##desk"] (rare word, split into subwords)
        #    This allows the model to handle any word, even ones not seen
        #    in training, by splitting them into known subword units.
        #
        # 2. Vocabulary lookup:
        #    Each subword is mapped to its integer ID in DistilBERT's
        #    30,522-token vocabulary.
        #
        # 3. Special tokens:
        #    [CLS] (ID 101) is prepended — this is the "classification token".
        #    Its final hidden state is what we use as the document representation.
        #    [SEP] (ID 102) is appended — marks end of input.
        #    [PAD] (ID 0) fills remaining positions up to max_length.

        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",   # Pad short sequences to max_length
            truncation=True,        # Cut sequences longer than max_length
            return_tensors="pt",    # Return PyTorch tensors (not lists)
        )

        return {
            # Shape: [max_length] — integer token IDs
            "input_ids": encoding["input_ids"].squeeze(0),

            # Shape: [max_length] — 1 for real tokens, 0 for [PAD] tokens
            # WHY attention_mask? The model attends over all positions.
            # Without masking, padding tokens (which are meaningless zeros)
            # would pollute the attention scores. The mask tells the model
            # "ignore these positions."
            "attention_mask": encoding["attention_mask"].squeeze(0),

            # Shape: [] (scalar tensor) — integer class ID
            # dtype=torch.long because cross_entropy expects Long (int64) targets
            "category_labels": torch.tensor(
                CATEGORY_TO_ID[ex["category"]], dtype=torch.long
            ),
            "priority_labels": torch.tensor(
                PRIORITY_TO_ID[ex["priority"]], dtype=torch.long
            ),
        }


# ─── CONVENIENCE LOADERS ──────────────────────────────────────────────────────

def load_tokenizer() -> DistilBertTokenizerFast:
    """
    Download (first time) or load (cached) the DistilBERT tokenizer.

    WHY DistilBertTokenizerFast (not DistilBertTokenizer)?

        The "Fast" suffix means the tokenizer is backed by the Rust-based
        `tokenizers` library from HuggingFace. It's 10-100x faster than
        the pure-Python version, which matters when tokenizing batches.

        For our small dataset the speed difference is minor, but using
        Fast tokenizers is best practice — there's no downside.

    WHY call this once and pass the result around?

        Loading a tokenizer downloads/reads the vocabulary files from
        HuggingFace Hub (or local cache). Doing this once and sharing
        the instance across train/val/test datasets avoids redundant
        reads. Tokenizers are stateless after initialization, so sharing
        is safe.
    """
    return DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)


def load_splits(
    data_dir: Path,
    tokenizer: DistilBertTokenizerFast,
    max_length: int = MAX_LENGTH,
) -> tuple[ITSMDataset, ITSMDataset, ITSMDataset]:
    """
    Load train, val, and test datasets from a data directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing train.jsonl, val.jsonl, test.jsonl.
    tokenizer : DistilBertTokenizerFast
        Pre-loaded tokenizer to share across all splits.
    max_length : int
        Sequence length for padding/truncation.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset) : tuple of ITSMDataset

    Raises
    ------
    FileNotFoundError
        If any of the three split files are missing.
    """
    datasets = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the generator first:\n"
                f"  python -m data.generator.gen --n 2000 --seed 42 --output data/raw"
            )
        datasets.append(ITSMDataset(path, tokenizer, max_length))

    train_ds, val_ds, test_ds = datasets
    return train_ds, val_ds, test_ds
