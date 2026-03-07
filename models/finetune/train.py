"""
models/finetune/train.py
────────────────────────
M2 training pipeline: fine-tunes DualHeadDistilBERT + LoRA on the synthetic
ITSM dataset using HuggingFace Trainer.

HOW TO RUN (from the itsm-triage-agent root directory):

    # Quick test — 1 epoch, tiny batch (good to verify it runs)
    python -m models.finetune.train --epochs 1 --batch-size 16

    # Full training — 3 epochs (recommended first run)
    python -m models.finetune.train --epochs 3 --batch-size 16

    # With GPU (if available — detected automatically)
    python -m models.finetune.train --epochs 3 --batch-size 32

OUTPUT:
    models/finetuned/distilbert-lora/
    ├── adapter_config.json      ← LoRA config (tiny, commitable)
    ├── adapter_model.safetensors ← LoRA weights only (~1.2MB)
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── training_args.bin

HOW HuggingFace TRAINER WORKS (the big picture):

    The Trainer is HuggingFace's high-level training loop. It handles:
      - Moving data and model to GPU automatically
      - Mixed-precision training (fp16/bf16) for speed
      - Gradient accumulation, clipping
      - Checkpointing every N steps
      - Logging (loss, learning rate) to stdout and optionally W&B/TensorBoard
      - Evaluation on the val set after each epoch

    You configure it via TrainingArguments (what to do) and then pass
    your model + datasets to Trainer (how to do it).

    WHY NOT WRITE A MANUAL TRAINING LOOP?

        A manual PyTorch loop (zero_grad → forward → loss → backward →
        step) is 100+ lines for production-quality training. Trainer
        gives you all of that for free, battle-tested and GPU-optimized.
        On the job, you almost always use Trainer or similar frameworks
        (PyTorch Lightning, Accelerate) rather than manual loops.

        The custom part we DO write: DualHeadTrainer.prediction_step()
        to handle our non-standard (dual-head) output format.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import Trainer, TrainingArguments

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
# Ensure project root is on sys.path when running as __main__
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.finetune.dataset import load_splits, load_tokenizer
from models.finetune.model import DualHeadOutput, build_model


# ─── CUSTOM TRAINER ───────────────────────────────────────────────────────────

class DualHeadTrainer(Trainer):
    """
    Trainer subclass that handles our dual-label output format.

    WHY SUBCLASS TRAINER?

        The default Trainer.prediction_step() tries to extract a single
        "logits" field from the model output — it doesn't know about
        cat_logits and pri_logits. Without overriding, the Trainer would
        crash or behave unexpectedly during evaluation.

        By overriding prediction_step() with prediction_loss_only=True,
        we tell the Trainer: "during evaluation, only compute and record
        the loss value. Don't try to extract logits or compute metrics."

        This is intentional for M2 — we just want to see train/val loss
        curves to verify the model is learning. Full accuracy evaluation
        (per-class F1, confusion matrix, etc.) is the job of M5: the
        eval harness.

    SUBCLASSING PATTERN:

        HuggingFace designed Trainer to be subclassed. The recommended
        way to customize behavior is to override specific methods rather
        than modify Trainer's source code. This is the Open/Closed Principle:
        open for extension (subclassing), closed for modification.
    """

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override to always run in loss-only mode.

        WHY prediction_loss_only=True?

            Our DualHeadOutput has cat_logits and pri_logits but no
            single "logits" field. The default Trainer prediction_step
            looks for outputs.logits and crashes if it's not there
            (unless prediction_loss_only=True, in which case it only
            needs outputs.loss).

            For M2, loss is all we need to verify training is working.
            The full metric evaluation (accuracy, F1) belongs in M5.

        The `prediction_loss_only` parameter we receive from the Trainer
        is overridden here — we always use True regardless of what the
        TrainingArguments say.
        """
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=True,   # Force loss-only eval
            ignore_keys=ignore_keys,
        )


# ─── TRAINING ENTRY POINT ─────────────────────────────────────────────────────

def train(
    data_dir:       Path,
    output_dir:     Path,
    epochs:         int   = 3,
    batch_size:     int   = 16,
    learning_rate:  float = 2e-4,
    warmup_ratio:   float = 0.1,
    weight_decay:   float = 0.01,
    lora_r:         int   = 8,
    lora_alpha:     int   = 16,
    lora_dropout:   float = 0.1,
    seed:           int   = 42,
) -> None:
    """
    Full M2 training run.

    Parameters
    ----------
    data_dir : Path
        Directory containing train.jsonl, val.jsonl, test.jsonl.
    output_dir : Path
        Where to save the trained model and adapter weights.
    epochs : int
        Number of full passes over the training set.
        3 is usually enough for convergence on 1500 examples.
    batch_size : int
        Examples per gradient update. 16 fits comfortably on most GPUs.
        Use 32 if you have a GPU with 8GB+ VRAM.
    learning_rate : float
        Peak learning rate. 2e-4 is higher than full fine-tuning (1e-5)
        because LoRA adapters are randomly initialized and need larger
        steps early on. The scheduler will ramp up and then decay this.
    warmup_ratio : float
        Fraction of total training steps used for linear warmup.
        During warmup, LR linearly increases from 0 to learning_rate.
        WHY WARMUP? Randomly initialized heads can produce large gradients
        early in training. Warmup prevents these from destabilizing the
        pre-trained encoder weights.
    weight_decay : float
        L2 regularization on all non-bias, non-LayerNorm parameters.
        Helps prevent overfitting on our small dataset.
    lora_r, lora_alpha, lora_dropout : LoRA hyperparameters (see model.py).
    seed : int
        Random seed for reproducibility.
    """

    logger.info(f"Starting M2 training — epochs={epochs}, lr={learning_rate}, "
                f"batch={batch_size}, lora_r={lora_r}")

    # ── DEVICE DETECTION ─────────────────────────────────────────────────────
    #
    # Check if a CUDA GPU is available. If yes, use it — training on GPU is
    # typically 10-30x faster than CPU for transformer models.
    #
    # The Trainer respects this automatically via the `no_cuda` argument —
    # we pass False (don't disable CUDA) when GPU is available.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                    f"({torch.cuda.get_device_properties(0).total_memory // 1024**3}GB VRAM)")

    # ── TOKENIZER + DATASETS ─────────────────────────────────────────────────

    logger.info("Loading tokenizer and datasets...")
    tokenizer = load_tokenizer()
    train_ds, val_ds, _ = load_splits(data_dir, tokenizer)
    logger.info(f"Train: {len(train_ds)} examples  |  Val: {len(val_ds)} examples")

    # ── MODEL ─────────────────────────────────────────────────────────────────

    logger.info("Building model (downloading DistilBERT weights if needed)...")
    model = build_model(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.print_trainable_parameters()

    # ── TRAINING ARGUMENTS ───────────────────────────────────────────────────
    #
    # TrainingArguments is the configuration object for Trainer.
    # It controls everything about HOW training runs.
    #
    # Key arguments explained:
    #
    # output_dir:
    #   Where checkpoints and the final model are saved.
    #
    # num_train_epochs:
    #   How many times to pass over the full training set.
    #
    # per_device_train_batch_size:
    #   Batch size PER device. If you had 4 GPUs this would give an
    #   effective batch size of 4 × batch_size. We have 1 GPU (or CPU).
    #
    # eval_strategy="epoch":
    #   Run evaluation on val set after every epoch. This tells you
    #   if the model is overfitting (train loss drops but val loss rises).
    #
    # save_strategy="epoch":
    #   Save a checkpoint after every epoch. The best checkpoint is
    #   loaded at the end (load_best_model_at_end=True).
    #
    # load_best_model_at_end=True:
    #   After training, reload the epoch with the best val loss.
    #   This prevents using an overfit model if training ran too long.
    #
    # metric_for_best_model="loss":
    #   "Best" means lowest val loss. Since we're not computing accuracy
    #   during training, loss is our only signal.
    #
    # fp16:
    #   Mixed-precision training — some operations run in float16 instead
    #   of float32. ~2x speedup on modern GPUs, no accuracy loss for
    #   classification. Only enable on GPU.
    #
    # dataloader_num_workers:
    #   Background workers for loading batches from disk. 0 = main thread
    #   only (safe on Windows, which has issues with multiprocessing +
    #   DataLoader). Set to 2-4 on Linux for a speed boost.
    #
    # report_to="none":
    #   Don't send metrics to W&B, TensorBoard, etc. We just want stdout.

    use_fp16 = (device == "cuda")

    # ── WARMUP STEPS ──────────────────────────────────────────────────────────
    #
    # transformers v5.2 deprecated warmup_ratio in favor of warmup_steps.
    # We calculate warmup_steps manually so we can keep using the ratio
    # as a CLI arg (which is more intuitive — "10% of training" vs "32 steps").
    #
    # total_steps = ceil(n_examples / batch_size) * epochs
    # warmup_steps = floor(total_steps * warmup_ratio)
    #
    # For our defaults: ceil(1586/16) * 3 = 100 * 3 = 300 total steps
    # warmup_steps = floor(300 * 0.1) = 30

    import math
    # Use actual train dataset size — NOT a hardcoded constant.
    # This was previously hardcoded to 1586 (old 2000-ticket dataset).
    # Now it correctly adapts to any dataset size (2k, 10k, 100k tickets).
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = max(1, int(total_steps * warmup_ratio))
    logger.info(f"Schedule: {total_steps} total steps, {warmup_steps} warmup steps")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,      # v5.2+: use steps not ratio
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,        # lower loss = better
        fp16=use_fp16,
        dataloader_num_workers=0,       # 0 for Windows compatibility
        seed=seed,
        report_to="none",               # No W&B / TensorBoard
        logging_steps=50,               # Log metrics every 50 steps
        # logging_dir removed: v5.2 deprecated it (use TENSORBOARD_LOGGING_DIR env var)
        save_total_limit=2,             # Keep only the 2 most recent checkpoints
    )

    # ── TRAINER ──────────────────────────────────────────────────────────────

    trainer = DualHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # No compute_metrics — M5 eval harness handles full metrics
    )

    # ── TRAIN ─────────────────────────────────────────────────────────────────

    logger.info("Starting training...")
    trainer.train()

    # ── SAVE ADAPTER WEIGHTS ──────────────────────────────────────────────────
    #
    # We DON'T save the full DistilBERT weights — those are unchanged
    # (frozen). We only save the LoRA adapter matrices.
    #
    # model.encoder is the PEFT-wrapped encoder. Its .save_pretrained()
    # saves only the adapter weights (~1.2MB) plus the config.
    # The full DistilBERT base (~260MB) is left in HuggingFace's cache.
    #
    # WHY SAVE ONLY ADAPTERS?
    #   - 260MB base model is always available from HuggingFace Hub
    #   - 1.2MB adapters is what's actually unique to our training run
    #   - This is the standard deployment pattern for LoRA models

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LoRA adapter to {adapter_dir}")
    model.encoder.save_pretrained(str(adapter_dir))

    # Also save the tokenizer — inference code needs it to tokenize inputs
    tokenizer.save_pretrained(str(adapter_dir))

    # Save the classification head weights separately
    # (these aren't part of the PEFT model, they're raw nn.Linear weights)
    heads_path = adapter_dir / "classification_heads.pt"
    torch.save(
        {
            "cat_head": model.cat_head.state_dict(),
            "pri_head": model.pri_head.state_dict(),
        },
        heads_path,
    )

    logger.info(f"Training complete. Model saved to: {adapter_dir}")
    logger.info(f"To run inference, point your model loader at: {adapter_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="M2: Fine-tune DualHeadDistilBERT + LoRA on ITSM tickets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory with train/val/test JSONL files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned/distilbert-lora",
        help="Where to save model weights (default: models/finetuned/distilbert-lora)",
    )
    parser.add_argument("--epochs",        type=int,   default=3,    help="Training epochs (default: 3)")
    parser.add_argument("--batch-size",    type=int,   default=16,   help="Batch size (default: 16)")
    parser.add_argument("--lr",            type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--warmup-ratio",  type=float, default=0.1,  help="LR warmup fraction (default: 0.1)")
    parser.add_argument("--weight-decay",  type=float, default=0.01, help="L2 regularization (default: 0.01)")
    parser.add_argument("--lora-r",        type=int,   default=8,    help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha",    type=int,   default=16,   help="LoRA alpha (default: 16)")
    parser.add_argument("--lora-dropout",  type=float, default=0.1,  help="LoRA dropout (default: 0.1)")
    parser.add_argument("--seed",          type=int,   default=42,   help="Random seed (default: 42)")

    args = parser.parse_args()

    # Resolve paths relative to project root
    data_dir   = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
