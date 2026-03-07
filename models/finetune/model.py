"""
models/finetune/model.py
────────────────────────
DualHeadDistilBERT: a shared DistilBERT encoder with two classification
heads — one for ticket category, one for priority.

ARCHITECTURE OVERVIEW:

    Input text (tokenized)
          │
    ┌─────▼──────────────────────────────────────┐
    │  DistilBERT encoder (6 transformer layers)  │
    │  ← LoRA adapters on Q and V projections     │
    └─────┬──────────────────────────────────────┘
          │  last_hidden_state: [batch, seq_len, 768]
          │
          ▼  take [CLS] token at position 0
    [CLS] representation: [batch, 768]
          │
          ▼ Dropout(0.1)
    ┌─────┴──────┐
    │            │
    ▼            ▼
 cat_head     pri_head
(768 → 8)   (768 → 4)
    │            │
    ▼            ▼
cat_logits   pri_logits
 [batch, 8]  [batch, 4]

WHY TWO HEADS ON A SHARED ENCODER?

    OPTION A — Two separate models:
        Train one DistilBERT for category, another for priority.
        Pro: fully independent.
        Con: 2x parameters, 2x training time, 2x inference cost.
             The two models can't share what they learn about IT language.

    OPTION B — Two-head model (what we use):
        One shared encoder learns a rich representation of the ticket text.
        Two small linear layers make independent predictions from it.
        Pro: category and priority share 99% of the parameters.
             "Hardware issue with urgent timeline" → the encoder learns that
             both "hardware" and "urgent" are salient, and each head
             independently extracts what it needs.
        Con: shared gradients can sometimes conflict (multi-task learning
             tension). For our task the labels are correlated enough that
             this is rarely a problem.

    OPTION C — Hierarchical: predict category first, use it as input to priority.
        More complex, overkill for this task.

    Two-head is the industry standard for multi-task classification when
    both tasks share the same input.

WHY LoRA INSTEAD OF FULL FINE-TUNING?

    Full fine-tuning updates all 66M parameters of DistilBERT.
    LoRA (Low-Rank Adaptation) freezes the original weights and injects
    tiny trainable "adapter" matrices into the attention layers:

        Frozen weight: W  (e.g., 768×768 = 589,824 parameters)
        LoRA delta:    ΔW = A × B   where A is 768×r, B is r×768
        At inference:  W + ΔW

    With rank r=8, ΔW has only 2×(768×8) = 12,288 parameters — 98% fewer
    than W. We get most of the fine-tuning benefit at a fraction of the cost.

    WHY IS THIS IMPORTANT FOR YOUR PORTFOLIO?
        LoRA is how all production LLMs are customized today (GPT-4 fine-tune,
        Llama, Mistral, etc.). Knowing how to use PEFT/LoRA is a core skill
        for AI engineering roles. Employers will ask about this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DistilBertModel
from transformers.utils import ModelOutput

from models.finetune.dataset import BASE_MODEL_NAME, NUM_CATEGORIES, NUM_PRIORITIES


# ─── OUTPUT CONTAINER ─────────────────────────────────────────────────────────
#
# WHY ModelOutput INSTEAD OF A PLAIN TUPLE OR DATACLASS?
#
# HuggingFace Trainer has specific expectations about what model.forward()
# returns. It looks for a .loss attribute (for backprop) and can
# optionally look for .logits (for evaluation metrics).
#
# transformers.utils.ModelOutput is a special base class that acts like
# both a dataclass (attribute access: outputs.loss) AND a dict
# (key access: outputs["loss"]) AND a tuple (index access: outputs[0]).
#
# This triple compatibility is what lets our custom model work seamlessly
# with the Trainer without any monkey-patching.
#
# The ORDERING of fields matters: outputs[0] must be the loss (if present),
# because some Trainer code accesses index 0 directly.

@dataclass
class DualHeadOutput(ModelOutput):
    """
    Output from DualHeadDistilBERT.forward().

    Fields
    ------
    loss : Optional[Tensor]
        Combined cross-entropy loss (category + priority).
        Present only when labels are passed to forward().
        Shape: [] (scalar).
    cat_logits : Tensor
        Raw unnormalized scores for each category class.
        Shape: [batch_size, NUM_CATEGORIES] = [batch, 8].
    pri_logits : Tensor
        Raw unnormalized scores for each priority class.
        Shape: [batch_size, NUM_PRIORITIES] = [batch, 4].
    """
    loss:       Optional[torch.FloatTensor] = None
    cat_logits: Optional[torch.FloatTensor] = None
    pri_logits: Optional[torch.FloatTensor] = None


# ─── MODEL ────────────────────────────────────────────────────────────────────

class DualHeadDistilBERT(nn.Module):
    """
    Dual-head ticket classifier: shared DistilBERT encoder + LoRA + two heads.

    Parameters
    ----------
    lora_r : int
        LoRA rank. Controls the size of the adapter matrices. Higher rank =
        more parameters = more expressive but slower. r=8 is a standard
        starting point; r=16 if you have more data or need more capacity.
    lora_alpha : int
        LoRA scaling factor. The adapter output is scaled by lora_alpha/r.
        Setting alpha = 2*r (here: 16) is a common heuristic that works well
        across tasks without tuning.
    lora_dropout : float
        Dropout rate on the LoRA adapter activations. Regularization to
        prevent adapter overfitting on small datasets.
    hidden_dropout : float
        Dropout rate on the [CLS] representation before the classification
        heads. Standard regularization for classification models.
    num_categories : int
        Output size of the category head. Defaults to NUM_CATEGORIES (8).
    num_priorities : int
        Output size of the priority head. Defaults to NUM_PRIORITIES (4).
    """

    def __init__(
        self,
        lora_r:          int   = 8,
        lora_alpha:      int   = 16,
        lora_dropout:    float = 0.1,
        hidden_dropout:  float = 0.1,
        num_categories:  int   = NUM_CATEGORIES,
        num_priorities:  int   = NUM_PRIORITIES,
    ) -> None:
        super().__init__()

        # ── STEP 1: Load DistilBERT base encoder ──────────────────────────────
        #
        # DistilBertModel (note: NOT DistilBertForSequenceClassification).
        # We use the base model (just the encoder, no head) because we're
        # adding our own classification heads below.
        #
        # from_pretrained() downloads the weights from HuggingFace Hub on
        # first call and caches them locally (~260MB). Subsequent calls
        # load from cache (~0.5s).
        #
        # WHY NOT use DistilBertForSequenceClassification and fine-tune both
        # the head and the encoder?
        #   That model has only ONE head (for one label). We need TWO heads.
        #   Loading the base model and adding custom heads is more flexible.

        base_encoder = DistilBertModel.from_pretrained(BASE_MODEL_NAME)

        # ── STEP 2: Wrap with LoRA using PEFT ─────────────────────────────────
        #
        # LoraConfig specifies:
        #   - task_type: FEATURE_EXTRACTION means "use the model as an encoder,
        #     don't add a classification head." PEFT won't add its own head.
        #   - r: rank of the adapter matrices
        #   - lora_alpha: scaling factor (see docstring above)
        #   - target_modules: which weight matrices to inject adapters into
        #   - lora_dropout: regularization
        #   - bias: "none" means don't modify bias terms (standard practice)
        #
        # WHY target "q_lin" and "v_lin" (not all attention weights)?
        #
        #   In DistilBERT's MultiHeadSelfAttention, each layer has:
        #     q_lin: W_Q  — projects hidden states to query vectors
        #     k_lin: W_K  — projects to key vectors
        #     v_lin: W_V  — projects to value vectors
        #     out_lin: W_O — projects concatenated heads back to hidden dim
        #
        #   The original LoRA paper (Hu et al., 2021) found that adapting
        #   only Q and V gives most of the benefit with fewer parameters.
        #   Adapting all four (q,k,v,out) improves results marginally but
        #   roughly doubles the adapter parameter count.
        #
        #   PEFT matches these names as substrings — any module whose name
        #   contains "q_lin" or "v_lin" gets an adapter. In DistilBERT's
        #   naming convention that means:
        #     distilbert.transformer.layer.0.attention.q_lin  ← gets LoRA
        #     distilbert.transformer.layer.0.attention.v_lin  ← gets LoRA
        #     ... × 6 layers = 12 adapter pairs total

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=lora_dropout,
            bias="none",
        )

        # get_peft_model() wraps the base encoder:
        #   - Freezes ALL original parameters (requires_grad = False)
        #   - Injects LoRA adapter matrices (requires_grad = True)
        # Result: only ~0.5% of parameters are trainable — much faster training

        self.encoder = get_peft_model(base_encoder, lora_config)

        # ── STEP 3: Dropout for regularization ───────────────────────────────
        #
        # Dropout randomly zeros out neurons during training. This prevents
        # the classification heads from over-relying on any single feature
        # in the [CLS] representation. Standard practice before classification
        # layers — DistilBertForSequenceClassification does the same thing.

        self.dropout = nn.Dropout(hidden_dropout)

        # ── STEP 4: Classification heads ──────────────────────────────────────
        #
        # nn.Linear(in_features, out_features) = a fully connected layer.
        # It's just a matrix multiply: output = input @ W^T + b
        #
        # The [CLS] hidden state has 768 dimensions (DistilBERT's hidden size).
        # We project it down to num_categories (8) or num_priorities (4).
        #
        # These are randomly initialized — they have NO pre-trained weights.
        # That's intentional: the pre-trained DistilBERT had a different
        # task (masked language modeling), so its original head is useless
        # for us. We train these from scratch, which is why they need a
        # higher learning rate than the LoRA adapters.
        #
        # WHY "head" as the naming convention?
        #   In ML, "head" refers to the task-specific part of the model
        #   that sits on top of a general-purpose backbone. Naming them
        #   cat_head and pri_head makes the architecture immediately clear.

        self.cat_head = nn.Linear(768, num_categories)
        self.pri_head = nn.Linear(768, num_priorities)

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        category_labels:  Optional[torch.Tensor] = None,
        priority_labels:  Optional[torch.Tensor] = None,
    ) -> DualHeadOutput:
        """
        Forward pass: text → dual classification logits (+ optional loss).

        WHY OPTIONAL LABELS?

            During training: labels ARE provided → we compute loss here.
            During inference: labels are NOT provided → we just return logits.
            The Trainer calls forward() both ways, so we support both modes.

        Parameters
        ----------
        input_ids : Tensor [batch, seq_len]
            Token IDs from the tokenizer.
        attention_mask : Tensor [batch, seq_len]
            1 for real tokens, 0 for padding.
        category_labels : Tensor [batch] (optional)
            Integer class IDs for category (0-7).
        priority_labels : Tensor [batch] (optional)
            Integer class IDs for priority (0-3).

        Returns
        -------
        DualHeadOutput
            .loss: scalar loss (if labels provided)
            .cat_logits: [batch, 8]
            .pri_logits: [batch, 4]
        """

        # ── ENCODER ───────────────────────────────────────────────────────────
        #
        # Pass the token IDs through DistilBERT.
        # Output: last_hidden_state shape [batch, seq_len, 768]
        # Each position gets a 768-dimensional contextual representation.
        # "Contextual" means the embedding for token i depends on ALL other
        # tokens in the sequence (via self-attention). This is the key
        # difference from classic bag-of-words models.

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # ── [CLS] POOLING ─────────────────────────────────────────────────────
        #
        # last_hidden_state: [batch, seq_len, 768]
        # We take position 0 → the [CLS] token's final representation.
        #
        # WHY [CLS] and not the average of all tokens?
        #
        #   The [CLS] token was designed for classification. During BERT's
        #   pre-training, the model was explicitly trained to accumulate
        #   sentence-level meaning into [CLS] for the next-sentence prediction
        #   task. So it's a learned "summary" of the whole input.
        #
        #   Mean pooling (averaging all token vectors) is an alternative and
        #   sometimes outperforms [CLS] on semantic similarity tasks. For
        #   classification, [CLS] is simpler and works well.
        #
        # [:, 0, :] means:
        #   : = all items in the batch
        #   0 = position 0 ([CLS] is always first)
        #   : = all 768 hidden dims
        # Result shape: [batch, 768]

        cls_hidden = encoder_outputs.last_hidden_state[:, 0, :]

        # Apply dropout for regularization (only active during training;
        # nn.Dropout automatically disables in eval mode)
        cls_hidden = self.dropout(cls_hidden)

        # ── CLASSIFICATION HEADS ──────────────────────────────────────────────
        #
        # Linear projection: [batch, 768] → [batch, num_classes]
        # These are "logits" — raw unnormalized scores.
        #
        # WHY NOT apply softmax here?
        #   torch.nn.functional.cross_entropy() already applies log-softmax
        #   internally (it's numerically more stable). Applying softmax before
        #   cross_entropy() would cause loss to be computed on probabilities
        #   instead of logits, which is less numerically stable and incorrect.
        #
        #   At inference time, if you want probabilities you can apply
        #   F.softmax(logits, dim=-1) after this function returns.

        cat_logits = self.cat_head(cls_hidden)   # [batch, 8]
        pri_logits = self.pri_head(cls_hidden)   # [batch, 4]

        # ── LOSS ──────────────────────────────────────────────────────────────
        #
        # Compute loss only during training (when labels are provided).
        # At inference time (no labels), we skip this block.

        loss = None
        if category_labels is not None and priority_labels is not None:

            # cross_entropy(input, target):
            #   input:  [batch, num_classes] — logits
            #   target: [batch] — integer class IDs
            #
            # Internally: loss = -log(softmax(logits)[true_class])
            # Averaged over the batch.
            #
            # WHY EQUAL WEIGHTING (cat_loss + pri_loss)?
            #
            #   Both tasks have roughly the same difficulty and importance.
            #   If one task dominated (e.g., 10× loss), the model would
            #   neglect the other. Equal weighting is the right default;
            #   you could experiment with e.g. 0.6*cat + 0.4*pri if you
            #   care more about category accuracy.

            cat_loss = F.cross_entropy(cat_logits, category_labels)
            pri_loss = F.cross_entropy(pri_logits, priority_labels)
            loss = cat_loss + pri_loss

        return DualHeadOutput(
            loss=loss,
            cat_logits=cat_logits,
            pri_logits=pri_logits,
        )

    @classmethod
    def from_pretrained(cls, adapter_dir: Path) -> "DualHeadDistilBERT":
        """
        Load a fine-tuned model from a saved adapter directory.

        This is the inference counterpart to the training flow. During
        training we called `build_model()` which internally calls
        `get_peft_model()` to inject NEW random LoRA adapters. Here we
        instead load the SAVED adapter weights from disk.

        WHY USE __new__ INSTEAD OF __init__?

            `DualHeadDistilBERT.__init__` calls `get_peft_model()` which
            creates fresh, randomly-initialized LoRA adapters — the opposite
            of what we want for inference. We'd have to immediately throw
            those away and load the saved ones. That wastes compute and is
            confusing.

            Python's `__new__` allocates a bare instance without calling
            `__init__`. We then manually call `nn.Module.__init__` (which
            sets up PyTorch's internal state) and wire up the components
            ourselves. This is a standard pattern in ML codebases for
            creating models from checkpoints.

        WHY IS THIS A classmethod?

            Following the HuggingFace convention:
              model = AutoModel.from_pretrained("bert-base-uncased")
              model = DualHeadDistilBERT.from_pretrained(adapter_dir)

            Class methods that construct instances are called "named
            constructors" or "factory class methods". They live on the class
            (not an instance) and return a new instance — making the intent
            immediately clear from the call site.

        Parameters
        ----------
        adapter_dir : Path
            Directory containing:
              - adapter_config.json         (LoRA config)
              - adapter_model.safetensors   (LoRA weights)
              - tokenizer files             (vocab, config)
              - classification_heads.pt     (cat_head + pri_head weights)

        Returns
        -------
        DualHeadDistilBERT
            Fully loaded model in eval mode, ready for inference.
        """
        from peft import PeftModel

        # ── 1. Allocate model shell (bypass __init__) ─────────────────────────
        model = cls.__new__(cls)
        nn.Module.__init__(model)   # required: sets up PyTorch Module internals

        # ── 2. Load base DistilBERT + LoRA adapter ────────────────────────────
        #
        # DistilBertModel: fresh base encoder (frozen during training, unchanged)
        # PeftModel.from_pretrained: wraps base with the saved LoRA adapter,
        # overwriting the random adapter matrices with our trained values.
        base_encoder = DistilBertModel.from_pretrained(BASE_MODEL_NAME)
        model.encoder = PeftModel.from_pretrained(base_encoder, str(adapter_dir))

        # ── 3. Recreate dropout + classification heads (shell) ────────────────
        model.dropout  = nn.Dropout(0.1)
        model.cat_head = nn.Linear(768, NUM_CATEGORIES)
        model.pri_head = nn.Linear(768, NUM_PRIORITIES)

        # ── 4. Load saved head weights ────────────────────────────────────────
        #
        # map_location="cpu": load to CPU first regardless of where we're
        # running inference. This prevents errors if the checkpoint was saved
        # on GPU but you're loading on a CPU-only machine.
        #
        # weights_only=True: PyTorch 2.x security recommendation. Prevents
        # arbitrary code execution from malicious .pt files by only loading
        # tensor data, not Python objects. Always use this for inference.
        heads_path = Path(adapter_dir) / "classification_heads.pt"
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        model.cat_head.load_state_dict(heads["cat_head"])
        model.pri_head.load_state_dict(heads["pri_head"])

        # ── 5. Switch to eval mode ────────────────────────────────────────────
        #
        # model.eval() does two things:
        #   1. Disables Dropout (dropout randomly zeros neurons during training
        #      but should be OFF at inference time for deterministic outputs)
        #   2. Tells BatchNorm layers to use running stats (not mini-batch stats)
        #
        # CRITICAL: forgetting model.eval() is a common bug. Dropout during
        # inference produces slightly different outputs on each call —
        # making your model appear "random" and your eval numbers wrong.
        model.eval()

        return model

    def print_trainable_parameters(self) -> None:
        """
        Print a summary of trainable vs total parameters.

        WHY THIS METHOD?
            When you first build a LoRA model, you want to verify that
            PEFT actually froze the base weights. This method gives you
            a human-readable confirmation. You should see something like:

                Trainable: 294,912  |  Total: 66,955,394  |  Trainable%: 0.44%

            If ALL parameters show as trainable, something went wrong with
            the LoRA config.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        pct       = 100 * trainable / total if total > 0 else 0.0
        print(f"Trainable: {trainable:,}  |  Total: {total:,}  |  Trainable%: {pct:.2f}%")


# ─── FACTORY FUNCTION ─────────────────────────────────────────────────────────

def build_model(
    lora_r:       int   = 8,
    lora_alpha:   int   = 16,
    lora_dropout: float = 0.1,
) -> DualHeadDistilBERT:
    """
    Build and return a freshly initialized DualHeadDistilBERT.

    WHY A FACTORY FUNCTION INSTEAD OF CALLING THE CLASS DIRECTLY?

        Training scripts often need a single place to change model
        hyperparameters. The factory function gives you a named entry point
        that's easy to find, grep for, and modify. It also makes it easy
        to add logic later (e.g., loading from checkpoint, A/B testing
        different configs).

    Parameters
    ----------
    lora_r, lora_alpha, lora_dropout : see DualHeadDistilBERT docstring

    Returns
    -------
    DualHeadDistilBERT
        Ready for training. Call .print_trainable_parameters() to confirm
        LoRA is configured correctly.
    """
    model = DualHeadDistilBERT(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    return model
