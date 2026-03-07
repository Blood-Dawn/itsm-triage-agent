"""
data/schema/ticket.py
─────────────────────
The single source of truth for what an ITSM ticket looks like.

Every component in this project — the generator, the model, the API,
the eval harness — imports from this file. If you need to add a new
category or change a field, you do it HERE and everything downstream
picks it up.

This is "schema-driven development": define the shape of your data
once, validate it everywhere, never let bad data silently pass through.
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
#
# enum.Enum — Python's built-in enumeration type. We use it to define a
#   closed set of valid values (categories and priorities). Unlike plain
#   strings, enums prevent typos at construction time and are iterable,
#   so the generator can loop over all categories.
#
# datetime — Standard library for timestamps. We type-hint created_at
#   so Pydantic can validate that the field is actually a datetime, not
#   a random string someone passed in.
#
# uuid — Standard library for generating universally unique identifiers.
#   Every ticket gets a UUID so we can track it across the pipeline
#   (generator → training → inference → eval) without collisions.
#
# pydantic.BaseModel — The foundation of Pydantic. Any class that inherits
#   from BaseModel gets automatic validation, serialization (to JSON/dict),
#   and deserialization (from JSON/dict). This is why we use Pydantic
#   instead of plain dataclasses — we get validation for free.
#
# pydantic.Field — Lets us add metadata to fields: default values,
#   descriptions (which show up in FastAPI's auto-generated docs later),
#   and validation constraints (min_length, max_length, etc.).
#
# typing.Literal — Used for the 'split' field to restrict values to
#   exactly "train", "val", or "test". Similar idea to Enum but lighter
#   weight for one-off fields that don't need their own class.

from enum import Enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from typing import Literal


# ─── ENUMS ───────────────────────────────────────────────────────────────────
#
# WHY ENUMS INSTEAD OF PLAIN STRINGS?
#
# If category were just a `str`, the generator could produce "Hardwre"
# (typo), the model would train on it, and the eval harness would count
# it as a separate class — giving you 9 categories instead of 8 and
# silently corrupting your metrics. With an Enum, Python raises a
# ValueError the moment you try to create an invalid value.
#
# WHY THESE 8 CATEGORIES?
#
# These map to the most common ITSM ticket categories in real IT
# helpdesks (based on ITIL frameworks). They're broad enough to be
# realistic but narrow enough that a classifier can learn meaningful
# distinctions. "Other" is the catch-all for edge cases — every real
# ticketing system has one.

class Category(str, Enum):
    """
    The 8 ticket categories our model will classify into.

    We inherit from BOTH str and Enum. This is a Python pattern called
    a "string enum." It means:
      - Category.HARDWARE == "hardware"  (True — it behaves like a string)
      - Category.HARDWARE is a valid Enum member (iterable, validated)

    Why both? Because when we serialize to JSON, we want the plain string
    "hardware", not "Category.HARDWARE". The str inheritance gives us that
    automatically via Pydantic's JSON serialization.
    """
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK  = "network"
    SECURITY = "security"
    ACCESS   = "access"
    EMAIL    = "email"
    PRINTER  = "printer"
    OTHER    = "other"


class Priority(str, Enum):
    """
    Ticket priority levels, from most to least urgent.

    The distribution in real ITSM systems is NOT uniform:
      P1 (Critical):  ~5%  — Production down, all users affected
      P2 (High):     ~20%  — Major feature broken, workaround exists
      P3 (Medium):   ~50%  — Normal request, standard SLA
      P4 (Low):      ~25%  — Nice-to-have, cosmetic, informational

    We encode this distribution knowledge HERE in the schema (as a
    comment) and enforce it in the GENERATOR. This way, if an interviewer
    asks "why is your dataset skewed toward P3?", you can say: "because
    that's how real helpdesk data is distributed — most tickets are
    routine requests, not emergencies."
    """
    P1 = "P1"  # Critical — production down
    P2 = "P2"  # High     — major impact, workaround available
    P3 = "P3"  # Medium   — standard request
    P4 = "P4"  # Low      — informational / nice-to-have


# ─── PRIORITY WEIGHTS ────────────────────────────────────────────────────────
#
# WHY DEFINE WEIGHTS HERE AND NOT IN THE GENERATOR?
#
# Because these weights are a property of the DATA DOMAIN, not of the
# generation process. If someone builds a different generator (say, one
# that reads from a database instead of Faker), they should use the
# same distribution. Keeping it next to the Priority enum makes that
# connection explicit.
#
# The generator will do:
#   random.choices(list(Priority), weights=PRIORITY_WEIGHTS.values())
#
# This produces a realistic, skewed dataset rather than uniform random.

PRIORITY_WEIGHTS: dict[Priority, float] = {
    Priority.P1: 0.05,   # 5% of tickets
    Priority.P2: 0.20,   # 20%
    Priority.P3: 0.50,   # 50%
    Priority.P4: 0.25,   # 25%
}


# ─── THE TICKET MODEL ────────────────────────────────────────────────────────
#
# This is the core data model. Every synthetic ticket, every API request,
# and every eval test case will be an instance of this class (or a subset
# of its fields).
#
# WHY PYDANTIC BaseModel INSTEAD OF A DATACLASS?
#
# 1. Validation: BaseModel validates types on construction. If you pass
#    priority="URGENT", it raises ValidationError immediately. A dataclass
#    would silently accept it.
#
# 2. Serialization: .model_dump() → dict, .model_dump_json() → JSON string.
#    Dataclasses need you to write your own to_dict() method.
#
# 3. FastAPI integration: When we build the API, FastAPI expects Pydantic
#    models. Using Pydantic here means the same Ticket class works in
#    both the generator and the API — zero duplication.
#
# 4. Schema export: Ticket.model_json_schema() produces a JSON Schema
#    document. Useful for documentation and for validating external data.

class Ticket(BaseModel):
    """
    A single ITSM (IT Service Management) support ticket.

    This model represents both:
    - Generated synthetic data (all fields populated by the generator)
    - Inference input (only 'text' is provided; model predicts the rest)

    Fields are ordered by their role in the ML pipeline:
    1. Identifiers (ticket_id, created_at) — for tracking/logging
    2. Input features (text) — what the model sees
    3. Labels (category, priority, next_action) — what the model predicts
    4. Metadata (split) — for dataset management, not used by the model
    """

    # ── IDENTIFIERS ──────────────────────────────────────────────────────
    #
    # ticket_id: A UUID (universally unique identifier) that looks like
    #   "550e8400-e29b-41d4-a716-446655440000". We use uuid4() which
    #   generates a random UUID — no sequential IDs that could leak
    #   information about generation order.
    #
    #   default_factory=uuid4 means "call uuid4() each time a new Ticket
    #   is created." This is different from default=uuid4() which would
    #   call it ONCE and give every ticket the SAME ID. This is a common
    #   Python gotcha with mutable defaults.

    ticket_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this ticket"
    )

    # created_at: A synthetic timestamp. We don't use this for training
    #   (the model only sees 'text'), but it makes the data look realistic
    #   and could be useful for time-series analysis later.

    created_at: datetime = Field(
        description="When the ticket was submitted"
    )

    # ── INPUT FEATURE ────────────────────────────────────────────────────
    #
    # text: This is the ONLY field the model sees during inference.
    #   It's a concatenation of the ticket title and description,
    #   like a real helpdesk ticket would have.
    #
    #   min_length=10: Rejects garbage like "help" (too short to classify)
    #   max_length=2000: Prevents absurdly long inputs that would waste
    #     tokens and money when calling the LLM API baseline.
    #
    #   WHY ONE 'text' FIELD INSTEAD OF SEPARATE 'title' AND 'description'?
    #
    #   Because the model processes a single text input. If we had separate
    #   fields, we'd need to concatenate them before tokenization anyway.
    #   Keeping it as one field means the schema matches what the model
    #   actually consumes — no preprocessing step needed.

    text: str = Field(
        ...,  # ... means "required, no default" in Pydantic
        min_length=10,
        max_length=2000,
        description="Ticket title + description (the model's input)"
    )

    # ── LABELS (what the model learns to predict) ────────────────────────
    #
    # category: One of 8 values from the Category enum. This is the
    #   PRIMARY classification target — the model's main job is to
    #   sort tickets into the right category.

    category: Category = Field(
        ...,
        description="The ITSM category this ticket belongs to"
    )

    # priority: One of P1-P4 from the Priority enum. This is the
    #   SECONDARY classification target. In our two-head model design
    #   (from the system design doc), category and priority are predicted
    #   simultaneously by two separate classification heads sharing
    #   the same encoder backbone.

    priority: Priority = Field(
        ...,
        description="Urgency level: P1 (critical) through P4 (low)"
    )

    # next_action: A short, actionable recommendation. Unlike category
    #   and priority (which are classification labels), this is a
    #   FREE-TEXT field — the model generates it rather than picking
    #   from a fixed set.
    #
    #   For the fine-tuned DistilBERT model, we DON'T predict this
    #   (DistilBERT is an encoder, not a text generator). Instead,
    #   we use template-based actions from the training data.
    #   For the LLM baseline, the API generates this naturally.
    #
    #   WHY INCLUDE IT IN THE SCHEMA IF DISTILBERT CAN'T GENERATE IT?
    #
    #   Because it's part of the training data (ground truth), it's
    #   used in the eval harness to compare LLM vs template output,
    #   and the API returns it to the user. The schema should represent
    #   the COMPLETE ticket, not just what one specific model uses.

    next_action: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Recommended next step for the support agent"
    )

    # ── METADATA ─────────────────────────────────────────────────────────
    #
    # split: Which dataset partition this ticket belongs to.
    #   - "train": Used to train the model (80% of data)
    #   - "val":   Used during training to check for overfitting (10%)
    #   - "test":  NEVER seen during training — used only for final eval (10%)
    #
    #   WHY IS THIS IN THE SCHEMA AND NOT ASSIGNED LATER?
    #
    #   Because the split is a property of the ticket, not of the file
    #   it's stored in. If we stored train/val/test in separate files
    #   and someone accidentally mixed them up, we'd have data leakage
    #   (test data in training). With the split on each ticket, we can
    #   verify at any time: "is this ticket supposed to be in this file?"
    #
    #   Literal["train", "val", "test"] is like a mini-enum. We use
    #   Literal instead of a full Enum because this field doesn't need
    #   to be iterated over or referenced elsewhere — it's just a label.

    split: Literal["train", "val", "test"] = Field(
        ...,
        description="Dataset partition: train (80%), val (10%), test (10%)"
    )

    # ── PYDANTIC CONFIGURATION ───────────────────────────────────────────
    #
    # model_config is Pydantic v2's way of configuring model behavior.
    # (In Pydantic v1, this was an inner class called Config.)
    #
    # use_enum_values=True: When serializing to JSON, output "hardware"
    #   instead of "Category.HARDWARE". This makes the JSON clean and
    #   compatible with any consumer (JavaScript, pandas, etc.).
    #
    # json_schema_extra: Adds an example to the auto-generated JSON
    #   Schema. This shows up in FastAPI's Swagger docs, making the
    #   API self-documenting.

    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "example": {
                "ticket_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2026-03-01T09:15:00",
                "text": "VPN connection drops every 10 minutes. I'm working remote and losing access to internal tools. Already tried restarting the client.",
                "category": "network",
                "priority": "P2",
                "next_action": "Check VPN server logs for session timeouts. Reset user VPN credentials and verify split-tunnel configuration.",
                "split": "train"
            }
        }
    }
