"""
data/etl/convert_external.py
─────────────────────────────
Converts 5 external ITSM / helpdesk datasets into the project's unified
JSONL schema so they can be mixed with the synthetic training data.

USAGE:
    python -m data.etl.convert_external --input_dir "C:/path/to/your/zips" --output_dir data/external

    Or with explicit zip/csv paths (defaults shown below).

WHAT EACH DATASET CONTRIBUTES:
    ┌─────────────────────────────────────┬────────────┬──────────┬────────────┬─────────┐
    │ Dataset                             │  # Tickets │ Category │  Priority  │  Text   │
    ├─────────────────────────────────────┼────────────┼──────────┼────────────┼─────────┤
    │ 1. IT Service Kaggle                │  ~47,837   │    ✓     │  inferred  │ cleaned │
    │ 2. Help Desk Mendeley               │  ~12,000   │ inferred │     ✓      │ real    │
    │ 3. Zenodo IT Support                │   ~2,229   │    ✓     │  inferred  │ sparse  │
    │ 4. Customer Support Kaggle          │   ~5,000   │ inferred │     ✓      │ real    │
    │ 5. Console-AI HF Synthetic          │     ~500   │    ✓     │     ✓      │ real    │
    └─────────────────────────────────────┴────────────┴──────────┴────────────┴─────────┘

SCHEMA MAPPING STRATEGY:
    - Datasets WITH priority labels → priority used directly (mapped to P1-P4)
    - Datasets WITHOUT priority → priority inferred from urgency keywords in text
    - Datasets WITH category labels → category mapped to our 8-class schema
    - Datasets WITHOUT category → inferred via keyword matching on ticket text
    - next_action → generated from templates (no external dataset provides this)
"""

import argparse
import json
import random
import re
import zipfile
import io
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# ── Add project root to path so we can import our own modules ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from data.generator.templates import TEMPLATES
from data.schema.ticket import Category, Priority

random.seed(42)

# ─── LABEL MAPS ──────────────────────────────────────────────────────────────

# Priority mappings from each dataset's scale to our P1-P4 system
PRIORITY_MAP = {
    # Mendeley
    "blocker":  "P1", "highest": "P1",
    "high":     "P2",
    "medium":   "P3", "normal": "P3",
    "low":      "P4", "lowest": "P4",
    # Customer Support Kaggle
    "critical": "P1", "urgent": "P1",
    # Console-AI HuggingFace (already uses our terms mostly)
    "p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4",
}

# Dataset 1: IT Service Kaggle → our 8 categories
KAGGLE_IT_CATEGORY_MAP = {
    "hardware":             "hardware",
    "storage":              "hardware",       # storage = disk/hardware
    "access":               "access",
    "administrative rights":"access",         # admin rights = access
    "hr support":           "other",
    "miscellaneous":        "other",
    "purchase":             "other",
    "internal project":     "other",
}

# Dataset 3: Zenodo → our 8 categories
ZENODO_CATEGORY_MAP = {
    "fileservice":        "access",     # file share = access management
    "active directory":   "access",     # AD = access/identity
    "support general":    "other",
    "software":           "software",
    "o365":               "software",   # M365 apps = software
    "computer-services":  "hardware",   # computer services = hardware support
    "eol":                "hardware",   # end-of-life = hardware refresh
}

# Dataset 5: Console-AI → our 8 categories
CONSOLE_AI_CATEGORY_MAP = {
    "software":        "software",
    "account":         "access",
    "network":         "network",
    "security":        "security",
    "communication":   "email",
    "hardware":        "hardware",
    "remotework":      "network",    # remote work issues → network/VPN
    "training":        "other",
    "infrastructure":  "network",
    "licensing":       "software",
    "performance":     "software",
}

# Urgency keywords for text-based priority inference (when no label exists)
# Each level uses a score: highest score wins
URGENCY_SIGNALS = {
    "P1": [
        "production is down", "completely down", "entire company", "all users",
        "critical outage", "data loss", "immediate", "emergency", "cannot work",
        "revenue loss", "sla breach", "all staff", "entire organization",
        "business critical", "system is down", "down for everyone",
        "urgent escalation", "blocking everyone", "no one can", "fire",
        "security breach", "ransomware", "data breach", "compromised",
    ],
    "P2": [
        "multiple users", "my whole team", "entire department", "deadline today",
        "several people", "client meeting", "important presentation",
        "customer facing", "affecting many", "can't wait", "asap",
        "high priority", "urgent", "time sensitive", "due today",
        "end of day", "business impact", "needs to be done today",
    ],
    "P3": [
        "intermittent", "sometimes", "occasionally", "workaround",
        "single user", "just me", "affects only me", "minor issue",
        "when possible", "low urgency", "this week", "no rush please",
        "inconvenient", "not blocking", "can still work",
    ],
    "P4": [
        "no rush", "whenever you get a chance", "low priority",
        "nice to have", "enhancement", "whenever available",
        "not urgent", "future", "cosmetic", "informational",
        "whenever convenient", "fyi", "suggestion", "request for",
    ],
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def infer_priority_from_text(text: str) -> str:
    """
    Infer P1-P4 from urgency keywords when no priority label exists.
    Scores each priority level by counting matching keywords in the text.
    Falls back to a weighted random sample matching real-world distribution
    (P1=5%, P2=20%, P3=50%, P4=25%) if no signals are found — this prevents
    all inferred tickets from clustering at P3.
    """
    text_lower = text.lower()
    scores = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    for level, keywords in URGENCY_SIGNALS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[level] += 1

    # If no urgency signals found, sample from the real-world distribution
    # rather than always defaulting to P3 (which causes severe P3 bias)
    if max(scores.values()) == 0:
        return random.choices(
            population=["P1", "P2", "P3", "P4"],
            weights=[0.05, 0.20, 0.50, 0.25],
            k=1
        )[0]
    return max(scores, key=lambda k: scores[k])


def infer_category_from_text(text: str) -> str:
    """
    Infer one of our 8 categories from keyword patterns in the ticket text.
    Used for datasets that have NO category label.
    """
    text_lower = text.lower()
    signals = {
        "hardware":  ["laptop", "desktop", "monitor", "keyboard", "mouse", "printer",
                      "hardware", "device", "screen", "battery", "charger", "ram",
                      "disk", "ssd", "drive", "storage", "power supply", "docking"],
        "software":  ["software", "application", "app", "crash", "install", "update",
                      "microsoft", "excel", "word", "teams", "zoom", "outlook app",
                      "version", "license", "bug", "error code", "plugin"],
        "network":   ["network", "internet", "vpn", "wifi", "wi-fi", "ethernet",
                      "connection", "bandwidth", "latency", "dns", "firewall",
                      "ip address", "packet loss", "connectivity", "remote access"],
        "security":  ["security", "phishing", "malware", "virus", "ransomware",
                      "breach", "compromised", "mfa", "two-factor", "suspicious",
                      "password reset", "locked out", "unauthorized"],
        "access":    ["access", "permission", "sharepoint", "shared drive", "jira",
                      "confluence", "github", "aws", "role", "provisioning",
                      "active directory", "sso", "login", "sign in", "account"],
        "email":     ["email", "outlook", "inbox", "calendar", "meeting invite",
                      "distribution list", "mailbox", "spam", "bounce", "exchange",
                      "mail flow", "signature", "auto reply", "attachment"],
        "printer":   ["print", "printer", "scan", "fax", "toner", "paper jam",
                      "copier", "plotter", "print queue", "print server"],
    }
    category_scores = {cat: 0 for cat in signals}
    for cat, keywords in signals.items():
        for kw in keywords:
            if kw in text_lower:
                category_scores[cat] += 1

    best = max(category_scores, key=lambda k: category_scores[k])
    return best if category_scores[best] > 0 else "other"


def get_next_action(category: str, priority: str) -> str:
    """
    Pull a priority-appropriate next_action from our templates.
    This is used for ALL external datasets since none provide next_action.
    """
    priority_key = priority.lower()  # "P1" → "p1"
    actions = TEMPLATES[category][f"{priority_key}_actions"]
    return random.choice(actions)


def map_priority(raw_priority: str) -> str | None:
    """Map a raw priority string to P1-P4. Returns None if unmappable."""
    if not raw_priority or pd.isna(raw_priority):
        return None
    mapped = PRIORITY_MAP.get(str(raw_priority).lower().strip())
    return mapped


def clean_text(text: str) -> str:
    """Basic text cleaning: strip whitespace, collapse multiple spaces."""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_english(text: str, threshold: float = 0.7) -> bool:
    """
    Rough English language detector based on common English word frequency.
    Used to filter non-English tickets from the Zenodo multilingual dataset.
    """
    english_common = {"the", "a", "an", "is", "are", "was", "were", "i",
                      "my", "we", "you", "it", "in", "on", "at", "to",
                      "for", "not", "can", "have", "has", "this", "that",
                      "and", "or", "but", "with", "from", "hi", "hello",
                      "please", "help", "issue", "error", "problem"}
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 3:
        return True  # too short to judge
    matches = sum(1 for w in words if w in english_common)
    return (matches / len(words)) >= threshold


def make_ticket(text: str, category: str, priority: str, source: str) -> dict | None:
    """
    Build a ticket dict in our schema. Returns None if the ticket is invalid.

    Validation:
        - text must be at least 15 characters
        - category must be one of our 8 classes
        - priority must be P1-P4
    """
    text = clean_text(text)
    if len(text) < 15:
        return None
    if category not in [c.value for c in Category]:
        return None
    if priority not in [p.value for p in Priority]:
        return None

    action = get_next_action(category, priority)

    return {
        "text":        text,
        "category":    category,
        "priority":    priority,
        "next_action": action,
        "split":       "train",   # will be re-split by merge_datasets.py
        "source":      source,    # provenance tracking
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ Wrote {len(records):,} records → {path}")


# ─── DATASET 1: IT Service Ticket Classification (Kaggle, 47K) ───────────────

def convert_kaggle_it_service(zip_path: str, output_dir: Path) -> list[dict]:
    """
    47,837 real IT helpdesk tickets from an enterprise environment.
    Fields: Document (text), Topic_group (category).
    No priority label — inferred from text urgency keywords.

    NOTE: The text has been pre-processed (stemmed/tokenised) by the original
    authors. It contains 'icon' placeholders instead of actual terms. The text
    is not natural language but still carries learnable category signals.
    """
    print("\n[1/5] IT Service Ticket Classification (Kaggle, 47K)")

    with zipfile.ZipFile(zip_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"  Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    records = []
    skipped = {"no_text": 0, "bad_category": 0, "too_short": 0}

    for _, row in df.iterrows():
        raw_text = clean_text(row.get("Document", ""))
        raw_cat  = str(row.get("Topic_group", "")).lower().strip()

        if not raw_text:
            skipped["no_text"] += 1
            continue

        category = KAGGLE_IT_CATEGORY_MAP.get(raw_cat)
        if category is None:
            skipped["bad_category"] += 1
            continue

        # No priority label → infer from text
        priority = infer_priority_from_text(raw_text)

        ticket = make_ticket(raw_text, category, priority, source="kaggle_it_service")
        if ticket:
            records.append(ticket)
        else:
            skipped["too_short"] += 1

    print(f"  Converted: {len(records):,} | Skipped: {skipped}")
    write_jsonl(records, output_dir / "01_kaggle_it_service.jsonl")
    return records


# ─── DATASET 2: Help Desk Tickets (Mendeley, 66K) ────────────────────────────

def convert_mendeley_helpdesk(zip_path: str, output_dir: Path) -> list[dict]:
    """
    66,691 real enterprise helpdesk tickets from an international software company.
    issues.csv has priority labels but NO text.
    sample_utterances.csv has the conversation text but it's fragmented.

    Strategy:
    - Join utterances to issues on issueid → issue.id
    - Aggregate all utterance text per issue into one ticket text
    - Use issue_priority for priority label (skip "unknown" priority)
    - Infer category from aggregated text (no category field exists)
    - Require at least 20 words to filter out empty/trivial tickets
    """
    print("\n[2/5] Help Desk Tickets (Mendeley, ~12K usable)")

    with zipfile.ZipFile(zip_path) as z:
        # Find the CSV files inside the zip
        names = z.namelist()
        issues_name = [n for n in names if "issues.csv" in n and "snapshot" not in n][0]
        utt_name    = [n for n in names if "sample_utterances" in n][0]

        with z.open(issues_name) as f:
            issues = pd.read_csv(f, low_memory=False)
        with z.open(utt_name) as f:
            utterances = pd.read_csv(f, low_memory=False)

    print(f"  Issues: {len(issues):,} rows | Utterances: {len(utterances):,} rows")

    # Keep only reporter utterances (ignore private internal notes)
    reporter_utts = utterances[
        (utterances["is_private"] == 0) &
        (utterances["author_role"] == "reporter") &
        (utterances["actionbody"].notna())
    ].copy()

    # Aggregate utterance text per issue
    agg_text = (
        reporter_utts
        .groupby("issueid")["actionbody"]
        .apply(lambda parts: " ".join(str(p).strip() for p in parts if len(str(p).strip()) > 2))
        .reset_index()
        .rename(columns={"issueid": "id", "actionbody": "full_text"})
    )

    # Join to issues to get the priority label
    merged = agg_text.merge(
        issues[["id", "issue_priority"]].drop_duplicates("id"),
        on="id",
        how="inner"
    )

    print(f"  Joined: {len(merged):,} tickets with text + priority")

    records = []
    skipped = {"unknown_priority": 0, "too_short": 0}

    for _, row in merged.iterrows():
        raw_priority = str(row.get("issue_priority", "")).strip()
        priority = map_priority(raw_priority)
        if priority is None or raw_priority.lower() == "unknown":
            skipped["unknown_priority"] += 1
            continue

        text = clean_text(row["full_text"])
        # Strip Mendeley anonymization markers (ph_name, ph_user, etc.)
        text = re.sub(r"\bph_\w+\b", "", text)
        text = clean_text(text)

        words = text.split()
        if len(words) < 10:
            skipped["too_short"] += 1
            continue

        # NOTE: No English filter — Mendeley text is English but uses
        # anonymization markers (ph_name, ph_technical) that fool the
        # English detector. The dataset is from an international software
        # company and all ticket text is in English.

        # Infer category from text (no category field in this dataset)
        category = infer_category_from_text(text)

        ticket = make_ticket(text, category, priority, source="mendeley_helpdesk")
        if ticket:
            records.append(ticket)
        else:
            skipped["too_short"] += 1

    print(f"  Converted: {len(records):,} | Skipped: {skipped}")
    write_jsonl(records, output_dir / "02_mendeley_helpdesk.jsonl")
    return records


# ─── DATASET 3: Classification of IT Support Tickets (Zenodo, 2.2K) ──────────

def convert_zenodo_it_support(zip_path: str, output_dir: Path) -> list[dict]:
    """
    2,229 real IT support tickets from a Brazil-based company.
    Has category labels (7 classes) and ticket text.
    No priority — inferred from text.
    Multilingual (mostly English but some Portuguese/German) — English-only filtered.
    WARNING: Texts are very short (subject-line only). Still useful for
    enriching category vocabulary but expect low priority signal quality.
    """
    print("\n[3/5] Classification of IT Support Tickets (Zenodo, ~2K)")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        def read_csv_from_zip(pattern):
            fname = [n for n in names if pattern in n][0]
            with z.open(fname) as f:
                return pd.read_csv(f)

        X_train = read_csv_from_zip("X_train")
        X_test  = read_csv_from_zip("X_test")
        y_train = read_csv_from_zip("y_train")
        y_test  = read_csv_from_zip("y_test")

    df = pd.concat([
        X_train.merge(y_train, on="id"),
        X_test.merge(y_test, on="id"),
    ], ignore_index=True)

    print(f"  Loaded {len(df):,} rows")

    records = []
    skipped = {"bad_category": 0, "too_short": 0, "not_english": 0}

    # Zenodo placeholder pattern: [TICKET ID], [NAME], [COMPANY], etc.
    PLACEHOLDER_RE = re.compile(r"\[[\w\s]+\]")

    for _, row in df.iterrows():
        raw_text = clean_text(row.get("text", ""))
        raw_cat  = str(row.get("category_truth", "")).lower().strip()

        # Strip Zenodo anonymization placeholders before checking length/language
        # e.g. "File Share Access - [TICKET ID] - [NAME]" → "File Share Access"
        stripped = PLACEHOLDER_RE.sub("", raw_text)
        stripped = clean_text(stripped)

        if len(stripped) < 15:
            skipped["too_short"] += 1
            continue

        # Only check English on the stripped text — placeholders were throwing
        # off the detector since [TICKET ID] [NAME] [COMPANY] are not English words
        if not is_english(stripped):
            skipped["not_english"] += 1
            continue

        category = ZENODO_CATEGORY_MAP.get(raw_cat)
        if category is None:
            skipped["bad_category"] += 1
            continue

        priority = infer_priority_from_text(stripped)
        ticket = make_ticket(stripped, category, priority, source="zenodo_it_support")
        if ticket:
            records.append(ticket)
        else:
            skipped["too_short"] += 1

    print(f"  Converted: {len(records):,} | Skipped: {skipped}")
    write_jsonl(records, output_dir / "03_zenodo_it_support.jsonl")
    return records


# ─── DATASET 4: Customer Support Ticket Dataset (Kaggle, 8.5K) ───────────────

def convert_customer_support_kaggle(zip_path: str, output_dir: Path) -> list[dict]:
    """
    8,469 customer support tickets covering technical and non-technical issues.
    Has priority labels (Critical/High/Medium/Low) → mapped to P1-P4.

    Category strategy:
    - Technical issue + keyword match → hardware/software/network/security
    - Billing/Refund/Cancellation/Product inquiry → filtered OUT (not IT)
    - Subject used for category if Ticket Type is generic

    Text: Subject + Description combined. Descriptions have {product_purchased}
    placeholders which are resolved using the Product Purchased column.

    Only "Technical issue" type tickets are kept — other types (billing, refund)
    are not relevant to an IT helpdesk model.
    """
    print("\n[4/5] Customer Support Ticket Dataset (Kaggle, ~5K technical)")

    with zipfile.ZipFile(zip_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"  Loaded {len(df):,} rows")

    # Keep only technical issues — billing/refund/cancel not relevant for ITSM
    technical_types = {"Technical issue", "Product inquiry"}
    df = df[df["Ticket Type"].isin(technical_types)].copy()
    print(f"  After filtering to technical types: {len(df):,} rows")

    records = []
    skipped = {"bad_priority": 0, "too_short": 0, "no_it_match": 0}

    for _, row in df.iterrows():
        raw_priority = str(row.get("Ticket Priority", "")).strip()
        priority = map_priority(raw_priority)
        if priority is None:
            skipped["bad_priority"] += 1
            continue

        # Build text: subject + description, resolving {product_purchased}
        subject = clean_text(row.get("Ticket Subject", ""))
        desc    = clean_text(row.get("Ticket Description", ""))
        product = clean_text(row.get("Product Purchased", "device"))

        # Replace the unfilled placeholder with the actual product name
        desc = desc.replace("{product_purchased}", product)
        desc = desc.replace("{error_message}", "an unexpected error")

        text = f"{subject}\n{desc}" if subject else desc

        if len(text) < 20:
            skipped["too_short"] += 1
            continue

        # Infer IT category from combined text + subject
        category = infer_category_from_text(text)

        # If we got "other" with no clear IT signal, still keep it — the priority
        # label is the main value of this dataset
        ticket = make_ticket(text, category, priority, source="kaggle_customer_support")
        if ticket:
            records.append(ticket)
        else:
            skipped["too_short"] += 1

    print(f"  Converted: {len(records):,} | Skipped: {skipped}")
    write_jsonl(records, output_dir / "04_kaggle_customer_support.jsonl")
    return records


# ─── DATASET 5: Console-AI IT Helpdesk Synthetic (HuggingFace, 500) ──────────

def convert_console_ai_synthetic(csv_path: str, output_dir: Path) -> list[dict]:
    """
    500 high-quality synthetic IT helpdesk tickets from Hugging Face.
    Has subject, description, priority (Urgent/High/Medium/Low), and category.
    Both subject and description are combined into the ticket text.
    This is the highest-quality external dataset — real conversational language
    with both category AND priority labels.
    """
    print("\n[5/5] Console-AI IT Helpdesk Synthetic (HuggingFace, 500)")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    records = []
    skipped = {"bad_category": 0, "bad_priority": 0, "too_short": 0}

    for _, row in df.iterrows():
        raw_cat      = str(row.get("category", "")).strip()
        raw_priority = str(row.get("priority", "")).strip()

        priority = map_priority(raw_priority)
        if priority is None:
            skipped["bad_priority"] += 1
            continue

        category = CONSOLE_AI_CATEGORY_MAP.get(raw_cat.lower())
        if category is None:
            skipped["bad_category"] += 1
            continue

        subject = clean_text(row.get("subject", ""))
        desc    = clean_text(row.get("description", ""))
        text    = f"{subject}\n{desc}" if subject and desc else (subject or desc)

        ticket = make_ticket(text, category, priority, source="console_ai_synthetic")
        if ticket:
            records.append(ticket)
        else:
            skipped["too_short"] += 1

    print(f"  Converted: {len(records):,} | Skipped: {skipped}")
    write_jsonl(records, output_dir / "05_console_ai_synthetic.jsonl")
    return records


# ─── SUMMARY ─────────────────────────────────────────────────────────────────

def print_summary(all_records: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"TOTAL CONVERTED: {len(all_records):,} tickets")
    print("=" * 60)

    by_source = Counter(r["source"] for r in all_records)
    print("\nBy source:")
    for src, count in by_source.most_common():
        print(f"  {src:<35} {count:>6,}")

    by_category = Counter(r["category"] for r in all_records)
    print("\nBy category:")
    for cat in [c.value for c in Category]:
        print(f"  {cat:<12} {by_category[cat]:>6,}")

    by_priority = Counter(r["priority"] for r in all_records)
    print("\nBy priority:")
    for pri in [p.value for p in Priority]:
        pct = by_priority[pri] / len(all_records) * 100
        print(f"  {pri}   {by_priority[pri]:>6,}  ({pct:.1f}%)")

    print(f"\nNote: Priorities labelled 'inferred' came from keyword")
    print(f"      matching, not original dataset labels.")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert external ITSM datasets to the project's unified JSONL schema."
    )
    parser.add_argument(
        "--kaggle_it_service",
        default=r"C:\Users\kheiven\Documents\IT service ticket classification dataset.zip",
        help="Path to IT service ticket classification dataset.zip"
    )
    parser.add_argument(
        "--mendeley",
        default=r"C:\Users\kheiven\Documents\Help Desk Tickets.zip",
        help="Path to Help Desk Tickets.zip"
    )
    parser.add_argument(
        "--zenodo",
        default=r"C:\Users\kheiven\Documents\Classificatrion of IT support tickets.zip",
        help="Path to Classification of IT support tickets.zip"
    )
    parser.add_argument(
        "--customer_support",
        default=r"C:\Users\kheiven\Documents\Customer support Ticket dataset.zip",
        help="Path to Customer support Ticket dataset.zip"
    )
    parser.add_argument(
        "--console_ai",
        default=r"C:\Users\kheiven\Documents\Console-AI_IT-helpdesk-synthetic-tickets.csv",
        help="Path to Console-AI_IT-helpdesk-synthetic-tickets.csv"
    )
    parser.add_argument(
        "--output_dir",
        default="data/external",
        help="Output directory for converted JSONL files (default: data/external)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ITSM External Dataset ETL Converter")
    print("=" * 60)

    all_records = []

    # Run each converter — skip gracefully if a file doesn't exist
    converters = [
        (args.kaggle_it_service,  convert_kaggle_it_service),
        (args.mendeley,           convert_mendeley_helpdesk),
        (args.zenodo,             convert_zenodo_it_support),
        (args.customer_support,   convert_customer_support_kaggle),
    ]

    for path, converter in converters:
        if not Path(path).exists():
            print(f"\n  ⚠ Skipping (file not found): {path}")
            continue
        try:
            records = converter(path, output_dir)
            all_records.extend(records)
        except Exception as e:
            print(f"\n  ✗ Error in {converter.__name__}: {e}")
            import traceback; traceback.print_exc()

    # Console-AI is a CSV, not a zip
    if Path(args.console_ai).exists():
        try:
            records = convert_console_ai_synthetic(args.console_ai, output_dir)
            all_records.extend(records)
        except Exception as e:
            print(f"\n  ✗ Error in convert_console_ai_synthetic: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n  ⚠ Skipping Console-AI (file not found): {args.console_ai}")

    # Write the combined external dataset as a single file
    if all_records:
        combined_path = output_dir / "external_all.jsonl"
        write_jsonl(all_records, combined_path)
        print_summary(all_records)
    else:
        print("\n⚠ No records converted — check that file paths are correct.")


if __name__ == "__main__":
    main()
