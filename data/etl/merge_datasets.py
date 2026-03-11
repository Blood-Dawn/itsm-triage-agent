"""
data/etl/merge_datasets.py
───────────────────────────
Merges the synthetic dataset (data/raw/) with all converted external datasets
(data/external/) into a single unified train/val/test split.

USAGE:
    python -m data.etl.merge_datasets
    python -m data.etl.merge_datasets --synthetic_dir data/raw --external_dir data/external --output_dir data/merged

STRATEGY:
    - Synthetic data (10K tickets): included 100% — it has priority-correlated text,
      all 8 categories balanced, and 4 priority levels well represented.
    - External data (~50K+ tickets): included but capped per source to prevent
      any single external dataset from overwhelming the synthetic priority signal.
    - Final split is RE-DONE stratified by (category × priority) so every
      combination appears proportionally in train/val/test.

CAPS BY SOURCE (configurable):
    kaggle_it_service:       15,000  (great category signal, inferred priority)
    mendeley_helpdesk:       10,000  (real priority labels, inferred category)
    zenodo_it_support:        2,229  (all tickets, dataset is small)
    kaggle_customer_support:  5,000  (real priority labels, inferred category)
    console_ai_synthetic:       500  (all tickets, high quality)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.schema.ticket import Category, Priority

random.seed(42)

# ─── CAPS per external source ─────────────────────────────────────────────────
# Set to None for no cap (include all)
SOURCE_CAPS = {
    "kaggle_it_service":       15000,
    "mendeley_helpdesk":       10000,
    "zenodo_it_support":       None,   # small dataset — include all
    "kaggle_customer_support":  5000,
    "console_ai_synthetic":    None,   # small dataset — include all
}

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_split(records: list[dict]) -> list[dict]:
    """
    Assign train/val/test splits stratified by (category, priority).
    This ensures every combination appears proportionally in all splits.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["category"], r["priority"])
        groups[key].append(r)

    split_names = list(SPLIT_RATIOS.keys())
    cumulative = []
    running = 0.0
    for name in split_names:
        running += SPLIT_RATIOS[name]
        cumulative.append(running)

    result = []
    for key, group in groups.items():
        n = len(group)
        random.shuffle(group)
        prev_idx = 0
        for i, name in enumerate(split_names):
            end_idx = n if i == len(split_names) - 1 else int(n * cumulative[i])
            for ticket in group[prev_idx:end_idx]:
                ticket["split"] = name
                result.append(ticket)
            prev_idx = end_idx

    return result


def apply_source_caps(records: list[dict]) -> list[dict]:
    """
    Limit each external source to its configured cap.
    Synthetic data (source == None or "synthetic") is never capped.
    """
    by_source = defaultdict(list)
    for r in records:
        by_source[r.get("source", "synthetic")].append(r)

    capped = []
    for source, group in by_source.items():
        cap = SOURCE_CAPS.get(source)
        if cap is not None and len(group) > cap:
            random.shuffle(group)
            group = group[:cap]
            print(f"  Capped {source:<35} → {len(group):,} tickets")
        else:
            print(f"  Kept  {source:<35}   {len(group):,} tickets")
        capped.extend(group)

    return capped


def main():
    parser = argparse.ArgumentParser(
        description="Merge synthetic + external datasets into unified train/val/test splits."
    )
    parser.add_argument("--synthetic_dir",  default="data/raw",      help="Directory with synthetic JSONL (train.jsonl, val.jsonl, test.jsonl)")
    parser.add_argument("--external_dir",   default="data/external", help="Directory with converted external JSONL files")
    parser.add_argument("--output_dir",     default="data/merged",   help="Output directory for merged JSONL files")
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic_dir)
    external_dir  = Path(args.external_dir)
    output_dir    = Path(args.output_dir)

    print("=" * 60)
    print("ITSM Dataset Merger")
    print("=" * 60)

    # ── Load synthetic data ───────────────────────────────────────────────────
    synthetic_records = []
    for split_file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = synthetic_dir / split_file
        if path.exists():
            loaded = load_jsonl(path)
            for r in loaded:
                r["source"] = "synthetic"
            synthetic_records.extend(loaded)
    print(f"\nSynthetic data: {len(synthetic_records):,} tickets loaded from {synthetic_dir}/")

    # ── Load external data ────────────────────────────────────────────────────
    external_records = []
    combined_path = external_dir / "external_all.jsonl"
    if combined_path.exists():
        external_records = load_jsonl(combined_path)
        print(f"External data:  {len(external_records):,} tickets loaded from {combined_path}")
    else:
        # Load individual files if combined doesn't exist
        for jsonl_file in sorted(external_dir.glob("0*.jsonl")):
            loaded = load_jsonl(jsonl_file)
            external_records.extend(loaded)
            print(f"  Loaded {len(loaded):,} from {jsonl_file.name}")

    if not external_records:
        print("\n⚠ No external data found. Run convert_external.py first.")
        print("  Falling back to synthetic data only.")

    # ── Apply source caps to external data ───────────────────────────────────
    print("\nApplying source caps:")
    external_capped = apply_source_caps(external_records)

    # ── Combine all ──────────────────────────────────────────────────────────
    all_records = synthetic_records + external_capped
    print(f"\nTotal before split: {len(all_records):,} tickets")

    # ── Re-do stratified split on the combined dataset ───────────────────────
    print("Applying stratified train/val/test split...")
    all_records = stratified_split(all_records)

    # ── Write split files ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for r in all_records:
        by_split[r["split"]].append(r)

    for split_name, split_records in by_split.items():
        out_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(split_records, out_path)

    # Write combined file too
    write_jsonl(all_records, output_dir / "all_tickets.jsonl")

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(all_records)
    print(f"\n{'=' * 60}")
    print(f"FINAL MERGED DATASET: {total:,} tickets")
    print(f"{'=' * 60}")

    print("\nSplit breakdown:")
    for split_name in ["train", "val", "test"]:
        count = len(by_split[split_name])
        print(f"  {split_name:<6} {count:>7,}  ({count/total*100:.1f}%)")

    print("\nCategory distribution:")
    cat_counts = Counter(r["category"] for r in all_records)
    for cat in [c.value for c in Category]:
        print(f"  {cat:<12} {cat_counts[cat]:>7,}  ({cat_counts[cat]/total*100:.1f}%)")

    print("\nPriority distribution:")
    pri_counts = Counter(r["priority"] for r in all_records)
    for pri in [p.value for p in Priority]:
        print(f"  {pri}   {pri_counts[pri]:>7,}  ({pri_counts[pri]/total*100:.1f}%)")

    print(f"\n✓ Files written to {output_dir}/")
    print(f"\nNext step — retrain on merged data:")
    print(f"  python -m models.finetune.train --data_dir {output_dir}")


if __name__ == "__main__":
    main()
