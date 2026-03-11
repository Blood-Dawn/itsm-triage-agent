"""
data/generator/gen.py
─────────────────────
Synthetic ITSM ticket generator.

Usage:
    python -m data.generator.gen                    # 2000 tickets, seed 42
    python -m data.generator.gen --n 500            # 500 tickets
    python -m data.generator.gen --seed 123         # different seed
    python -m data.generator.gen --output data/raw  # custom output dir

What this script does:
    1. Generates N synthetic IT helpdesk tickets using Faker + templates
    2. Assigns realistic category/priority distributions
    3. Splits into train/val/test (80/10/10, stratified)
    4. Writes one JSONL file per split

Why this approach:
    - Faker gives us realistic names, dates, and filler text
    - Templates give us domain-specific vocabulary per category
    - Stratified split ensures every category/priority combo appears proportionally in all three sets
    - Fixed seed makes the entire process 100% reproducible
"""

# ─── STANDARD LIBRARY IMPORTS ────────────────────────────────────────────────
#
# argparse:  Python's built-in command-line argument parser. We use it
#            so you can control the generator from the terminal:
#            --n (how many tickets), --seed (reproducibility), --output (path).
#            This is standard in ML tools - every research paper includes
#            "run with --seed 42 to reproduce our results."
#
# json:      For writing JSONL (one JSON object per line). We don't use
#            pandas to_json() because we want line-by-line control and
#            no pandas dependency in the generator itself.
#
# random:    Python's built-in random number generator. We set the seed
#            once at the start so every random.choice() call produces the
#            same sequence every time. This is critical for reproducibility.
#
# pathlib:   Modern Python path handling. We use Path instead of os.path
#            because it's cleaner: Path("data") / "raw" / "train.jsonl"
#            vs os.path.join("data", "raw", "train.jsonl"). Same result,
#            more readable code.
#
# datetime:  For generating realistic ticket timestamps.
#
# collections.Counter: For counting how many tickets we generated per
#            category/priority - used in the summary report at the end.

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

# ─── THIRD-PARTY IMPORTS ─────────────────────────────────────────────────────
#
# faker.Faker: A library that generates realistic fake data - names,
#              dates, sentences, company names, etc. We use it to add
#              variability to our templates so tickets don't all sound
#              identical. Faker also accepts a seed for reproducibility.
#
# rich.console: A terminal formatting library. Makes the generator's
#               output pretty and readable with colors and tables.
#               Optional - the generator works without it, but it makes
#               the developer experience nicer. This is a small habit
#               that signals professionalism to code reviewers.

from faker import Faker
from rich.console import Console
from rich.table import Table as RichTable

# ─── PROJECT IMPORTS ─────────────────────────────────────────────────────────
#
# We import from our own modules:
#   - ticket.py: The schema (Ticket model, Category/Priority enums, weights)
#   - templates.py: The per-category text templates
#
# This is the schema-driven approach in action: the generator doesn't
# define what a ticket looks like - it IMPORTS that definition from
# the schema. If you add a new category to ticket.py, the generator
# will automatically try to generate tickets for it (and crash with a
# helpful error if you forgot to add templates).

from data.schema.ticket import (
    Ticket,
    Category,
    Priority,
    PRIORITY_WEIGHTS,
)
from data.generator.templates import TEMPLATES, _validate_templates

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
#
# DEFAULT_N:     How many tickets to generate. 2000 is enough for
#                fine-tuning DistilBERT (it's a small model) while being
#                fast to generate (< 5 seconds).
#
# DEFAULT_SEED:  42 is the universal convention for default random seeds
#                in ML. It comes from "The Hitchhiker's Guide to the
#                Galaxy" (the answer to everything). Using 42 means anyone
#                cloning your repo gets the exact same dataset without
#                having to read your docs.
#
# SPLIT_RATIOS:  80% train, 10% validation, 10% test. This is the most
#                common split in ML. Why not 70/15/15?  Because with
#                2000 tickets, 10% = 200 test samples, which is enough
#                to compute meaningful F1 scores. The more training data,
#                the better the fine-tuned model.

DEFAULT_N = 10000
DEFAULT_SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


# ─── CORE GENERATION LOGIC ──────────────────────────────────────────────────

def generate_ticket_text(category: str, priority: str, fake: Faker) -> tuple[str, str]:
    """
    Generate a realistic ticket text and next_action for a given category AND priority.

    KEY CHANGE FROM v1:
        In v1, priority was assigned randomly and was completely decoupled from
        the ticket text. A P1 "laptop won't turn on" and a P4 "laptop won't turn on"
        had IDENTICAL text — the model could not learn the difference.

        In v2, we inject a PRIORITY-SPECIFIC context sentence into every ticket.
        P1 tickets contain phrases like "production is completely down" and "entire
        team cannot work". P4 tickets contain phrases like "no rush" and "whenever
        IT has bandwidth". This gives the model learnable text signals to correlate
        with priority labels.

    HOW THE TEXT IS CONSTRUCTED:

        title:            "{subject}: {symptom}"  (short summary line)
        urgency_context:  priority-specific sentence injected into the body
        optional detail:  general context (added ~50% of the time for variety)
        faker sentences:  0-2 random sentences for natural length variation

    Parameters
    ----------
    category : str
        One of the 8 category values ("hardware", "software", etc.)
    priority : str
        One of "P1", "P2", "P3", "P4" — controls which context and action pool
        is selected, ensuring text correlates with the assigned priority label.
    fake : Faker
        A seeded Faker instance for reproducible random data.

    Returns
    -------
    tuple[str, str]
        (ticket_text, next_action) where next_action is priority-appropriate.
    """
    template = TEMPLATES[category]

    subject = random.choice(template["subjects"])
    symptom = random.choice(template["symptoms"])

    # Select the priority-specific context and action.
    # e.g. priority="P1" → template["p1_contexts"] and template["p1_actions"]
    priority_key = priority.lower()   # "P1" → "p1"
    urgency_context = random.choice(template[f"{priority_key}_contexts"])
    action          = random.choice(template[f"{priority_key}_actions"])

    # ── BUILD THE TICKET TEXT ────────────────────────────────────────────
    #
    # Structure mirrors a real helpdesk ticket:
    #
    #   Line 1 (title):  "VPN connection: drops every few minutes"
    #   Body sentence 1: "Hi, my VPN connection is drops every few minutes."
    #   Body sentence 2: [urgency_context — the priority signal for the model]
    #   Body sentence 3: [optional general detail for variety]
    #   Body sentence 4+: [0-2 Faker sentences for natural length variation]

    title = f"{subject}: {symptom}"

    sentences = [
        f"Hi, my {subject} is {symptom}.",
        urgency_context,   # <— this is the priority signal
    ]

    # Add a general detail sentence ~50% of the time for variety.
    # Without this, every ticket would have an identical two-sentence structure.
    if random.random() < 0.5:
        sentences.append(random.choice(template["details"]))

    # Add 0-2 Faker sentences for natural length variation.
    # Faker sentences add noise that mirrors real ticket padding.
    for _ in range(random.randint(0, 2)):
        sentences.append(fake.sentence())

    body = " ".join(sentences)
    text = f"{title}\n{body}"

    return text, action


def assign_priority(category: str) -> Priority:
    """
    Assign a priority level using the weighted distribution from the schema.

    WHY NOT JUST random.choice(list(Priority))?

    That would give each priority 25% probability - uniform distribution.
    Real ITSM data is heavily skewed: most tickets are P3 (routine) and
    very few are P1 (critical). Our PRIORITY_WEIGHTS enforce:
      P1=5%, P2=20%, P3=50%, P4=25%

    This matters for two reasons:
    1. The model sees what it will see in production (mostly P3/P4)
    2. The eval metrics reflect real-world performance (rare P1 tickets
       are HARDER to classify because the model sees fewer examples)

    random.choices() is different from random.choice():
      - random.choice(list)           → picks one item, uniform probability
      - random.choices(list, weights) → picks one item, weighted probability

    The [0] at the end is because random.choices() returns a LIST of
    results (it can pick multiple items). We only want one, so we take
    the first element.
    """
    return random.choices(
        population=list(Priority),
        weights=[PRIORITY_WEIGHTS[p] for p in Priority],
        k=1,  # pick 1 item
    )[0]


def generate_timestamp(fake: Faker) -> datetime:
    """
    Generate a realistic ticket creation timestamp.

    WHY NOT JUST datetime.now()?

    Because all 2000 tickets would have timestamps within seconds of each
    other, which doesn't look realistic. Instead, we spread them across
    the last 90 days during business hours (8 AM - 6 PM, weekdays).

    This doesn't affect the model (it only sees 'text'), but:
    1. It makes the dataset look professional to anyone inspecting it
    2. It enables future time-series analysis if you want it
    3. Interviewers who look at your data will notice the attention to detail
    """
    # fake.date_time_between() picks a random datetime in the range.
    # We use -90 days to "now" to simulate a rolling 3-month window.
    dt = fake.date_time_between(start_date="-90d", end_date="now")

    # Clamp to business hours: replace the hour with 8-18 range
    # This is a simplification - real tickets come in 24/7, but
    # the majority are during business hours. Good enough for synthetic data.
    business_hour = random.randint(8, 17)
    business_minute = random.randint(0, 59)
    return dt.replace(hour=business_hour, minute=business_minute, second=0)


def stratified_split(
    tickets: list[dict],
    ratios: dict[str, float],
) -> list[dict]:
    """
    Assign train/val/test splits while maintaining category × priority
    distribution in each split.

    WHY STRATIFIED INSTEAD OF RANDOM?

    With random splitting and 2000 tickets:
      - P1 tickets = ~100 (5% of 2000)
      - P1 + hardware = ~12-13 (100 / 8 categories)
      - Random test split of 12 tickets could easily give you 0 or 1

    That means your eval would have almost no P1-hardware test cases,
    making the accuracy number meaningless for that combination.

    Stratified splitting groups tickets by (category, priority) FIRST,
    then splits each group proportionally. This guarantees that if 5%
    of all tickets are P1, then ~5% of EACH split is P1.

    HOW IT WORKS:

    1. Group all tickets by (category, priority) → "strata"
    2. For each stratum, assign the first 80% to train, next 10% to val,
       last 10% to test
    3. This ensures proportional representation in every split

    Parameters
    ----------
    tickets : list[dict]
        All generated tickets (without split field yet)
    ratios : dict[str, float]
        {"train": 0.80, "val": 0.10, "test": 0.10}

    Returns
    -------
    list[dict]
        Same tickets, now with the 'split' field populated
    """
    # ── STEP 1: Group by (category, priority) ────────────────────────────
    #
    # defaultdict(list) creates a dict where missing keys automatically
    # get an empty list. This avoids checking "if key not in groups: groups[key] = []"
    # before every append. It's a common Python pattern for grouping.

    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for ticket in tickets:
        key = (ticket["category"], ticket["priority"])
        groups[key].append(ticket)

    # ── STEP 2: Split each group proportionally ──────────────────────────
    #
    # For a group of 25 tickets with 80/10/10 split:
    #   train_end = int(25 * 0.80) = 20  → tickets 0-19
    #   val_end   = int(25 * 0.90) = 22  → tickets 20-21
    #   test      = remaining              → tickets 22-24
    #
    # The "remaining" approach for test ensures no tickets are lost to
    # rounding. If we calculated all three with int(), we might lose
    # 1-2 tickets per group due to float→int truncation.

    split_names = list(ratios.keys())        # ["train", "val", "test"]
    cumulative = []
    running = 0.0
    for name in split_names:
        running += ratios[name]
        cumulative.append(running)
    # cumulative = [0.80, 0.90, 1.00]

    result = []
    for key, group in groups.items():
        n = len(group)
        # Shuffle within each group so the split isn't ordered by
        # generation time (which would be an information leak)
        random.shuffle(group)

        prev_idx = 0
        for i, name in enumerate(split_names):
            if i == len(split_names) - 1:
                # Last split gets everything remaining (avoids rounding loss)
                end_idx = n
            else:
                end_idx = int(n * cumulative[i])

            for ticket in group[prev_idx:end_idx]:
                ticket["split"] = name
                result.append(ticket)

            prev_idx = end_idx

    return result


# ─── MAIN GENERATOR FUNCTION ────────────────────────────────────────────────

def generate_dataset(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    output_dir: str = "data/raw",
) -> Path:
    """
    Generate a complete synthetic ITSM ticket dataset.

    This is the main entry point. It:
    1. Seeds all random generators for reproducibility
    2. Validates templates against the schema
    3. Generates N tickets with realistic distributions
    4. Splits into train/val/test (stratified)
    5. Writes one JSONL file per split
    6. Prints a summary report

    Parameters
    ----------
    n : int
        Total number of tickets to generate (default: 2000)
    seed : int
        Random seed for reproducibility (default: 42)
    output_dir : str
        Directory to write the JSONL files (default: data/raw)

    Returns
    -------
    Path
        The output directory path
    """
    console = Console()

    # ── SEED EVERYTHING ──────────────────────────────────────────────────
    #
    # WHY SEED BOTH random AND Faker?
    #
    # Python's random module and Faker have SEPARATE internal random
    # number generators. If you only seed random, Faker's output still
    # changes between runs. You need to seed both to get full
    # reproducibility.
    #
    # This is a common ML gotcha - many "reproducibility" bugs come from
    # forgetting to seed one of multiple RNG sources. In PyTorch, you'd
    # also need torch.manual_seed() and torch.cuda.manual_seed_all().
    # We don't need those here (no ML training in the generator), but
    # keep this pattern in mind for training scripts.

    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    console.print(f"\n[bold cyan]Generating {n} ITSM tickets (seed={seed})[/bold cyan]\n")

    # ── VALIDATE TEMPLATES ───────────────────────────────────────────────
    #
    # This calls the validation function from templates.py that checks:
    #   - Every category in the schema has a matching template
    #   - Every template has all required keys
    #   - Every key has enough entries for variety
    #
    # We do this BEFORE generating anything so a template bug is caught
    # immediately, not after 10 minutes of generation.

    _validate_templates()

    # ── GENERATE TICKETS ─────────────────────────────────────────────────
    #
    # We iterate over N tickets. For each ticket:
    #   1. Pick a category (uniform random - each category gets ~250 tickets)
    #   2. Pick a priority (weighted random - mirrors real distribution)
    #   3. Generate text from the category's templates
    #   4. Generate a realistic timestamp
    #   5. Create a Pydantic Ticket object (validates all fields)
    #   6. Convert to dict for later JSON serialization
    #
    # WHY UNIFORM CATEGORY DISTRIBUTION?
    #
    # Unlike priority (which is skewed), we want roughly equal numbers of
    # tickets per category. This is called a "balanced" dataset for
    # classification. If we had 1800 software tickets and only 10 printer
    # tickets, the model would just predict "software" for everything
    # and get 90% accuracy while being useless for printers.
    #
    # In a real production system, categories ARE skewed. But for TRAINING,
    # we balance them so the model learns all categories equally well.
    # This is standard ML practice called "balanced sampling."

    categories = list(Category)
    tickets_raw: list[dict] = []

    for i in range(n):
        # Uniform category selection
        cat = random.choice(categories)

        # Weighted priority selection
        pri = assign_priority(cat.value)

        # Generate text and action from templates (priority-correlated)
        text, action = generate_ticket_text(cat.value, pri.value, fake)

        # Generate a realistic timestamp
        ts = generate_timestamp(fake)

        # Create the Pydantic Ticket (validates all fields automatically).
        # If ANY field is invalid (wrong type, too short, unknown category),
        # Pydantic raises a ValidationError HERE - not downstream when
        # the model tries to train on bad data.
        ticket = Ticket(
            created_at=ts,
            text=text,
            category=cat,
            priority=pri,
            next_action=action,
            split="train",  # Placeholder - overwritten by stratified_split()
        )

        # .model_dump() converts the Pydantic object to a plain dict.
        #
        # mode="json" tells Pydantic to serialize complex types:
        #   - UUID → string ("550e8400-...")
        #   - datetime → ISO format string ("2026-03-01T09:15:00")
        #   - Enum → string value ("hardware" not "Category.HARDWARE")
        #
        # Without mode="json", UUIDs and datetimes would stay as Python
        # objects, which json.dumps() can't handle.
        tickets_raw.append(ticket.model_dump(mode="json"))

    # ── STRATIFIED SPLIT ─────────────────────────────────────────────────

    console.print("[dim]Applying stratified train/val/test split...[/dim]")
    tickets_final = stratified_split(tickets_raw, SPLIT_RATIOS)

    # ── WRITE JSONL FILES ────────────────────────────────────────────────
    #
    # JSONL = JSON Lines: one JSON object per line.
    #
    # WHY JSONL INSTEAD OF CSV?
    #   - Nested fields (like metadata) are trivial in JSON, painful in CSV
    #   - No quoting/escaping issues with commas in text fields
    #   - Pandas reads it: pd.read_json("train.jsonl", lines=True)
    #   - HuggingFace reads it: load_dataset("json", data_files=...)
    #   - You can stream it line by line without loading the whole file
    #
    # WHY ONE FILE PER SPLIT INSTEAD OF ONE FILE WITH A 'split' COLUMN?
    #   - Harder to accidentally load test data into training
    #   - Each file is self-contained - you can send just train.jsonl
    #   - load_dataset() can map files to splits automatically
    #
    # ensure_ascii=False: Allows non-ASCII characters in ticket text.
    #   Without this, "café" becomes "caf\u00e9" which is ugly and harder
    #   to debug when you inspect the data.

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = Counter()

    # Group tickets by split
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for t in tickets_final:
        by_split[t["split"]].append(t)

    for split_name, split_tickets in by_split.items():
        filepath = out / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for ticket in split_tickets:
                # json.dumps() converts a Python dict to a JSON string.
                # We write one per line → JSONL format.
                f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        split_counts[split_name] = len(split_tickets)

    # Also write the full dataset as a single file (useful for inspection)
    all_path = out / "all_tickets.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for ticket in tickets_final:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    # ── SUMMARY REPORT ───────────────────────────────────────────────────
    #
    # Print a nice table showing what was generated. This isn't just
    # cosmetic - it's a quick sanity check. If you see 0 tickets in
    # any category or the priority distribution looks wrong, you know
    # immediately that something is broken.

    console.print(f"\n[bold green]✓ Generated {n} tickets → {out}/[/bold green]\n")

    # Split summary
    split_table = RichTable(title="Split Summary")
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Count", justify="right")
    split_table.add_column("Percentage", justify="right")
    for name in ["train", "val", "test"]:
        count = split_counts[name]
        pct = f"{count / n * 100:.1f}%"
        split_table.add_row(name, str(count), pct)
    console.print(split_table)

    # Category distribution
    cat_counter = Counter(t["category"] for t in tickets_final)
    cat_table = RichTable(title="\nCategory Distribution")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Percentage", justify="right")
    for cat in Category:
        count = cat_counter[cat.value]
        pct = f"{count / n * 100:.1f}%"
        cat_table.add_row(cat.value, str(count), pct)
    console.print(cat_table)

    # Priority distribution
    pri_counter = Counter(t["priority"] for t in tickets_final)
    pri_table = RichTable(title="\nPriority Distribution (target: P1=5%, P2=20%, P3=50%, P4=25%)")
    pri_table.add_column("Priority", style="cyan")
    pri_table.add_column("Count", justify="right")
    pri_table.add_column("Actual %", justify="right")
    pri_table.add_column("Target %", justify="right")
    for pri in Priority:
        count = pri_counter[pri.value]
        actual = f"{count / n * 100:.1f}%"
        target = f"{PRIORITY_WEIGHTS[pri] * 100:.0f}%"
        pri_table.add_row(pri.value, str(count), actual, target)
    console.print(pri_table)

    # File paths
    console.print("\n[dim]Files written:[/dim]")
    for f in sorted(out.glob("*.jsonl")):
        size_kb = f.stat().st_size / 1024
        console.print(f"  [dim]{f}[/dim]  ({size_kb:.1f} KB)")

    console.print(
        f"\n[bold]Reproduce with:[/bold]  "
        f"python -m data.generator.gen --n {n} --seed {seed}\n"
        f"[dim]Priority labels are now correlated with ticket text (v2 templates).[/dim]\n"
    )

    return out


# ─── CLI ENTRY POINT ─────────────────────────────────────────────────────────
#
# WHY if __name__ == "__main__"?
#
# This is Python's standard pattern for "run this code only when the
# file is executed directly, not when it's imported." Without it,
# importing this module (e.g., for testing) would immediately generate
# 2000 tickets.
#
# argparse creates a command-line interface:
#   python -m data.generator.gen --n 500 --seed 123 --output data/raw
#
# Each add_argument() defines:
#   - The flag name (--n, --seed, --output)
#   - The type (int, str)
#   - The default value
#   - A help string (shown with python gen.py --help)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic ITSM tickets for model training."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of tickets to generate (default: {DEFAULT_N})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for JSONL files (default: data/raw)"
    )

    args = parser.parse_args()
    generate_dataset(n=args.n, seed=args.seed, output_dir=args.output)
