#!/usr/bin/env python3
"""Create a stratified film sample without turning integer IDs into floats."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "comments_raw.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "comments_sample_2555.csv"),
    )
    parser.add_argument("--max-per-film", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("film_id", ""))].append(row)

    rng = random.Random(args.seed)
    sampled = []
    for film_id in sorted(groups):
        group = groups[film_id]
        take = min(len(group), args.max_per_film)
        sampled.extend(rng.sample(group, take))

    rng.shuffle(sampled)

    # Preserve all raw fields and their textual form.
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sampled:
            cleaned = {
                key: "" if value is None else value
                for key, value in row.items()
            }
            writer.writerow(cleaned)

    print(f"Written: {output_path}")
    print(f"Rows: {len(sampled)}")
    print(f"Films: {len(groups)}")


if __name__ == "__main__":
    main()
