#!/usr/bin/env python3
"""Open suggested film URLs one by one and mark them as verified interactively."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "films_resolved.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "films_resolved.csv"))
    parser.add_argument("--storage-state", default=str(PROJECT_ROOT / "session" / "storage_state.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []

    storage_state = Path(args.storage_state)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context_args: dict[str, Any] = {}
        if storage_state.exists():
            context_args["storage_state"] = str(storage_state)
        context = browser.new_context(**context_args)
        page = context.new_page()

        for row in rows:
            url = row.get("resolved_url", "").strip()
            if not url or row.get("url_verified") == "1":
                continue

            print(f"\n{row.get('film_id')} — {row.get('film_ru')} / {row.get('film_en')}")
            print(url)
            page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            page.bring_to_front()

            while True:
                answer = input(
                    "[y] correct film, [n] wrong, [s] skip, [q] save and quit: "
                ).strip().casefold()
                if answer in {"y", "n", "s", "q"}:
                    break

            if answer == "y":
                row["url_verified"] = "1"
                row["resolution_notes"] = "verified_manually"
            elif answer == "n":
                row["url_verified"] = "0"
                row["resolution_notes"] = "wrong_match"
                row["resolved_url"] = ""
                row["rezka_path"] = ""
            elif answer == "q":
                break

        browser.close()

    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
