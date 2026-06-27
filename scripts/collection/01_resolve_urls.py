#!/usr/bin/env python3
"""Resolve film pages through HdRezkaApi search.

The script does not silently approve ambiguous matches. It writes candidates and
an automatic suggestion; the final url_verified column must be checked manually.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from HdRezkaApi.search import HdRezkaSearch
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "films_master.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "films_resolved.csv"))
    parser.add_argument(
        "--origin",
        default="https://rezka.fi/",
        help="Working mirror, including scheme and trailing slash.",
    )
    parser.add_argument(
        "--storage-state",
        default=str(PROJECT_ROOT / "session" / "storage_state.json"),
        help="Playwright storage state created by 00_capture_session.py.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-score",
        type=float,
        default=78.0,
        help="Minimum score for an automatic suggestion, not automatic verification.",
    )
    return parser.parse_args()


def norm(value: str) -> str:
    value = value.casefold().replace("ё", "е")
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def canonical_path(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def cookies_from_storage_state(path: Path, origin: str) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    host = urlparse(origin).hostname or ""
    cookies: dict[str, str] = {}
    for item in payload.get("cookies", []):
        domain = str(item.get("domain", "")).lstrip(".")
        if host == domain or host.endswith("." + domain) or domain.endswith("." + host):
            cookies[str(item["name"])] = str(item["value"])
    return cookies


def flatten_results(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if hasattr(raw, "all"):
        value = raw.all
        return [x for x in value if isinstance(x, dict)]
    try:
        return [x for x in list(raw) if isinstance(x, dict)]
    except TypeError:
        return []


def score_candidate(row: dict[str, str], candidate: dict[str, Any]) -> float:
    title = str(candidate.get("title", ""))
    url = str(candidate.get("url", ""))
    ru = row.get("film_ru", "")
    en = row.get("film_en", "")
    year = row.get("year", "")

    title_score = max(
        fuzz.token_set_ratio(norm(ru), norm(title)),
        fuzz.token_set_ratio(norm(en), norm(title)),
    )
    year_bonus = 8 if year and year in f"{title} {url}" else 0
    film_bonus = 3 if "/films/" in url or "/cartoons/" in url else 0
    return min(100.0, float(title_score + year_bonus + film_bonus))


def search_queries(row: dict[str, str]) -> list[str]:
    year = row.get("year", "").strip()
    candidates = [
        f"{row.get('film_ru', '').strip()} {year}".strip(),
        f"{row.get('film_en', '').strip()} {year}".strip(),
        row.get("film_ru", "").strip(),
        row.get("film_en", "").strip(),
    ]
    seen: set[str] = set()
    result: list[str] = []
    for query in candidates:
        key = norm(query)
        if query and key not in seen:
            seen.add(key)
            result.append(query)
    return result


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
        source_fields = list(rows[0].keys()) if rows else []

    cookies = cookies_from_storage_state(Path(args.storage_state), args.origin)
    search = HdRezkaSearch(args.origin, cookies=cookies)

    extra_fields = [
        "rezka_path",
        "resolved_url",
        "resolution_score",
        "resolution_query",
        "candidate_urls_json",
        "url_verified",
        "resolution_notes",
    ]
    fields = source_fields + [field for field in extra_fields if field not in source_fields]

    for index, row in enumerate(rows, start=1):
        existing_url = row.get("rezka_page_url", "").strip()
        existing_path = canonical_path(existing_url)
        all_candidates: dict[str, dict[str, Any]] = {}

        if existing_url:
            all_candidates[existing_path or existing_url] = {
                "title": row.get("film_ru", ""),
                "url": urljoin(args.origin, existing_path),
                "query": "existing_master",
                "score": 100.0,
            }

        for query in search_queries(row):
            try:
                raw = search(query)
                results = flatten_results(raw)
            except Exception as exc:
                print(
                    f"[{index}/{len(rows)}] Search failed for {row.get('film_ru')!r}, "
                    f"query={query!r}: {exc}",
                    file=sys.stderr,
                )
                continue

            for candidate in results:
                url = str(candidate.get("url", "")).strip()
                if not url:
                    continue
                path = canonical_path(url)
                score = score_candidate(row, candidate)
                record = {
                    "title": str(candidate.get("title", "")),
                    "url": urljoin(args.origin, path),
                    "path": path,
                    "query": query,
                    "score": round(score, 1),
                    "rating": candidate.get("rating"),
                }
                previous = all_candidates.get(path)
                if previous is None or float(previous["score"]) < score:
                    all_candidates[path] = record

        ranked = sorted(
            all_candidates.values(),
            key=lambda item: float(item.get("score", 0)),
            reverse=True,
        )[: args.top_k]

        best = ranked[0] if ranked else {}
        score = float(best.get("score", 0))
        suggested = score >= args.min_score

        row["rezka_path"] = str(best.get("path") or canonical_path(str(best.get("url", ""))))
        row["resolved_url"] = str(best.get("url", "")) if suggested else ""
        row["resolution_score"] = str(best.get("score", "")) if best else ""
        row["resolution_query"] = str(best.get("query", "")) if best else ""
        row["candidate_urls_json"] = json.dumps(ranked, ensure_ascii=False)
        row["url_verified"] = "0"
        row["resolution_notes"] = (
            "manual_check_required" if suggested else "no_confident_match"
        )

        print(
            f"[{index}/{len(rows)}] {row.get('film_ru')}: "
            f"{row['resolved_url'] or 'NO MATCH'} ({row['resolution_score'] or '-'})"
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {output_path}")
    print("Open the CSV and set url_verified=1 only for confirmed film pages.")


if __name__ == "__main__":
    main()
