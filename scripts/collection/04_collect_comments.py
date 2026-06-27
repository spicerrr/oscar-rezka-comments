#!/usr/bin/env python3
"""Collect every comments page for confirmed Rezka film pages.

Observed Rezka comments endpoint:
    /ajax/get_comments/
        ?news_id=<film id>
        &cstart=<page number>
        &type=0
        &comment_id=0
        &skin=hdrezka
        &t=<unix milliseconds>

The first response contains both comments HTML and pagination. The collector
reads the maximum cstart value and requests every page directly; it does not
click the numbered pagination in the browser.

Default paths are resolved from the project folder, not from the terminal cwd.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import secrets
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from playwright.sync_api import APIResponse, BrowserContext, Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOSCOW = ZoneInfo("Europe/Moscow")
MONTHS_RU = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


@dataclass
class Film:
    film_id: str
    film_ru: str
    film_en: str
    year: str
    url: str
    nomination_codes: str
    verified: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "films_resolved.csv"),
        help="Film registry produced by 01_resolve_urls.py.",
    )
    parser.add_argument(
        "--storage-state",
        default=str(PROJECT_ROOT / "session" / "storage_state.json"),
        help="Saved Playwright session.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "comments_raw.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(PROJECT_ROOT / "data" / "comments_raw.jsonl"),
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "artifacts" / "comments_raw"),
        help="One untouched JSON response per film and page.",
    )
    parser.add_argument(
        "--log",
        default=str(PROJECT_ROOT / "data" / "collection_log.csv"),
    )
    parser.add_argument(
        "--film-id",
        action="append",
        default=[],
        help="Collect only selected film_id. Repeat for several films.",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Also collect nonempty resolved_url rows with url_verified != 1.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Diagnostic cap. Omit to collect all pages.",
    )
    parser.add_argument("--delay-min", type=float, default=0.8)
    parser.add_argument("--delay-max", type=float, default=1.5)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium without a visible window. Requires a valid saved session.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch pages even when raw JSON already exists.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(MOSCOW).isoformat(timespec="seconds")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def is_login_page(page: Page) -> bool:
    try:
        return (
            page.locator('form[action="/ajax/login/"]').count() > 0
            or page.title().strip().casefold() == "вход"
            or "/login" in page.url.casefold()
        )
    except Exception:
        return False


def film_news_id(url: str) -> str:
    match = re.search(r"/(\d+)-[^/]+\.html(?:[?#].*)?$", url)
    if not match:
        raise ValueError(f"Cannot extract news_id from URL: {url}")
    return match.group(1)


def origin_from_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def choose_url(row: dict[str, str]) -> str:
    return (
        row.get("resolved_url", "").strip()
        or row.get("rezka_page_url", "").strip()
    )


def load_films(path: Path, include_unverified: bool, selected: set[str]) -> list[Film]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    films: list[Film] = []
    for row in rows:
        url = choose_url(row)
        verified = row.get("url_verified", "").strip() == "1"
        film_id = row.get("film_id", "").strip()

        if not url:
            continue
        if selected and film_id not in selected:
            continue
        if not include_unverified and not verified:
            continue

        films.append(
            Film(
                film_id=film_id,
                film_ru=row.get("film_ru", "").strip(),
                film_en=row.get("film_en", "").strip(),
                year=row.get("year", "").strip(),
                url=url,
                nomination_codes=row.get("nomination_codes", "").strip(),
                verified=verified,
            )
        )
    return films


def load_or_create_salt(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    salt = secrets.token_hex(32)
    path.write_text(salt, encoding="utf-8")
    return salt


def hash_author(author: str, salt: str) -> str:
    payload = f"{salt}\0{author.casefold().strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_russian_date(raw: str, collected_at: datetime) -> str:
    """Best-effort conversion; the original value is always preserved."""
    value = normalize_space(raw)
    value = re.sub(r"^оставлен\s+", "", value, flags=re.I)

    time_match = re.search(r"(\d{1,2}):(\d{2})$", value)
    hour = int(time_match.group(1)) if time_match else 0
    minute = int(time_match.group(2)) if time_match else 0

    lowered = value.casefold()
    if lowered.startswith("сегодня"):
        dt = collected_at.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt.isoformat(timespec="minutes")
    if lowered.startswith("вчера"):
        dt = (collected_at - timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        return dt.isoformat(timespec="minutes")

    match = re.search(
        r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})(?:\s+(\d{1,2}):(\d{2}))?",
        lowered,
    )
    if not match:
        return ""

    day = int(match.group(1))
    month = MONTHS_RU.get(match.group(2))
    year = int(match.group(3))
    if month is None:
        return ""

    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    return datetime(year, month, day, hour, minute, tzinfo=MOSCOW).isoformat(
        timespec="minutes"
    )


def max_page_from_navigation(navigation_html: str) -> int:
    pages = [
        int(x)
        for x in re.findall(r"loadComments\(\d+,\s*(\d+)", navigation_html)
    ]
    soup = BeautifulSoup(navigation_html, "lxml")
    pages.extend(
        int(text)
        for text in (
            normalize_space(node.get_text())
            for node in soup.find_all(["a", "span"])
        )
        if text.isdigit()
    )
    return max(pages, default=1)


def direct_child_comment_container(li: Any, comment_id: str) -> Any:
    return li.find("div", id=f"comment-id-{comment_id}", recursive=False)


def parse_comments_html(
    comments_html: str,
    *,
    film: Film,
    page_number: int,
    endpoint_url: str,
    response_sha256: str,
    collected_at: datetime,
    salt: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(comments_html, "lxml")
    result: list[dict[str, Any]] = []

    for li in soup.select("li.comments-tree-item[data-id]"):
        comment_id = str(li.get("data-id", "")).strip()
        if not comment_id:
            continue

        container = direct_child_comment_container(li, comment_id)
        if container is None:
            continue

        message = container.select_one(":scope > .b-comment > .message")
        if message is None:
            message = container.select_one(".b-comment .message")
        if message is None:
            continue

        author_node = message.select_one(":scope > .info > .name")
        date_node = message.select_one(":scope > .info > .date")
        text_node = message.find(id=f"comm-id-{comment_id}")
        like_node = message.select_one(
            f'.b-comment__like_it[data-comment_id="{comment_id}"]'
        )

        author = normalize_space(author_node.get_text(" ", strip=True) if author_node else "")
        date_raw = normalize_space(date_node.get_text(" ", strip=True) if date_node else "")
        comment_text = normalize_space(text_node.get_text(" ", strip=True) if text_node else "")

        likes_raw = like_node.get("data-likes_num", "0") if like_node else "0"
        try:
            likes = int(str(likes_raw).strip())
        except ValueError:
            likes = 0

        parent_li = li.find_parent("li", class_="comments-tree-item")
        parent_comment_id = (
            str(parent_li.get("data-id", "")).strip() if parent_li else ""
        )

        try:
            thread_depth = int(str(li.get("data-indent", "0")).strip())
        except ValueError:
            thread_depth = 0

        result.append(
            {
                "comment_id": comment_id,
                "film_id": film.film_id,
                "film_ru": film.film_ru,
                "film_en": film.film_en,
                "film_year": film.year,
                "film_url": film.url,
                "nomination_codes": film.nomination_codes,
                "comments_page": page_number,
                "author_hash": hash_author(author, salt) if author else "",
                "published_at_raw": date_raw,
                "published_at_iso": parse_russian_date(date_raw, collected_at),
                "comment_text": comment_text,
                "likes": likes,
                "parent_comment_id": parent_comment_id,
                "thread_depth": thread_depth,
                "is_reply": int(bool(parent_comment_id or thread_depth > 0)),
                "comment_url": f"{film.url}#comment{comment_id}",
                "collected_at": collected_at.isoformat(timespec="seconds"),
                "source_endpoint": endpoint_url,
                "source_response_sha256": response_sha256,
            }
        )

    return result


def request_json_with_retries(
    context: BrowserContext,
    url: str,
    retries: int,
) -> tuple[dict[str, Any], bytes]:
    delay = 2.0
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response: APIResponse = context.request.get(
                url,
                headers={
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": origin_from_url(url),
                },
                timeout=90_000,
            )
            body = response.body()

            if response.status == 200:
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict) or "comments" not in payload:
                    raise ValueError("Unexpected comments response structure")
                return payload, body

            if response.status in {403, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status}")

            raise RuntimeError(f"HTTP {response.status}: {body[:300]!r}")
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def comments_endpoint(film_url: str, page_number: int) -> str:
    news_id = film_news_id(film_url)
    origin = origin_from_url(film_url)
    timestamp_ms = int(time.time() * 1000)
    return urljoin(
        origin,
        (
            "ajax/get_comments/"
            f"?t={timestamp_ms}"
            f"&news_id={news_id}"
            f"&cstart={page_number}"
            "&type=0"
            "&comment_id=0"
            "&skin=hdrezka"
        ),
    )


def save_raw_page(path: Path, payload: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return raw


def load_raw_page(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def write_dataset(rows: list[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "comment_id",
        "film_id",
        "film_ru",
        "film_en",
        "film_year",
        "film_url",
        "nomination_codes",
        "comments_page",
        "author_hash",
        "published_at_raw",
        "published_at_iso",
        "comment_text",
        "likes",
        "parent_comment_id",
        "thread_depth",
        "is_reply",
        "comment_url",
        "collected_at",
        "source_endpoint",
        "source_response_sha256",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_log(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "film_id",
        "film_ru",
        "status",
        "pages_expected",
        "pages_saved",
        "comments_parsed",
        "started_at",
        "finished_at",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_authenticated(
    context: BrowserContext,
    page: Page,
    first_film_url: str,
    storage_state: Path,
    headless: bool,
) -> None:
    page.goto(first_film_url, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(1_500)

    if not is_login_page(page):
        return

    if headless:
        raise RuntimeError(
            "Saved session is missing or expired. Run without --headless, "
            "log in in the opened browser, then continue."
        )

    print("\nRezka redirected to the login page.")
    print("Log in in the opened browser, then open:")
    print(first_film_url)
    print("Make sure the film page is visible.")
    input("Return to the terminal and press Enter... ")

    context.storage_state(path=str(storage_state))
    page.goto(first_film_url, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(1_500)

    if is_login_page(page):
        raise RuntimeError("Authentication failed: the film still redirects to login.")

    context.storage_state(path=str(storage_state))


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    storage_state = Path(args.storage_state).expanduser().resolve()
    output_csv = Path(args.output).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    salt_path = PROJECT_ROOT / "session" / "author_salt.txt"

    films = load_films(
        input_path,
        include_unverified=args.include_unverified,
        selected=set(args.film_id),
    )
    if not films:
        raise SystemExit(
            "No films selected. Verify URLs with 03_validate_urls.py, or pass "
            "--include-unverified for a diagnostic run."
        )

    salt = load_or_create_salt(salt_path)
    all_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    context_args: dict[str, Any] = {}
    if storage_state.exists():
        context_args["storage_state"] = str(storage_state)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(**context_args)
        page = context.new_page()
        ensure_authenticated(
            context,
            page,
            films[0].url,
            storage_state,
            args.headless,
        )

        for film_index, film in enumerate(films, start=1):
            started_at = now_iso()
            pages_saved = 0
            pages_expected = 0
            film_rows: list[dict[str, Any]] = []
            status = "ok"
            error = ""

            print(f"\n[{film_index}/{len(films)}] {film.film_ru}")

            try:
                first_raw_path = raw_dir / film.film_id / "page_001.json"
                first_endpoint = comments_endpoint(film.url, 1)

                if first_raw_path.exists() and not args.refresh:
                    payload, raw = load_raw_page(first_raw_path)
                    print("  page 1: cache")
                else:
                    payload, _network_raw = request_json_with_retries(
                        context, first_endpoint, args.retries
                    )
                    raw = save_raw_page(first_raw_path, payload)
                    print("  page 1: fetched")
                    time.sleep(random.uniform(args.delay_min, args.delay_max))

                pages_expected = max_page_from_navigation(
                    str(payload.get("navigation", ""))
                )
                if args.max_pages is not None:
                    pages_expected = min(pages_expected, args.max_pages)

                print(f"  pages: {pages_expected}")

                collected_at = datetime.now(MOSCOW)
                film_rows.extend(
                    parse_comments_html(
                        str(payload.get("comments", "")),
                        film=film,
                        page_number=1,
                        endpoint_url=first_endpoint,
                        response_sha256=hashlib.sha256(raw).hexdigest(),
                        collected_at=collected_at,
                        salt=salt,
                    )
                )
                pages_saved = 1

                for page_number in range(2, pages_expected + 1):
                    raw_path = raw_dir / film.film_id / f"page_{page_number:03d}.json"
                    endpoint = comments_endpoint(film.url, page_number)

                    if raw_path.exists() and not args.refresh:
                        page_payload, page_raw = load_raw_page(raw_path)
                        source = "cache"
                    else:
                        page_payload, _network_raw = request_json_with_retries(
                            context, endpoint, args.retries
                        )
                        page_raw = save_raw_page(raw_path, page_payload)
                        source = "fetched"
                        time.sleep(random.uniform(args.delay_min, args.delay_max))

                    parsed = parse_comments_html(
                        str(page_payload.get("comments", "")),
                        film=film,
                        page_number=page_number,
                        endpoint_url=endpoint,
                        response_sha256=hashlib.sha256(page_raw).hexdigest(),
                        collected_at=datetime.now(MOSCOW),
                        salt=salt,
                    )
                    film_rows.extend(parsed)
                    pages_saved += 1
                    print(
                        f"  page {page_number}/{pages_expected}: "
                        f"{source}, comments={len(parsed)}"
                    )

                # Dedupe within the film while preserving first occurrence.
                deduped: dict[str, dict[str, Any]] = {}
                for row in film_rows:
                    deduped.setdefault(str(row["comment_id"]), row)
                film_rows = list(deduped.values())
                all_rows.extend(film_rows)

                print(f"  total unique comments: {len(film_rows)}")

            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR: {error}", file=sys.stderr)

            log_rows.append(
                {
                    "film_id": film.film_id,
                    "film_ru": film.film_ru,
                    "status": status,
                    "pages_expected": pages_expected,
                    "pages_saved": pages_saved,
                    "comments_parsed": len(film_rows),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "error": error,
                }
            )

            # Safe checkpoint after each film.
            write_dataset(all_rows, output_csv, output_jsonl)
            write_log(log_rows, log_path)

        context.storage_state(path=str(storage_state))
        browser.close()

    print("\nDone.")
    print(f"CSV:   {output_csv}")
    print(f"JSONL: {output_jsonl}")
    print(f"Log:   {log_path}")
    print(f"Rows:  {len(all_rows)}")


if __name__ == "__main__":
    main()
