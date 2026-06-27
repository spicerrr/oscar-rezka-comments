#!/usr/bin/env python3
"""
Download film posters from the film URLs stored in comments_raw.jsonl.

Input:
    data/comments_raw.jsonl

Output:
    posters/film_001.jpg
    posters/film_002.jpg
    ...
    posters/poster_manifest.csv

The script reuses the Playwright session saved by 00_capture_session.py.
It does not bypass access controls. If the saved session has expired, run:
    python scripts/00_capture_session.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from PIL import Image
from playwright.sync_api import BrowserContext, Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "comments_raw.jsonl"),
        help="Raw comments JSONL. Film URLs are deduplicated by film_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "posters"),
    )
    parser.add_argument(
        "--storage-state",
        default=str(PROJECT_ROOT / "session" / "storage_state.json"),
        help="Playwright session created by 00_capture_session.py.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.2,
        help="Pause between film pages in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload posters that already exist.",
    )
    return parser.parse_args()


def read_films(path: Path) -> list[dict[str, str]]:
    films: dict[str, dict[str, str]] = {}

    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            film_id = str(row.get("film_id", "")).strip()
            film_url = str(row.get("film_url", "")).strip()
            if not film_id or not film_url:
                continue

            films.setdefault(
                film_id,
                {
                    "film_id": film_id,
                    "film_ru": str(row.get("film_ru", "")),
                    "film_en": str(row.get("film_en", "")),
                    "film_url": film_url,
                },
            )

    return sorted(films.values(), key=lambda item: item["film_id"])


def is_login_page(page: Page) -> bool:
    try:
        return (
            page.locator('form[action="/ajax/login/"]').count() > 0
            or page.title().strip().casefold() == "вход"
        )
    except Exception:
        return False


def absolute_url(page_url: str, value: str | None) -> str:
    if not value:
        return ""
    return urljoin(page_url, value.strip())


def poster_candidates(page: Page) -> list[dict[str, Any]]:
    """
    Collect explicit poster candidates and visible portrait images.
    A score is used because mirrors can slightly change CSS classes.
    """
    return page.evaluate(
        """
        () => {
          const output = [];
          const seen = new Set();

          function add(url, score, source, width = 0, height = 0) {
            if (!url) return;
            try {
              url = new URL(url, location.href).href;
            } catch (_) {
              return;
            }
            if (seen.has(url)) return;
            seen.add(url);

            const lower = url.toLowerCase();
            if (
              lower.includes('avatar') ||
              lower.includes('logo') ||
              lower.includes('smile') ||
              lower.includes('emoji') ||
              lower.includes('favicon')
            ) return;

            if (lower.includes('/uploads/posts/')) score += 35;
            if (lower.includes('poster') || lower.includes('cover')) score += 18;
            if (height > width * 1.15) score += 25;
            if (width >= 180 && height >= 250) score += 20;

            output.push({url, score, source, width, height});
          }

          const og = document.querySelector('meta[property="og:image"]');
          if (og) add(og.content, 80, 'og:image');

          const twitter = document.querySelector('meta[name="twitter:image"]');
          if (twitter) add(twitter.content, 70, 'twitter:image');

          const selectors = [
            '.b-sidecover img',
            '.b-post__infotable_left img',
            '.b-content__main img[itemprop="image"]',
            'img[itemprop="image"]',
            '.b-content__main img',
            '.b-sidecover'
          ];

          selectors.forEach((selector, index) => {
            document.querySelectorAll(selector).forEach((img) => {
              const url =
                img.currentSrc ||
                img.src ||
                img.dataset.src ||
                img.getAttribute('data-original') ||
                img.getAttribute('data-src');
              add(
                url,
                65 - index * 4,
                selector,
                img.naturalWidth || img.clientWidth || 0,
                img.naturalHeight || img.clientHeight || 0
              );
            });
          });

          document.querySelectorAll('img').forEach((img) => {
            const rect = img.getBoundingClientRect();
            if (rect.width < 120 || rect.height < 170) return;
            const url =
              img.currentSrc ||
              img.src ||
              img.dataset.src ||
              img.getAttribute('data-original') ||
              img.getAttribute('data-src');
            add(
              url,
              10,
              'all-visible-images',
              img.naturalWidth || rect.width || 0,
              img.naturalHeight || rect.height || 0
            );
          });

          return output.sort((a, b) => b.score - a.score);
        }
        """
    )


def save_as_jpeg(content: bytes, output: Path) -> None:
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
        # Do not upscale; only reduce very large source images.
        if image.width > 1600:
            new_height = round(image.height * 1600 / image.width)
            image = image.resize((1600, new_height), Image.Resampling.LANCZOS)
        image.save(output, "JPEG", quality=94, optimize=True)


def download_image(
    context: BrowserContext,
    image_url: str,
    referer: str,
) -> bytes:
    response = context.request.get(
        image_url,
        headers={
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": referer,
        },
        timeout=90_000,
    )
    if response.status != 200:
        raise RuntimeError(f"Poster HTTP {response.status}")
    content_type = response.headers.get("content-type", "")
    if "image" not in content_type.casefold():
        raise RuntimeError(f"Unexpected content type: {content_type}")
    return response.body()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    storage_state = Path(args.storage_state).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    films = read_films(input_path)

    if not films:
        raise SystemExit("No films with film_id and film_url found.")

    context_args: dict[str, Any] = {
        "viewport": {"width": 1440, "height": 1100},
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
    }
    if storage_state.exists():
        context_args["storage_state"] = str(storage_state)

    manifest: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(**context_args)
        page = context.new_page()

        for index, film in enumerate(films, start=1):
            film_id = film["film_id"]
            output_path = output_dir / f"{film_id}.jpg"

            if output_path.exists() and not args.overwrite:
                print(f"[{index}/{len(films)}] {film_id}: already exists")
                manifest.append(
                    {
                        **film,
                        "status": "exists",
                        "poster_url": "",
                        "output_file": str(output_path),
                        "error": "",
                    }
                )
                continue

            status = "error"
            poster_url = ""
            error = ""

            try:
                page.goto(
                    film["film_url"],
                    wait_until="domcontentloaded",
                    timeout=90_000,
                )
                page.wait_for_timeout(1_500)

                if is_login_page(page):
                    raise RuntimeError(
                        "Saved Rezka session has expired. "
                        "Run scripts/00_capture_session.py again."
                    )

                # Give lazy-loaded images a chance to appear.
                page.evaluate("window.scrollTo(0, 500)")
                page.wait_for_timeout(800)

                candidates = poster_candidates(page)
                if not candidates:
                    raise RuntimeError("No poster image candidate found.")

                last_error: Exception | None = None
                for candidate in candidates[:8]:
                    try:
                        poster_url = absolute_url(page.url, candidate["url"])
                        content = download_image(context, poster_url, page.url)
                        save_as_jpeg(content, output_path)
                        status = "downloaded"
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc

                if status != "downloaded":
                    raise RuntimeError(
                        f"Poster candidates failed: {last_error}"
                    )

                print(
                    f"[{index}/{len(films)}] {film_id}: "
                    f"{film['film_ru']} -> {output_path.name}"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{index}/{len(films)}] {film_id}: ERROR: {error}"
                )

            manifest.append(
                {
                    **film,
                    "status": status,
                    "poster_url": poster_url,
                    "output_file": str(output_path) if output_path.exists() else "",
                    "error": error,
                }
            )
            time.sleep(max(0.0, args.delay))

        # Refresh the saved session if it still works.
        storage_state.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(storage_state))
        browser.close()

    manifest_path = output_dir / "poster_manifest.csv"
    with manifest_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "film_id",
                "film_ru",
                "film_en",
                "film_url",
                "status",
                "poster_url",
                "output_file",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    downloaded = sum(item["status"] == "downloaded" for item in manifest)
    existing = sum(item["status"] == "exists" for item in manifest)
    failed = len(manifest) - downloaded - existing

    print("\nDone.")
    print(f"Downloaded: {downloaded}")
    print(f"Already existed: {existing}")
    print(f"Failed: {failed}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
