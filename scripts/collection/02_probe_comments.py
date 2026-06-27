#!/usr/bin/env python3
"""Probe one confirmed film page and discover how comments are loaded.

If the saved browser session is missing or expired, the script pauses and lets
the user log in manually. All default paths are relative to the project folder,
not to the current terminal directory.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, Response, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIKELY_SELECTORS = [
    ".comments-tree-item",
    ".comments-tree-list > li",
    ".comments-list .comment",
    ".comment",
    "[data-comment-id]",
    "[id^='comment-id-']",
    "[id^='comment_']",
    "[class*='comment-item']",
]

LOAD_MORE_PATTERNS = [
    re.compile(r"показать\s+ещ[её]", re.I),
    re.compile(r"загрузить\s+ещ[её]", re.I),
    re.compile(r"ещ[её]\s+комментар", re.I),
    re.compile(r"все\s+комментар", re.I),
    re.compile(r"load\s+more", re.I),
    re.compile(r"show\s+more", re.I),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Confirmed film URL.")
    parser.add_argument(
        "--storage-state",
        default=str(PROJECT_ROOT / "session" / "storage_state.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "probe"),
    )
    parser.add_argument("--max-clicks", type=int, default=30)
    return parser.parse_args()


def is_login_page(page: Page) -> bool:
    try:
        return (
            page.locator('form[action="/ajax/login/"]').count() > 0
            or page.title().strip().casefold() == "вход"
        )
    except Exception:
        return False


def looks_relevant(url: str) -> bool:
    lowered = url.casefold()
    keys = ("comment", "comments", "ajax", "dle", "load")
    return any(key in lowered for key in keys)


def safe_response_text(response: Response) -> str | None:
    try:
        content_type = response.headers.get("content-type", "")
        if not any(kind in content_type for kind in ("json", "text", "html", "javascript")):
            return None
        return response.text()[:1_000_000]
    except Exception:
        return None


def scroll_to_bottom(page: Page) -> None:
    previous_height = -1
    stable_rounds = 0
    for _ in range(25):
        height = page.evaluate("document.body.scrollHeight")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(500)
        if height == previous_height:
            stable_rounds += 1
        else:
            stable_rounds = 0
        previous_height = height
        if stable_rounds >= 3:
            break


def click_load_more(page: Page, max_clicks: int) -> list[str]:
    clicked: list[str] = []
    for _ in range(max_clicks):
        scroll_to_bottom(page)
        found = False
        for pattern in LOAD_MORE_PATTERNS:
            locator = page.get_by_text(pattern).last
            try:
                if locator.count() and locator.is_visible():
                    text = locator.inner_text(timeout=2_000).strip()
                    locator.click(timeout=5_000)
                    page.wait_for_timeout(1_000)
                    clicked.append(text)
                    found = True
                    break
            except Exception:
                continue
        if not found:
            break
    return clicked


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_state = Path(args.storage_state).expanduser().resolve()
    storage_state.parent.mkdir(parents=True, exist_ok=True)

    network: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context_args: dict[str, Any] = {}
        if storage_state.exists():
            context_args["storage_state"] = str(storage_state)
        context = browser.new_context(**context_args)
        page = context.new_page()

        def on_request(request: Any) -> None:
            if looks_relevant(request.url):
                network.append(
                    {
                        "kind": "request",
                        "method": request.method,
                        "url": request.url,
                        "post_data": request.post_data,
                        "resource_type": request.resource_type,
                    }
                )

        def on_response(response: Response) -> None:
            if looks_relevant(response.url):
                network.append(
                    {
                        "kind": "response",
                        "status": response.status,
                        "url": response.url,
                        "headers": dict(response.headers),
                        "body_preview": safe_response_text(response),
                    }
                )

        page.on("request", on_request)
        page.on("response", on_response)

        page.goto(args.url, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(2_000)

        if is_login_page(page):
            print("\nСохранённой авторизации нет или она истекла.")
            print("В открытом браузере войди в аккаунт.")
            print("После входа вручную открой эту страницу:")
            print(args.url)
            print("Проверь, что видны карточка фильма и комментарии.")
            input("Вернись в терминал и нажми Enter... ")

            # Сохраняем новую сессию и проверяем страницу повторно.
            context.storage_state(path=str(storage_state))
            page.goto(args.url, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_timeout(2_000)

        if is_login_page(page):
            browser.close()
            raise RuntimeError(
                "Авторизация не сработала: вместо фильма снова открылась страница «Вход»."
            )

        clicked = click_load_more(page, args.max_clicks)
        scroll_to_bottom(page)
        page.wait_for_timeout(2_000)

        selector_counts = {}
        for selector in LIKELY_SELECTORS:
            try:
                selector_counts[selector] = page.locator(selector).count()
            except Exception:
                selector_counts[selector] = -1

        commentish_dom = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('[class*="comment" i], [id*="comment" i]'))
                .slice(0, 1000)
                .map((el) => ({
                    tag: el.tagName,
                    id: el.id || null,
                    className: typeof el.className === 'string' ? el.className : null,
                    textPreview: (el.innerText || '').trim().slice(0, 500)
                }))
            """
        )

        (output_dir / "page.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(output_dir / "page.png"), full_page=True)
        (output_dir / "network.json").write_text(
            json.dumps(network, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "selector_counts.json").write_text(
            json.dumps(selector_counts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "commentish_dom.json").write_text(
            json.dumps(commentish_dom, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "probe_summary.json").write_text(
            json.dumps(
                {
                    "url": args.url,
                    "page_title": page.title(),
                    "final_url": page.url,
                    "clicked_load_more": clicked,
                    "selector_counts": selector_counts,
                    "network_records": len(network),
                    "commentish_elements": len(commentish_dom),
                    "timestamp_unix": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        context.storage_state(path=str(storage_state))
        browser.close()

    print(f"\nProbe complete: {output_dir}")
    print(f"Session state: {storage_state}")


if __name__ == "__main__":
    main()
