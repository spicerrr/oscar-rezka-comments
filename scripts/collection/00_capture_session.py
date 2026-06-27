#!/usr/bin/env python3
"""Open Rezka in a real browser, let the user log in, and save the session.

All default paths are resolved from the project folder, not from the current
terminal directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin",
        default="https://rezka.fi/",
        help="Working Rezka mirror, including scheme and trailing slash.",
    )
    parser.add_argument(
        "--film-url",
        default="https://rezka.fi/films/drama/81513-marti-velikolepnyy-2025-latest.html",
        help="A film page used to verify that login succeeded.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "session" / "storage_state.json"),
        help="Where to save Playwright cookies/local storage.",
    )
    return parser.parse_args()


def is_login_page(page) -> bool:
    try:
        return (
            page.locator('form[action="/ajax/login/"]').count() > 0
            or page.title().strip().casefold() == "вход"
        )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(args.origin, wait_until="domcontentloaded", timeout=90_000)

        print("\nВ открытом браузере:")
        print("1. Войди в аккаунт Rezka.")
        print("2. Вставь в адресную строку страницу фильма:")
        print(f"   {args.film_url}")
        print("3. Убедись, что открылась карточка фильма, а не страница «Вход».")
        print("4. Прокрути до блока комментариев и проверь, что он виден.")
        input("5. Вернись в терминал и нажми Enter... ")

        # Проверяем текущую страницу и при необходимости открываем фильм повторно.
        if is_login_page(page):
            print("\nСессия не подтверждена: браузер всё ещё на странице входа.")
            print("Войди, открой фильм и затем нажми Enter ещё раз.")
            input()
        page.goto(args.film_url, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(2_000)

        if is_login_page(page):
            browser.close()
            raise RuntimeError(
                "Вход не сохранился: страница фильма снова перенаправила на «Вход». "
                "Повтори запуск и проверь, что авторизация завершена до нажатия Enter."
            )

        context.storage_state(path=str(output))
        browser.close()

    payload = json.loads(output.read_text(encoding="utf-8"))
    print(f"\nСессия сохранена: {output}")
    print(f"Cookies в файле: {len(payload.get('cookies', []))}")
    print("Этот файл не загружай в чат и не коммить в Git.")


if __name__ == "__main__":
    main()
