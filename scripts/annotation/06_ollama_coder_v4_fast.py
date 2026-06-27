#!/usr/bin/env python3
"""
Fast and fault-tolerant local coding with Ollama + qwen3:8b.

Main changes:
- compact output codes reduce generation time;
- valid items are saved even if the model omits another ID;
- only missing IDs are retried;
- failed batches are split automatically;
- single-row failures are logged and skipped instead of crashing;
- Ctrl+C finalizes current checkpoint and can be resumed;
- frames and secondary flags are derived deterministically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "4.0.0-fast"

REL = {"F", "T", "L", "U", "E", "S", "Q"}
VAL = {"P", "N", "M", "0", "Q", "X"}
EMO = {"AD", "EN", "AM", "SE", "FT", "DI", "IA", "BO", "DG", "CF"}
PRIMARY = EMO | {"NE", "Q", "X"}
TARGET = {"FO", "PL", "AC", "CH", "VI", "MU", "PA", "ED", "AF", "IR", "AW", "OT"}
RHET = {"LI", "IS", "MO", "HY", "RQ", "Q"}
CMP = {"B", "W", "S", "M", "C", "X"}
OSC = {"D", "U", "A", "C", "N", "Q"}

REL_MAP = {
    "F": "film_reaction",
    "T": "technical_platform",
    "L": "release_or_translation_request",
    "U": "interpersonal_only",
    "E": "trailer_or_expectation",
    "S": "spam_or_irrelevant",
    "Q": "unclear",
}
VAL_MAP = {
    "P": "positive",
    "N": "negative",
    "M": "mixed",
    "0": "neutral",
    "Q": "unclear",
    "X": "not_applicable",
}
EMO_MAP = {
    "AD": "admiration",
    "EN": "enjoyment",
    "AM": "amusement",
    "SE": "sadness_or_empathy",
    "FT": "fear_or_tension",
    "DI": "disappointment",
    "IA": "irritation_or_anger",
    "BO": "boredom",
    "DG": "disgust",
    "CF": "confusion",
    "NE": "no_explicit_emotion",
    "Q": "unclear",
    "X": "not_applicable",
}
TARGET_MAP = {
    "FO": "film_overall",
    "PL": "plot",
    "AC": "acting",
    "CH": "characters",
    "VI": "visuals",
    "MU": "music_or_sound",
    "PA": "pace_or_duration",
    "ED": "ending",
    "AF": "adaptation_or_franchise",
    "IR": "ideology_or_representation",
    "AW": "award_worthiness",
    "OT": "other",
}
RHET_MAP = {
    "LI": "literal",
    "IS": "irony_or_sarcasm",
    "MO": "mockery",
    "HY": "hyperbole",
    "RQ": "rhetorical_question",
    "Q": "unclear",
}
CMP_MAP = {
    "B": "better",
    "W": "worse",
    "S": "similar",
    "M": "mixed",
    "C": "no_clear_ranking",
    "X": "not_applicable",
}
OSC_MAP = {
    "D": "deserved",
    "U": "undeserved",
    "A": "ambivalent",
    "C": "discussion_without_position",
    "N": "not_mentioned",
    "Q": "unclear",
}

OUTPUT_FIELDS = [
    "contains_film_reaction",
    "relevance_primary",
    "valence",
    "emotions",
    "primary_emotion",
    "praise_targets",
    "criticism_targets",
    "frames",
    "rhetorical_modes",
    "comparison_present",
    "comparison_result",
    "oscar_stance",
    "intensity",
    "mentions_platform_issue",
    "interpersonal_component",
    "pre_release_or_trailer",
    "needs_human_review",
    "evidence_fragment",
]


@dataclass
class Task:
    indexes: list[int]
    tries: int = 0


@dataclass
class Metrics:
    started_at: float
    completed: int = 0
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "comments_sample_2555.csv"),
    )
    p.add_argument("--job-name", default="sample_2555_v4")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument(
        "--batch-char-budget",
        type=int,
        default=12000,
        help="Maximum total comment characters per request.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=2500)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--single-retries", type=int, default=2)
    p.add_argument("--restart", action="store_true")
    p.add_argument(
        "--with-evidence",
        action="store_true",
        help="Ask for a short evidence quote. Slower; use for validation only.",
    )
    return p.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError("Input must be CSV or JSONL.")


def stable_id(row: dict[str, Any], index: int) -> str:
    value = str(row.get("comment_id", "")).strip()
    if value:
        return value.removesuffix(".0")
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(f"{index}\0{raw}".encode()).hexdigest()[:24]


def request_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama and retry."
        ) from exc


def get_json(url: str, timeout: int = 15) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_model(base_url: str, model: str) -> None:
    try:
        data = get_json(f"{base_url.rstrip('/')}/api/tags")
    except Exception as exc:
        raise SystemExit(
            "Ollama is not reachable. Open Ollama or run `ollama serve`."
        ) from exc
    names = {str(x.get("name", "")) for x in data.get("models", [])}
    if model not in names:
        raise SystemExit(f"Model {model} is missing. Run: ollama pull {model}")


def enum_array(values: set[str], max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": sorted(values)},
        "uniqueItems": True,
        "maxItems": max_items,
    }


def schema(with_evidence: bool) -> dict[str, Any]:
    properties = {
        "i": {"type": "string"},
        "r": {"type": "string", "enum": sorted(REL)},
        "v": {"type": "string", "enum": sorted(VAL)},
        "e": enum_array(EMO, 2),
        "pe": {"type": "string", "enum": sorted(PRIMARY)},
        "p": enum_array(TARGET, 4),
        "c": enum_array(TARGET, 4),
        "rm": enum_array(RHET, 2),
        "cr": {"type": "string", "enum": sorted(CMP)},
        "os": {"type": "string", "enum": sorted(OSC)},
        "z": {"type": "integer", "minimum": 0, "maximum": 3},
        "rev": {"type": "boolean"},
    }
    required = list(properties)
    if with_evidence:
        properties["q"] = {"type": "string", "maxLength": 100}
        required.append("q")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": required,
                },
            }
        },
        "required": ["x"],
    }


def clean_json(content: str) -> str:
    value = content.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.S)
    return value.strip()


def valid_list(value: Any, allowed: set[str], max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        item = str(item)
        if item in allowed and item not in result:
            result.append(item)
        if len(result) == max_items:
            break
    return result


def derive_frames(
    emotions: list[str],
    praise: list[str],
    criticism: list[str],
    comparison: str,
    oscar: str,
) -> list[str]:
    targets = set(praise) | set(criticism)
    frames = []
    if targets & {"visuals", "music_or_sound"}:
        frames.append("aesthetic_craft")
    if targets & {"plot", "pace_or_duration", "ending"}:
        frames.append("narrative_quality")
    if targets & {"acting", "characters"}:
        frames.append("performance")
    if emotions or "film_overall" in targets:
        frames.append("entertainment_and_affect")
    if "ideology_or_representation" in targets:
        frames.append("ideology_and_representation")
    if comparison != "not_applicable":
        frames.append("comparison_and_intertextuality")
    if oscar != "not_mentioned":
        frames.append("award_legitimacy")
    if "adaptation_or_franchise" in targets:
        frames.append("adaptation_or_franchise_expectations")
    return frames[:4]


def normalize_item(
    raw: dict[str, Any],
    row: dict[str, Any],
    expected_id: str,
    with_evidence: bool,
) -> dict[str, Any] | None:
    if str(raw.get("i", "")).strip().removesuffix(".0") != expected_id:
        return None

    r = str(raw.get("r", "Q"))
    v = str(raw.get("v", "Q"))
    pe = str(raw.get("pe", "Q"))
    cr = str(raw.get("cr", "X"))
    oscar_code = str(raw.get("os", "N"))
    if r not in REL or v not in VAL or pe not in PRIMARY:
        return None
    if cr not in CMP or oscar_code not in OSC:
        return None

    e_codes = valid_list(raw.get("e"), EMO, 2)
    p_codes = valid_list(raw.get("p"), TARGET, 4)
    c_codes = valid_list(raw.get("c"), TARGET, 4)
    rm_codes = valid_list(raw.get("rm"), RHET, 2)
    z = max(0, min(3, int(raw.get("z", 0))))
    review = bool(raw.get("rev"))

    text = str(row.get("comment_text", ""))
    award_re = re.compile(
        r"\b(оскар\w*|oscar\w*|наград\w*|номинац\w*|преми\w*|статуэт\w*)",
        re.I,
    )
    ideology_re = re.compile(
        r"\b(повест\w*|идеолог\w*|политическ\w*|пропаганд\w*|"
        r"репрезентац\w*|лгбт\w*|woke\w*|воук\w*|фемини\w*|"
        r"гендер\w*|расизм\w*|сексизм\w*)",
        re.I,
    )
    admiration_re = re.compile(
        r"\b(восхит\w*|восторг\w*|великолеп\w*|шедевр\w*|"
        r"гениаль\w*|потряса\w*|превосход\w*|обожа\w*|лучший\w*)",
        re.I,
    )

    # Hard semantic gates.
    if not award_re.search(text):
        oscar_code = "N"
        p_codes = [x for x in p_codes if x != "AW"]
        c_codes = [x for x in c_codes if x != "AW"]
    if not ideology_re.search(text):
        p_codes = [x for x in p_codes if x != "IR"]
        c_codes = [x for x in c_codes if x != "IR"]
    if "AD" in e_codes and not admiration_re.search(text):
        e_codes = ["EN" if x == "AD" else x for x in e_codes]
    if pe == "AD" and not admiration_re.search(text):
        pe = "EN"

    contains = r == "F"
    if not contains:
        v, e_codes, pe = "X", [], "X"
        p_codes, c_codes, rm_codes = [], [], []
        cr, oscar_code, z = "X", "N", 0
    else:
        if not rm_codes:
            rm_codes = ["LI"]
        if pe in EMO and pe not in e_codes:
            e_codes = [pe, *e_codes][:2]
        if not e_codes and pe in EMO:
            pe = "NE"

        if v == "0" and p_codes and not c_codes:
            v = "P"
        elif v == "0" and c_codes and not p_codes:
            v = "N"
        elif v == "M" and p_codes and not c_codes:
            v = "P"
        elif v == "M" and c_codes and not p_codes:
            v = "N"

        if v == "P" and not p_codes:
            p_codes = ["FO"]
        if v == "N" and not c_codes:
            c_codes = ["FO"]

    emotions = [EMO_MAP[x] for x in e_codes]
    praise = [TARGET_MAP[x] for x in p_codes]
    criticism = [TARGET_MAP[x] for x in c_codes]
    primary = EMO_MAP[pe]
    comparison = CMP_MAP[cr]
    oscar = OSC_MAP[oscar_code]

    evidence = ""
    if with_evidence:
        candidate = str(raw.get("q", "")).strip()
        if candidate and candidate in text:
            evidence = candidate
        elif candidate:
            review = True

    return {
        "comment_id": expected_id,
        "contains_film_reaction": contains,
        "relevance_primary": REL_MAP[r],
        "valence": VAL_MAP[v],
        "emotions": emotions,
        "primary_emotion": primary,
        "praise_targets": praise,
        "criticism_targets": criticism,
        "frames": derive_frames(
            emotions, praise, criticism, comparison, oscar
        ),
        "rhetorical_modes": [RHET_MAP[x] for x in rm_codes],
        "comparison_present": comparison != "not_applicable",
        "comparison_result": comparison,
        "oscar_stance": oscar,
        "intensity": z,
        "mentions_platform_issue": r == "T",
        "interpersonal_component": r == "U",
        "pre_release_or_trailer": r == "E",
        "needs_human_review": review,
        "evidence_fragment": evidence,
    }


def make_payload(
    rows: list[dict[str, Any]],
    ids: list[str],
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    objects = []
    for row, comment_id in zip(rows, ids):
        text = str(row.get("comment_text", "")).strip()[: args.max_chars]
        objects.append(
            {
                "i": comment_id,
                "f": str(row.get("film_ru") or row.get("film_en") or ""),
                "rp": str(row.get("is_reply", "0")) in {"1", "true", "True"},
                "t": text,
            }
        )
    evidence_instruction = (
        "\nq — точная короткая цитата из t, максимум 100 символов."
        if args.with_evidence
        else ""
    )
    user = (
        "Закодируй все объекты. На каждый i верни ровно один элемент массива x."
        + evidence_instruction
        + "\n"
        + json.dumps(objects, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": schema(args.with_evidence),
        "options": {
            "temperature": 0,
            "seed": 20260626,
            "num_ctx": args.num_ctx,
            "num_predict": 110 * len(rows) + 80,
        },
        "keep_alive": -1,
    }


def call_model(
    rows: list[dict[str, Any]],
    ids: list[str],
    prompt: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    response = request_json(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        make_payload(rows, ids, prompt, args),
        args.timeout,
    )
    content = str((response.get("message") or {}).get("content", ""))
    parsed = json.loads(clean_json(content))
    raw_items = parsed.get("x", [])
    if not isinstance(raw_items, list):
        raw_items = []

    row_map = {comment_id: row for comment_id, row in zip(ids, rows)}
    valid: dict[str, dict[str, Any]] = {}

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        comment_id = str(raw.get("i", "")).strip().removesuffix(".0")
        if comment_id not in row_map or comment_id in valid:
            continue
        normalized = normalize_item(
            raw,
            row_map[comment_id],
            comment_id,
            args.with_evidence,
        )
        if normalized is not None:
            valid[comment_id] = normalized

    missing = [comment_id for comment_id in ids if comment_id not in valid]
    return valid, missing, response


def paths(job_name: str) -> dict[str, Path]:
    data = PROJECT_ROOT / "data"
    art = PROJECT_ROOT / "artifacts" / "ollama_jobs" / job_name
    return {
        "checkpoint": data / f"{job_name}_ollama_checkpoint.jsonl",
        "csv": data / f"{job_name}_labeled_ollama.csv",
        "jsonl": data / f"{job_name}_labeled_ollama.jsonl",
        "errors": art / "errors.jsonl",
        "manifest": art / "manifest.json",
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA_VERSION:
                raise SystemExit(
                    "Checkpoint belongs to another schema. Use a new job name."
                )
            result[str(row["comment_id"])] = row
    return result


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_outputs(
    source: list[dict[str, Any]],
    ids: list[str],
    labels: dict[str, dict[str, Any]],
    out_paths: dict[str, Path],
    model: str,
) -> None:
    merged = []
    for row, comment_id in zip(source, ids):
        label = labels.get(comment_id)
        if not label:
            continue
        item = dict(row)
        item["comment_id"] = comment_id
        for key in OUTPUT_FIELDS:
            item[f"llm_{key}"] = label[key]
        item["llm_provider"] = "ollama"
        item["llm_model"] = model
        item["llm_schema_version"] = SCHEMA_VERSION
        item["llm_coding_status"] = "ok"
        merged.append(item)

    out_paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    with out_paths["jsonl"].open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if merged:
        fields = []
        seen = set()
        for row in merged:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        with out_paths["csv"].open(
            "w", encoding="utf-8-sig", newline=""
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in merged:
                writer.writerow(
                    {k: csv_value(v) for k, v in row.items()}
                )


def build_initial_tasks(
    pending: list[int],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> deque[Task]:
    tasks: deque[Task] = deque()
    current: list[int] = []
    chars = 0
    for index in pending:
        text_chars = min(
            len(str(rows[index].get("comment_text", ""))),
            args.max_chars,
        )
        if current and (
            len(current) >= args.batch_size
            or chars + text_chars > args.batch_char_budget
        ):
            tasks.append(Task(current))
            current = []
            chars = 0
        current.append(index)
        chars += text_chars
    if current:
        tasks.append(Task(current))
    return tasks


def eta(metrics: Metrics, total: int) -> str:
    elapsed = max(time.time() - metrics.started_at, 0.001)
    rate = metrics.completed / elapsed
    if rate <= 0:
        return "ETA ?"
    seconds = (total - metrics.completed) / rate
    if seconds < 3600:
        return f"ETA {seconds / 60:.1f}m"
    return f"ETA {seconds / 3600:.2f}h"


def main() -> None:
    args = parse_args()
    verify_model(args.ollama_url, args.model)
    prompt = (
        PROJECT_ROOT / "prompts" / "comment_coder_v4_compact.txt"
    ).read_text(encoding="utf-8")

    source = read_records(Path(args.input).resolve())
    ids = [stable_id(row, i) for i, row in enumerate(source)]
    if len(ids) != len(set(ids)):
        raise SystemExit("comment_id values are not unique.")

    out = paths(args.job_name)
    if args.restart:
        for path in out.values():
            path.unlink(missing_ok=True)

    labels = load_checkpoint(out["checkpoint"])
    pending = [
        i
        for i, comment_id in enumerate(ids)
        if comment_id not in labels
        and str(source[i].get("comment_text", "")).strip()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    total_to_have = len(labels) + len(pending)
    tasks = build_initial_tasks(pending, source, args)
    metrics = Metrics(started_at=time.time())

    print(f"Model: {args.model}")
    print(f"Schema: {SCHEMA_VERSION}")
    print(f"Already saved: {len(labels)}")
    print(f"Selected now: {len(pending)}")
    print(f"Initial batches: {len(tasks)}")
    print(f"Batch size up to: {args.batch_size}")
    print(f"Context: {args.num_ctx}")
    print(f"Evidence quotes: {args.with_evidence}")
    print()

    interrupted = False
    try:
        while tasks:
            task = tasks.popleft()
            indexes = [
                i for i in task.indexes if ids[i] not in labels
            ]
            if not indexes:
                continue

            batch_rows = [source[i] for i in indexes]
            batch_ids = [ids[i] for i in indexes]
            metrics.requests += 1

            try:
                valid, missing, response = call_model(
                    batch_rows, batch_ids, prompt, args
                )
                metrics.prompt_tokens += int(
                    response.get("prompt_eval_count", 0)
                )
                metrics.output_tokens += int(
                    response.get("eval_count", 0)
                )

                if valid:
                    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    checkpoint_rows = []
                    for comment_id, label in valid.items():
                        record = {
                            **label,
                            "schema_version": SCHEMA_VERSION,
                            "llm_model": args.model,
                            "llm_coded_at": timestamp,
                        }
                        labels[comment_id] = record
                        checkpoint_rows.append(record)
                    append_jsonl(out["checkpoint"], checkpoint_rows)
                    metrics.completed += len(valid)

                if missing:
                    missing_indexes = [
                        i for i in indexes if ids[i] in set(missing)
                    ]
                    if len(missing_indexes) == 1:
                        if task.tries < args.single_retries:
                            tasks.appendleft(
                                Task(missing_indexes, task.tries + 1)
                            )
                        else:
                            append_jsonl(
                                out["errors"],
                                [{
                                    "comment_id": ids[missing_indexes[0]],
                                    "error": "model_omitted_id",
                                    "tries": task.tries + 1,
                                }],
                            )
                    else:
                        middle = max(1, len(missing_indexes) // 2)
                        tasks.appendleft(Task(missing_indexes[middle:]))
                        tasks.appendleft(Task(missing_indexes[:middle]))

                elapsed = max(time.time() - metrics.started_at, 0.001)
                rate = metrics.completed / elapsed
                print(
                    f"saved={len(labels)}/{total_to_have}; "
                    f"last={len(valid)}/{len(indexes)}; "
                    f"{rate:.2f} comments/s; {eta(metrics, len(pending))}"
                )

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                print(
                    f"Request with {len(indexes)} rows failed: {exc}",
                    file=sys.stderr,
                )
                if len(indexes) > 1:
                    middle = len(indexes) // 2
                    tasks.appendleft(Task(indexes[middle:]))
                    tasks.appendleft(Task(indexes[:middle]))
                elif task.tries < args.single_retries:
                    tasks.appendleft(Task(indexes, task.tries + 1))
                else:
                    append_jsonl(
                        out["errors"],
                        [{
                            "comment_id": ids[indexes[0]],
                            "error": f"{type(exc).__name__}: {exc}",
                            "tries": task.tries + 1,
                        }],
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Finalizing saved rows...", file=sys.stderr)
    finally:
        write_outputs(source, ids, labels, out, args.model)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "input": str(Path(args.input).resolve()),
            "saved_total": len(labels),
            "selected_this_run": len(pending),
            "requests_this_run": metrics.requests,
            "prompt_tokens": metrics.prompt_tokens,
            "output_tokens": metrics.output_tokens,
            "elapsed_seconds": time.time() - metrics.started_at,
            "interrupted": interrupted,
            "output_csv": str(out["csv"]),
            "output_jsonl": str(out["jsonl"]),
        }
        out["manifest"].parent.mkdir(parents=True, exist_ok=True)
        out["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\nSaved output:")
    print(out["csv"])
    print(f"Checkpoint rows: {len(labels)}")
    if interrupted:
        print("Run the same command again to resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
