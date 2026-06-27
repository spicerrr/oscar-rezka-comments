#!/usr/bin/env python3
"""Analyze final v4 Ollama labels and produce descriptive outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARRAY_COLUMNS = [
    "llm_emotions",
    "llm_praise_targets",
    "llm_criticism_targets",
    "llm_frames",
    "llm_rhetorical_modes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(
            PROJECT_ROOT / "data" / "sample_2555_v4_labeled_ollama.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "analysis_v4"),
    )
    parser.add_argument("--min-film-n", type=int, default=30)
    return parser.parse_args()


def parse_array(value) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def has_value(series: pd.Series, value: str) -> pd.Series:
    return series.map(lambda items: value in items)


def multilabel_summary(
    frame: pd.DataFrame,
    column: str,
    output: Path,
) -> pd.DataFrame:
    exploded = frame[["comment_id", column]].explode(column)
    exploded = exploded[exploded[column].notna() & exploded[column].ne("")]
    summary = (
        exploded[column]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="n")
    )
    summary["share_of_relevant_comments"] = summary["n"] / len(frame)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    return summary


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return math.sqrt(chi2 / denominator) if denominator > 0 else float("nan")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    for column in ARRAY_COLUMNS:
        frame[column] = frame[column].map(parse_array)

    reaction = frame[
        frame["llm_contains_film_reaction"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()
    if reaction.empty:
        raise SystemExit("No rows coded as film reactions.")

    summary = {
        "coded_rows": int(len(frame)),
        "film_reaction_rows": int(len(reaction)),
        "film_reaction_share": float(len(reaction) / len(frame)),
        "films": int(reaction["film_id"].nunique()),
    }
    (output / "corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for column in [
        "llm_valence",
        "llm_primary_emotion",
        "llm_comparison_result",
        "llm_oscar_stance",
    ]:
        table = (
            reaction[column]
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="n")
        )
        table["share"] = table["n"] / table["n"].sum()
        table.to_csv(
            output / f"overall_{column.removeprefix('llm_')}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    for column in ARRAY_COLUMNS:
        multilabel_summary(
            reaction,
            column,
            output / f"overall_{column.removeprefix('llm_')}.csv",
        )

    categories = pd.DataFrame(
        {
            "comment_id": reaction["comment_id"],
            "film_id": reaction["film_id"],
            "film_ru": reaction["film_ru"],
            "praise_film": has_value(
                reaction["llm_praise_targets"], "film_overall"
            ),
            "praise_acting": has_value(
                reaction["llm_praise_targets"], "acting"
            ),
            "criticism_plot": has_value(
                reaction["llm_criticism_targets"], "plot"
            ),
            "criticism_ideology": has_value(
                reaction["llm_criticism_targets"],
                "ideology_or_representation",
            ),
            "boredom": has_value(reaction["llm_emotions"], "boredom"),
            "disappointment": has_value(
                reaction["llm_emotions"], "disappointment"
            ),
            "admiration": has_value(
                reaction["llm_emotions"], "admiration"
            ),
            "disgust": has_value(reaction["llm_emotions"], "disgust"),
            "irony_or_mockery": reaction["llm_rhetorical_modes"].map(
                lambda values: bool(
                    {"irony_or_sarcasm", "mockery"} & set(values)
                )
            ),
            "comparison_present": reaction["llm_comparison_present"]
            .astype(str)
            .str.lower()
            .eq("true"),
            "oscar_discussion": reaction["llm_oscar_stance"].ne(
                "not_mentioned"
            ),
            "oscar_deserved": reaction["llm_oscar_stance"].eq(
                "deserved"
            ),
            "oscar_undeserved": reaction["llm_oscar_stance"].eq(
                "undeserved"
            ),
        }
    )

    category_columns = [
        column
        for column in categories.columns
        if column
        not in {"comment_id", "film_id", "film_ru"}
    ]

    category_summary = pd.DataFrame(
        {
            "category": category_columns,
            "n": [int(categories[column].sum()) for column in category_columns],
            "share": [
                float(categories[column].mean())
                for column in category_columns
            ],
        }
    ).sort_values("share", ascending=False)
    category_summary.to_csv(
        output / "requested_categories_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_film = (
        categories.groupby(["film_id", "film_ru"])[category_columns]
        .mean()
        .reset_index()
    )
    counts = (
        categories.groupby(["film_id", "film_ru"])
        .size()
        .rename("n_reactions")
        .reset_index()
    )
    by_film = counts.merge(by_film, on=["film_id", "film_ru"])
    by_film.to_csv(
        output / "requested_categories_by_film_shares.csv",
        index=False,
        encoding="utf-8-sig",
    )

    valence_by_film = pd.crosstab(
        [reaction["film_id"], reaction["film_ru"]],
        reaction["llm_valence"],
    )
    valence_by_film.to_csv(
        output / "valence_by_film_counts.csv",
        encoding="utf-8-sig",
    )
    valence_by_film.div(
        valence_by_film.sum(axis=1), axis=0
    ).to_csv(
        output / "valence_by_film_shares.csv",
        encoding="utf-8-sig",
    )

    eligible_ids = (
        reaction.groupby("film_id")
        .size()
        .loc[lambda values: values >= args.min_film_n]
        .index
    )
    eligible = reaction[reaction["film_id"].isin(eligible_ids)]
    table = pd.crosstab(eligible["film_id"], eligible["llm_valence"])
    tests = {}
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p_value, dof, expected = chi2_contingency(table)
        tests["valence_by_film"] = {
            "n": int(table.to_numpy().sum()),
            "films": int(table.shape[0]),
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p_value),
            "cramers_v": float(cramers_v(table)),
            "minimum_expected_count": float(expected.min()),
            "share_expected_below_5": float((expected < 5).mean()),
        }
    (output / "chi_square_tests.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_data = category_summary.sort_values("share").tail(12)
    plt.figure(figsize=(9, 6))
    plt.barh(plot_data["category"], plot_data["share"])
    plt.xlabel("Доля релевантных комментариев")
    plt.ylabel("Категория")
    plt.tight_layout()
    plt.savefig(output / "requested_categories.png", dpi=180)
    plt.close()

    valence = (
        reaction["llm_valence"]
        .value_counts(normalize=True)
        .sort_values()
    )
    plt.figure(figsize=(8, 5))
    plt.barh(valence.index.astype(str), valence.values)
    plt.xlabel("Доля релевантных комментариев")
    plt.ylabel("Валентность")
    plt.tight_layout()
    plt.savefig(output / "overall_valence.png", dpi=180)
    plt.close()

    frames = multilabel_summary(
        reaction,
        "llm_frames",
        output / "overall_frames.csv",
    ).sort_values("share_of_relevant_comments")
    plt.figure(figsize=(9, 6))
    plt.barh(
        frames["category"],
        frames["share_of_relevant_comments"],
    )
    plt.xlabel("Доля релевантных комментариев")
    plt.ylabel("Фрейм")
    plt.tight_layout()
    plt.savefig(output / "overall_frames.png", dpi=180)
    plt.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
