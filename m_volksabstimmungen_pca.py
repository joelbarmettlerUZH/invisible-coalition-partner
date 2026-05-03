#!/usr/bin/env python3
"""
PCA on party Parolen from Volksabstimmungen data.

Mirrors the Smartvote PCA approach: each vote is a dimension (like a question),
each party is an observation (like a politician). PCA is fit on the 6 main parties'
Parolen vectors, then LLM vote vectors are projected into the same space.

Matrix: (6 parties x N votes), where Ja=100, Nein=0, Stimmfreigabe/missing=50.

Usage:
    uv run python m_volksabstimmungen_pca.py
"""

import json
import os
from typing import Annotated, Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from _volksabstimmungen_constants import (
    EXCLUDED_TITLES,
    MAIN_PARTIES,
    PARTY_COLORS,
    PARTY_NAME_MAP,
    PAROLE_VALUES,
)

VorlagenId = Annotated[int, "Unique referendum identifier"]
PartyName = Annotated[str, "One of SP, Grüne, GLP, Die Mitte, FDP, SVP"]
ParoleValue = Annotated[float, "Encoded parole: 100=Ja, 0=Nein, 50=neutral/missing"]
VoteDict = Annotated[dict[str, Any], "Single vote entry from volksabstimmungen.json"]
ModelDict = Annotated[dict[str, Any], "Model entry from volksabstimmungen_model_answers.json"]

VOLKSABSTIMMUNGEN_FILE = "./data/volksabstimmungen/volksabstimmungen.json"
LLM_ANSWERS_FILE = "./data/answers/volksabstimmungen_model_answers.json"
OUTPUT_DIR = "./results/volksabstimmungen_pca"

DEFAULT_PAROLE: ParoleValue = 50

MIN_VALID_ANSWERS = 10


def load_json(filename: str) -> Any:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def get_usable_votes(data: list[VoteDict]) -> list[VoteDict]:
    """Filter out Stichfrage, Direkter Gegenentwurf, and votes without party Parolen."""
    return [v for v in data if v["titel"] not in EXCLUDED_TITLES and v["parolen"]["parties"]]


def normalize_party_name(name: str) -> PartyName:
    return PARTY_NAME_MAP.get(name, name)


def build_party_answer_matrix(votes: list[VoteDict]) -> tuple[np.ndarray, list[VorlagenId]]:
    """
    Build a (6 parties x N votes) matrix. Each party is an observation,
    each vote is a dimension. Mirrors the Smartvote approach where each
    politician is an observation and each question is a dimension.
    """
    vote_ids: list[VorlagenId] = [v["vorlagenId"] for v in votes]

    matrix = np.full((len(MAIN_PARTIES), len(votes)), DEFAULT_PAROLE, dtype=float)

    for j, vote in enumerate(votes):
        parolen_by_party: dict[PartyName, ParoleValue] = {}
        for p in vote["parolen"]["parties"]:
            normalized = normalize_party_name(p["name"])
            if normalized in MAIN_PARTIES and p["parole"] in PAROLE_VALUES:
                parolen_by_party[normalized] = PAROLE_VALUES[p["parole"]]

        for i, party in enumerate(MAIN_PARTIES):
            if party in parolen_by_party:
                matrix[i, j] = parolen_by_party[party]

    return matrix, vote_ids


def run_pca_and_export(
    X: np.ndarray,
    vote_ids: list[VorlagenId],
    n_components: int,
    output_filename: str,
    votes: list[VoteDict],
) -> tuple[PCA, np.ndarray]:
    """
    Fit PCA on the party Parolen matrix and export results.

    X shape: (6 parties x N votes). Each of the 6 parties is projected into
    n_components-dimensional space based on their voting pattern.
    """
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)

    result: dict[str, Any] = {
        "metadata": {
            "vote_ids": vote_ids,
            "vote_labels": [
                f"{v['abstimmtag']}: {v['titel'][:60]}"
                for v in votes
            ],
            "parties": MAIN_PARTIES,
            "pca": {
                "n_components": n_components,
                "mean": pca.mean_.tolist(),
                "components": pca.components_.tolist(),
                "explained_variance": pca.explained_variance_.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
        },
        "party_points": [],
        "model_points": [],
    }

    for i, party in enumerate(MAIN_PARTIES):
        result["party_points"].append({
            "name": party,
            "color": PARTY_COLORS.get(party, "#000000"),
            "coords": X_transformed[i].tolist(),
        })

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Exported {n_components}D PCA results to {output_filename}")

    return pca, X_transformed


def project_model(model_vote_vector: np.ndarray, pca: PCA, vote_ids: list[VorlagenId]) -> np.ndarray:
    """Project a model's vote vector into PCA space."""
    mean = np.array(pca.mean_)
    components = np.array(pca.components_)
    return np.dot(model_vote_vector - mean, components.T)


def validate_left_right(pca: PCA, X_transformed: np.ndarray) -> bool:
    """Check if PC1 captures the left-right spectrum."""
    print("\n=== PCA Validation ===")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    if len(pca.explained_variance_ratio_) > 1:
        cumulative = sum(pca.explained_variance_ratio_[:2])
        print(f"PC1+PC2 explain {cumulative*100:.1f}% of variance")

    print("\nParty positions on PC1:")
    pc1_values = X_transformed[:, 0]
    for i, party in enumerate(MAIN_PARTIES):
        print(f"  {party:12s}: {pc1_values[i]:+.2f}")

    expected_rank = list(range(1, 7))
    rho, p_value = spearmanr(expected_rank, pc1_values)

    # PCA sign is arbitrary; check both directions
    rho_neg, p_neg = spearmanr(expected_rank, -pc1_values)
    if abs(rho_neg) > abs(rho):
        rho, p_value = rho_neg, p_neg
        print("\n(PC1 negated for left-right interpretation)")

    print(f"\nSpearman rho with expected left-right ordering: {rho:.3f} (p={p_value:.4f})")

    if abs(rho) > 0.8 and p_value < 0.05:
        print("PC1 captures the left-right spectrum well.")
        return True
    else:
        # With only 6 observations, PCA serves primarily as visualization;
        # agreement-based analyses are the primary statistical tool.
        print("PC1 does NOT clearly capture the left-right spectrum.")
        return False


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(VOLKSABSTIMMUNGEN_FILE)
    votes = get_usable_votes(data)
    print(f"Loaded {len(votes)} usable votes (excluded Stichfrage + Gegenentwurf)")

    X, vote_ids = build_party_answer_matrix(votes)
    print(f"Party answer matrix shape: {X.shape} (parties x votes)")

    print("\nParole distribution per party:")
    for i, party in enumerate(MAIN_PARTIES):
        row = X[i]
        n_ja = (row == 100).sum()
        n_nein = (row == 0).sum()
        n_neutral = (row == 50).sum()
        print(f"  {party:12s}: Ja={n_ja:2d}  Nein={n_nein:2d}  Neutral/Missing={n_neutral:2d}")

    pca_2d, X_2d = run_pca_and_export(
        X, vote_ids, n_components=2,
        output_filename=f"{OUTPUT_DIR}/pca_2d.json",
        votes=votes,
    )

    validate_left_right(pca_2d, X_2d)

    run_pca_and_export(
        X, vote_ids, n_components=1,
        output_filename=f"{OUTPUT_DIR}/pca_1d.json",
        votes=votes,
    )

    try:
        llm_data: list[ModelDict] = load_json(LLM_ANSWERS_FILE)
        print(f"\nProjecting {len(llm_data)} LLM models into Volksabstimmungen PCA space...")

        for result_file, pca_model, ndim in [
            (f"{OUTPUT_DIR}/pca_2d.json", pca_2d, 2),
            (f"{OUTPUT_DIR}/pca_1d.json", pca_2d, 1),
        ]:
            result = load_json(result_file)
            result["model_points"] = []

            for model in llm_data:
                cond = model.get("conditions", {}).get("in_kuerze", {})
                lang_data = cond.get("de", {})
                answers: dict[VorlagenId, int] = {
                    a["vorlagenId"]: a["value"] for a in lang_data.get("answers", [])
                }

                valid_count = sum(1 for vid in vote_ids if answers.get(vid, -1) not in (-1, None))
                if valid_count < MIN_VALID_ANSWERS:
                    print(f"  SKIP {model['display']}: only {valid_count} valid answers (need {MIN_VALID_ANSWERS})")
                    continue

                vote_vector = np.array([
                    answers.get(vid, DEFAULT_PAROLE) if answers.get(vid, -1) != -1 else DEFAULT_PAROLE
                    for vid in vote_ids
                ], dtype=float)

                coords = project_model(vote_vector, pca_model, vote_ids)
                result["model_points"].append({
                    "name": model["name"],
                    "display": model["display"],
                    "coords": coords[:ndim].tolist(),
                    "valid_answers": valid_count,
                })

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  Updated {result_file} with {len(result['model_points'])} model points")

    except FileNotFoundError:
        print("\nNo LLM Volksabstimmungen data yet; run l_generate_volksabstimmungen_dataset.py first.")


if __name__ == "__main__":
    main()
