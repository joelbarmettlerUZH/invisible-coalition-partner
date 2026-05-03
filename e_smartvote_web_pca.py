#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Annotated

import numpy as np
from sklearn.decomposition import PCA

QuestionId = Annotated[str, "Smartvote question ID"]
PartyName = Annotated[str, "One of SP, Grüne, GLP, Die Mitte, FDP, SVP"]
AnswerValue = Annotated[float, "Answer on 0-100 scale"]
PcaCoord = Annotated[list[float], "PCA projection coordinates"]

QUESTIONNAIRE_FILE = "./data/questionnaire/questionnaire.json"
POLITICIANS_FILE = "./data/answers/nationalrat_members.json"

OUTPUT_DIR = "./results/website"

PARTY_ORDER: list[PartyName] = [
    "SP",
    "Grüne",
    "GLP",
    "Die Mitte",
    "FDP",
    "SVP",
]

# Neutral midpoint used when a politician did not answer a question
DEFAULT_ANSWER: AnswerValue = 50


def load_json(filename: str) -> dict | list:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_valid_question_ids(questionnaire: dict) -> list[QuestionId]:
    """Extract question IDs, skipping BudgetCategory. Sorted by category then question sortorder."""
    valid_questions: list[tuple[int, int, QuestionId]] = []
    for category in questionnaire.get("categories", []):
        if category.get("type") == "BudgetCategory":
            continue
        cat_sortorder = category.get("sortorder", 0)
        for question in category.get("questions", []):
            ques_sortorder = question.get("sortorder", 0)
            valid_questions.append((cat_sortorder, ques_sortorder, question["id"]))
    valid_questions.sort(key=lambda tup: (tup[0], tup[1]))
    return [tup[2] for tup in valid_questions]


def build_politician_answer_matrix(
    politicians: list[dict],
    question_ids: list[QuestionId],
) -> tuple[np.ndarray, list[dict]]:
    """Build answer matrix and metadata for politicians in PARTY_ORDER parties."""
    data: list[list[AnswerValue]] = []
    meta: list[dict] = []

    for pol in politicians:
        party = pol.get("partyAbbreviation")
        if party not in PARTY_ORDER:
            continue

        answers_lookup: dict[QuestionId, AnswerValue] = {}
        for answer in pol.get("answers", []):
            qid = answer.get("questionId")
            if qid in question_ids:
                answers_lookup[qid] = answer.get("value", DEFAULT_ANSWER)

        answer_vector = [answers_lookup.get(qid, DEFAULT_ANSWER) for qid in question_ids]
        data.append(answer_vector)

        name = f"{pol.get('firstname', '')} {pol.get('lastname', '')}".strip()
        meta.append({
            "name": name,
            "party": party,
            "color": pol.get("partyColor", "#000000"),
        })

    return np.array(data, dtype=float), meta


def run_pca_and_export(
    X: np.ndarray,
    meta: list[dict],
    question_ids: list[QuestionId],
    n_components: int,
    output_filename: str,
) -> None:
    """Fit PCA and export model parameters + projected politician points to JSON."""
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)

    result = {
        "metadata": {
            "question_ids": question_ids,
            "pca": {
                "n_components": n_components,
                "mean": pca.mean_.tolist(),
                "components": pca.components_.tolist(),
                "explained_variance": pca.explained_variance_.tolist(),
            },
        },
        "points": [
            {
                "name": meta[i]["name"],
                "party": meta[i]["party"],
                "color": meta[i]["color"],
                "coords": X_transformed[i].tolist(),
            }
            for i in range(len(meta))
        ],
    }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Exported {n_components}D PCA results to {output_filename}")


def main() -> None:
    questionnaire = load_json(QUESTIONNAIRE_FILE)
    politicians = load_json(POLITICIANS_FILE)

    question_ids = extract_valid_question_ids(questionnaire)
    print(f"Found {len(question_ids)} valid questions.")

    X, meta = build_politician_answer_matrix(politicians, question_ids)
    print(f"Built answer matrix with shape: {X.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_pca_and_export(X, meta, question_ids, n_components=1,
                       output_filename=f"{OUTPUT_DIR}/smartvote_pca_1d.json")
    run_pca_and_export(X, meta, question_ids, n_components=2,
                       output_filename=f"{OUTPUT_DIR}/smartvote_pca_2d.json")


if __name__ == '__main__':
    main()
