#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Annotated

import numpy as np

QuestionId = Annotated[str, "Smartvote question ID"]
AnswerValue = Annotated[float, "Answer on 0-100 scale"]
PcaCoord = Annotated[list[float], "PCA projection coordinates"]
ModelId = Annotated[str, "OpenRouter model ID"]
BatchLabel = Annotated[str, "Experiment wave: '2025' or '2026'"]

OUTPUT_DIR = "./results/website"

PCA_1D_FILE = f"{OUTPUT_DIR}/smartvote_pca_1d.json"
PCA_2D_FILE = f"{OUTPUT_DIR}/smartvote_pca_2d.json"

LLM_FILE = "./data/answers/all_model_answers.json"

# Gemini refused 97% of questions
EXCLUDE_MODELS: set[ModelId] = {"google/gemini-3.1-pro-preview"}

BATCH_CONFIG: dict[BatchLabel, dict[str, str]] = {
    "2025": {
        "output_1d": f"{OUTPUT_DIR}/smartvote_combined_1d.json",
        "output_2d": f"{OUTPUT_DIR}/smartvote_combined_2d.json",
    },
    "2026": {
        "output_1d": f"{OUTPUT_DIR}/smartvote_combined_1d_2026.json",
        "output_2d": f"{OUTPUT_DIR}/smartvote_combined_2d_2026.json",
    },
}


def load_json(filename: str) -> dict | list:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def project_point(
    answer_vector: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return np.dot(answer_vector - mean, components.T)


def build_llm_answer_vector(
    llm: dict,
    question_ids: list[QuestionId],
    default_value: AnswerValue = 50,
) -> np.ndarray:
    """Build an answer vector aligned to question_ids order, using default_value for missing answers."""
    answers_lookup: dict[QuestionId, AnswerValue] = {}
    for ans in llm.get("answers", []):
        qid = ans.get("questionId")
        if qid:
            answers_lookup[qid] = ans.get("value", default_value)
    return np.array([answers_lookup.get(qid, default_value) for qid in question_ids], dtype=float)


def combine_with_llms(
    pca_json: dict,
    llm_dataset: list[dict],
    default_value: AnswerValue = 50,
) -> dict:
    """Project each LLM into the existing PCA space and append to politician points."""
    metadata = pca_json.get("metadata", {})
    pca_info = metadata.get("pca", {})
    mean = np.array(pca_info.get("mean"))
    components = np.array(pca_info.get("components"))
    question_ids: list[QuestionId] = metadata.get("question_ids", [])

    points = pca_json.get("points", [])

    for llm in llm_dataset:
        answer_vector = build_llm_answer_vector(llm, question_ids, default_value)
        projected = project_point(answer_vector, mean, components)
        llm_point = {
            "name": llm.get("name"),
            "display": llm.get("display"),
            "country": llm.get("country"),
            "continent": llm.get("continent"),
            "size": llm.get("size"),
            "reasoning": llm.get("reasoning"),
            "open_source": llm.get("open_source"),
            "batch": llm.get("batch"),
            "family": llm.get("family"),
            "predecessor": llm.get("predecessor"),
            "coords": projected.tolist(),
            "type": "LLM",
        }
        points.append(llm_point)

    pca_json["points"] = points
    return pca_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Project LLM answers into PCA space")
    parser.add_argument("--batch", choices=["2025", "2026"], default="2025",
                        help="Which batch of models to process (default: 2025)")
    args = parser.parse_args()

    config = BATCH_CONFIG[args.batch]

    all_models = load_json(LLM_FILE)
    llm_dataset = [m for m in all_models
                   if m.get("batch") == args.batch
                   and m.get("name") not in EXCLUDE_MODELS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pca_1d = load_json(PCA_1D_FILE)
    combined_1d = combine_with_llms(pca_1d, llm_dataset)
    with open(config["output_1d"], "w", encoding="utf-8") as f:
        json.dump(combined_1d, f, indent=2)
    print(f"Combined 1D results saved to {config['output_1d']}")

    pca_2d = load_json(PCA_2D_FILE)
    combined_2d = combine_with_llms(pca_2d, llm_dataset)
    with open(config["output_2d"], "w", encoding="utf-8") as f:
        json.dump(combined_2d, f, indent=2)
    print(f"Combined 2D results saved to {config['output_2d']}")


if __name__ == '__main__':
    main()
