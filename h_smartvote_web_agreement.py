#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Annotated

QuestionId = Annotated[str, "Smartvote question ID"]
PartyName = Annotated[str, "One of SP, Grüne, GLP, Die Mitte, FDP, SVP"]
AnswerValue = Annotated[float, "Answer on 0-100 scale"]
ModelId = Annotated[str, "OpenRouter model ID"]
BatchLabel = Annotated[str, "Experiment wave: '2025' or '2026'"]
Percentage = Annotated[float, "Value between 0 and 100"]

QUESTIONNAIRE_FILE = "./data/questionnaire/questionnaire.json"
POLITICIANS_FILE = "./data/answers/nationalrat_members.json"

PARTY_ORDER: list[PartyName] = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]

LLM_FILE = "./data/answers/all_model_answers.json"

# Gemini refused 97% of questions
EXCLUDE_MODELS: set[ModelId] = {"google/gemini-3.1-pro-preview"}

OUTPUT_DIR = "./results/website"

BATCH_CONFIG: dict[BatchLabel, dict[str, str]] = {
    "2025": {
        "output_file": f"{OUTPUT_DIR}/smartvote_agreement.json",
    },
    "2026": {
        "output_file": f"{OUTPUT_DIR}/smartvote_agreement_2026.json",
    },
}


def load_json(filename: str) -> dict | list:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_valid_question_ids(questionnaire: dict) -> list[QuestionId]:
    """Extract question IDs, skipping BudgetCategory."""
    valid_questions: list[QuestionId] = []
    for category in questionnaire.get("categories", []):
        if category.get("type") == "BudgetCategory":
            continue
        for question in category.get("questions", []):
            valid_questions.append(question["id"])
    return valid_questions


def build_party_politicians(
    politicians: list[dict],
    valid_question_ids: list[QuestionId],
) -> dict[PartyName, list[dict[QuestionId, AnswerValue]]]:
    """Build per-party lists of politician answer dicts (question ID -> value)."""
    party_pols: dict[PartyName, list[dict[QuestionId, AnswerValue]]] = {
        party: [] for party in PARTY_ORDER
    }
    for pol in politicians:
        party = pol.get("partyAbbreviation")
        if party not in PARTY_ORDER:
            continue
        answers: dict[QuestionId, AnswerValue] = {}
        for ans in pol.get("answers", []):
            qid = ans.get("questionId")
            if qid in valid_question_ids and ans.get("value") is not None:
                answers[qid] = ans.get("value")
        if answers:
            party_pols[party].append(answers)
    return party_pols


def build_llm_answers(
    llm_dataset: list[dict],
    valid_question_ids: list[QuestionId],
) -> dict[ModelId, dict[QuestionId, AnswerValue]]:
    """Build per-LLM answer dicts, skipping unanswered questions."""
    llm_answers: dict[ModelId, dict[QuestionId, AnswerValue]] = {}
    for llm in llm_dataset:
        llm_name: ModelId = llm.get("name")
        answers: dict[QuestionId, AnswerValue] = {}
        for ans in llm.get("answers", []):
            qid = ans.get("questionId")
            if qid in valid_question_ids and ans.get("value") is not None:
                answers[qid] = ans.get("value")
        llm_answers[llm_name] = answers
    return llm_answers


def compute_agreement(
    party_pols: dict[PartyName, list[dict[QuestionId, AnswerValue]]],
    llm_answers: dict[ModelId, dict[QuestionId, AnswerValue]],
    valid_question_ids: list[QuestionId],
) -> dict[PartyName, dict[ModelId, Percentage | None]]:
    """Compute squared-difference agreement between each party's politicians and each LLM.

    For every politician-LLM pair sharing at least one answered question, we compute
    the mean squared difference and normalize to [0, 100]:
        agreement = 100 * (1 - mean_sq_diff / 10000)

    The denominator 10,000 = 100^2 is the maximum possible squared difference when
    answers range from 0 to 100. Returns None when no overlapping answers exist.
    """
    max_squared_diff: float = 100 ** 2
    agreements: dict[PartyName, dict[ModelId, Percentage | None]] = {
        party: {} for party in PARTY_ORDER
    }

    for party in PARTY_ORDER:
        for llm_name, llm_ans in llm_answers.items():
            total = 0
            sum_sq_diff = 0.0
            for pol in party_pols[party]:
                for qid in valid_question_ids:
                    if qid in pol and qid in llm_ans:
                        diff = pol[qid] - llm_ans[qid]
                        sum_sq_diff += diff * diff
                        total += 1
            if total > 0:
                avg_sq_diff = sum_sq_diff / total
                agreement = 100 * (1 - (avg_sq_diff / max_squared_diff))
                agreement = max(0.0, min(100.0, agreement))
            else:
                agreement = None
            agreements[party][llm_name] = agreement

    return agreements


def extract_party_meta(politicians: list[dict]) -> dict[PartyName, dict]:
    """Extract party metadata from the first encountered politician per party."""
    parties: dict[PartyName, dict] = {}
    for pol in politicians:
        party = pol.get("partyAbbreviation")
        if party in PARTY_ORDER and party not in parties:
            party_obj = pol.get("party", {})
            parties[party] = {
                "name": party_obj.get("name", party),
                "color": pol.get("partyColor", "#000000"),
            }
    return parties


def extract_llm_meta(llm_dataset: list[dict]) -> dict[ModelId, dict]:
    """Build a dictionary of full LLM metadata keyed by model ID."""
    return {llm.get("name"): llm for llm in llm_dataset}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute agreement scores")
    parser.add_argument("--batch", choices=["2025", "2026"], default="2025",
                        help="Which batch of models to process (default: 2025)")
    args = parser.parse_args()

    config = BATCH_CONFIG[args.batch]

    questionnaire = load_json(QUESTIONNAIRE_FILE)
    politicians = load_json(POLITICIANS_FILE)
    all_models = load_json(LLM_FILE)
    llm_dataset = [m for m in all_models
                   if m.get("batch") == args.batch
                   and m.get("name") not in EXCLUDE_MODELS]

    valid_question_ids = extract_valid_question_ids(questionnaire)

    party_pols = build_party_politicians(politicians, valid_question_ids)
    llm_ans = build_llm_answers(llm_dataset, valid_question_ids)
    agreements = compute_agreement(party_pols, llm_ans, valid_question_ids)

    parties_meta = extract_party_meta(politicians)
    llms_meta = extract_llm_meta(llm_dataset)

    result = {
        "agreements": agreements,
        "parties": parties_meta,
        "llms": llms_meta,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(config["output_file"], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Agreement scores saved to {config['output_file']}")


if __name__ == '__main__':
    main()
