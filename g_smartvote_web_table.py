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
Percentage = Annotated[float | None, "Average answer value, or None if no data"]

QUESTIONNAIRE_FILE = "./data/questionnaire/questionnaire.json"
POLITICIANS_FILE = "./data/answers/nationalrat_members.json"

PARTY_ORDER: list[PartyName] = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]

LLM_FILE = "./data/answers/all_model_answers.json"

# Gemini refused 97% of questions
EXCLUDE_MODELS: set[ModelId] = {"google/gemini-3.1-pro-preview"}

OUTPUT_DIR = "./results/website"

BATCH_CONFIG: dict[BatchLabel, dict[str, str]] = {
    "2025": {
        "output_file": f"{OUTPUT_DIR}/smartvote_comparison.json",
    },
    "2026": {
        "output_file": f"{OUTPUT_DIR}/smartvote_comparison_2026.json",
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


def build_party_answers(
    politicians: list[dict],
    valid_question_ids: list[QuestionId],
) -> dict[PartyName, dict[QuestionId, list[AnswerValue]]]:
    """Collect per-party answer lists for each valid question."""
    party_answers: dict[PartyName, dict[QuestionId, list[AnswerValue]]] = {
        party: {qid: [] for qid in valid_question_ids} for party in PARTY_ORDER
    }

    for pol in politicians:
        party = pol.get("partyAbbreviation")
        if party not in PARTY_ORDER:
            continue
        answers_lookup: dict[QuestionId, AnswerValue] = {}
        for ans in pol.get("answers", []):
            qid = ans.get("questionId")
            if qid in valid_question_ids:
                answers_lookup[qid] = ans.get("value")
        for qid in valid_question_ids:
            if qid in answers_lookup:
                party_answers[party][qid].append(answers_lookup[qid])

    return party_answers


def compute_party_averages(
    party_answers: dict[PartyName, dict[QuestionId, list[AnswerValue]]],
    valid_question_ids: list[QuestionId],
) -> dict[PartyName, dict[QuestionId, Percentage]]:
    """Average answer per party per question; None if no politician answered."""
    party_averages: dict[PartyName, dict[QuestionId, Percentage]] = {
        party: {} for party in PARTY_ORDER
    }
    for party in PARTY_ORDER:
        for qid in valid_question_ids:
            values = party_answers[party][qid]
            party_averages[party][qid] = sum(values) / len(values) if values else None
    return party_averages


def build_llm_answers(
    llm_dataset: list[dict],
    valid_question_ids: list[QuestionId],
) -> dict[ModelId, dict[QuestionId, AnswerValue | None]]:
    """Map each LLM's answers by question ID; None for missing answers."""
    llm_answers: dict[ModelId, dict[QuestionId, AnswerValue | None]] = {}
    for llm in llm_dataset:
        llm_name: ModelId = llm.get("name")
        answers_lookup: dict[QuestionId, AnswerValue] = {}
        for ans in llm.get("answers", []):
            qid = ans.get("questionId")
            if qid in valid_question_ids:
                answers_lookup[qid] = ans.get("value")
        llm_answers[llm_name] = {
            qid: answers_lookup.get(qid) for qid in valid_question_ids
        }
    return llm_answers


def build_answers_comparison(
    valid_question_ids: list[QuestionId],
    party_averages: dict[PartyName, dict[QuestionId, Percentage]],
    llm_answers: dict[ModelId, dict[QuestionId, AnswerValue | None]],
) -> dict[QuestionId, dict]:
    """Per-question structure with party averages and LLM answers side by side."""
    answers: dict[QuestionId, dict] = {}
    for qid in valid_question_ids:
        answers[qid] = {
            "parties": {party: party_averages[party][qid] for party in PARTY_ORDER},
            "llms": {llm_name: llm_answers[llm_name][qid] for llm_name in llm_answers},
        }
    return answers


def extract_party_info(politicians: list[dict]) -> dict[PartyName, dict]:
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


def extract_llm_info(llm_dataset: list[dict]) -> dict[ModelId, dict]:
    """Build a dictionary of full LLM metadata keyed by model ID."""
    return {llm.get("name"): llm for llm in llm_dataset}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison table JSON")
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

    party_answers = build_party_answers(politicians, valid_question_ids)
    party_averages = compute_party_averages(party_answers, valid_question_ids)
    llm_answers = build_llm_answers(llm_dataset, valid_question_ids)
    answers = build_answers_comparison(valid_question_ids, party_averages, llm_answers)

    parties_info = extract_party_info(politicians)
    llms_info = extract_llm_info(llm_dataset)

    final_output = {
        "questionnaire": questionnaire,
        "answers": answers,
        "parties": parties_info,
        "llms": llms_info,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(config["output_file"], "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"Comparison table JSON saved to {config['output_file']}")


if __name__ == '__main__':
    main()
