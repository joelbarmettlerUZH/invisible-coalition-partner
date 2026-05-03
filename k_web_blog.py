"""
Generate extra JSON data files needed for the blog article.

Usage:
    cd projects/llm-political-bias
    uv run python k_web_blog.py

Outputs:
    results/website/timeline_data.json      - Release date vs PC1 for tracked companies
    results/website/refusal_data.json       - Refusal rates across all models
    results/website/agreement_heatmap.json  - Flagship agreement matrix (models x parties)
"""

import json
import os
from typing import Annotated, Any

import numpy as np

QuestionId = Annotated[str, "Smartvote question ID"]
PartyName = Annotated[str, "One of SP, Grüne, GLP, Die Mitte, FDP, SVP"]
Percentage = Annotated[float, "Value between 0 and 100"]
PC1Value = Annotated[float, "First principal component score (negated for left-right display)"]
ModelDict = Annotated[dict[str, Any], "Model entry from all_model_answers.json"]
FamilyDict = Annotated[dict[str, Any], "Family entry from model_families.json"]
PCAData = Annotated[dict[str, Any], "PCA results JSON structure"]

OUTPUT_DIR = "results/website"
DATA_DIR = "data"

PARTY_ORDER: list[PartyName] = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def project(model: ModelDict, pca_data: PCAData, ndim: int = 2) -> np.ndarray:
    """Project a model's answers into PCA space. Returns negated PC1."""
    pca_info = pca_data["metadata"]["pca"]
    mean = np.array(pca_info["mean"])
    components = np.array(pca_info["components"])
    question_ids: list[QuestionId] = pca_data["metadata"]["question_ids"]

    lookup: dict[QuestionId, int] = {}
    for ans in model.get("answers", []):
        qid, val = ans.get("questionId"), ans.get("value", 50)
        if qid and val != -1:
            lookup[qid] = val
    vec = np.array([lookup.get(qid, 50) for qid in question_ids], dtype=float)
    coords = np.dot(vec - mean, components.T).flatten()
    # Negate PC1 so left-wing parties appear on the left side
    coords[0] = -coords[0]
    return coords[:ndim]


def generate_timeline(pca_1d: PCAData, all_models: list[ModelDict]) -> dict[str, Any]:
    """Generate timeline data: release date vs PC1 for tracked companies."""
    FAMILY_TO_COMPANY: dict[str, str] = {
        "openai-gpt": "OpenAI",
        "openai-gpt-mini": "OpenAI",
        "openai-mini": "OpenAI",
        "openai-reasoning": "OpenAI",
        "xai-grok": "xAI",
        "mistral": "Mistral",
        "anthropic-sonnet": "Anthropic",
        "anthropic-opus": "Anthropic",
        "anthropic-haiku": "Anthropic",
    }
    COMPANY_COLORS: dict[str, str] = {
        "OpenAI": "#10a37f",
        "xAI": "#1da1f2",
        "Mistral": "#ff7000",
        "Anthropic": "#d4a574",
    }

    company_data: dict[str, list[dict[str, Any]]] = {}
    for model in all_models:
        released = model.get("released")
        family = model.get("family")
        if not released or not family:
            continue
        company = FAMILY_TO_COMPANY.get(family)
        if not company:
            continue
        if company not in company_data:
            company_data[company] = []

        pc1: PC1Value = float(project(model, pca_1d, ndim=1)[0])
        display = model.get("display", model["name"].split("/")[-1])
        size = model.get("size") or 0
        company_data[company].append({
            "name": display, "pc1": pc1, "date": released, "size": size,
        })

    for company in company_data:
        points = company_data[company]
        points.sort(key=lambda p: (p["date"], -p["size"]))
        deduped: list[dict[str, Any]] = []
        seen_dates: set[str] = set()
        for p in points:
            if p["date"] not in seen_dates:
                seen_dates.add(p["date"])
                deduped.append(p)
        company_data[company] = deduped

    pol_points = pca_1d.get("points", [])
    party_centroids: dict[PartyName, PC1Value] = {}
    for party in PARTY_ORDER:
        pts = [p for p in pol_points if p.get("party") == party]
        if pts:
            party_centroids[party] = float(-np.mean([p["coords"][0] for p in pts]))

    return {
        "companies": {
            company: {
                "color": COMPANY_COLORS.get(company, "#888"),
                "models": company_data[company]
            }
            for company in ["OpenAI", "xAI", "Mistral", "Anthropic"]
            if company in company_data
        },
        "party_centroids": party_centroids
    }


def generate_refusal(all_models: list[ModelDict]) -> list[dict[str, Any]]:
    """Generate refusal rate data for all models."""
    questionnaire = load_json(f"{DATA_DIR}/questionnaire/questionnaire.json")
    question_ids: set[QuestionId] = set()
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        for q in cat.get("questions", []):
            question_ids.add(q["id"])
    n_questions = len(question_ids)

    refusal_data: list[dict[str, Any]] = []
    for m in all_models:
        answers = m.get("answers", [])
        relevant = [a for a in answers if a.get("questionId") in question_ids]
        explicit_refusals = sum(1 for a in relevant if a.get("value") == -1)
        rate: Percentage = round(explicit_refusals / n_questions * 100, 1) if n_questions > 0 else 0
        refusal_data.append({
            "name": m["name"],
            "display": m.get("display", m["name"].split("/")[-1]),
            "rate": rate,
            "refusals": explicit_refusals,
            "total": n_questions,
            "country": m.get("country", ""),
        })

    refusal_data.sort(key=lambda x: x["rate"], reverse=True)
    return refusal_data


def generate_agreement_heatmap(
    all_models: list[ModelDict],
    families: dict[str, FamilyDict],
) -> dict[str, Any]:
    """Generate agreement heatmap data for flagship models x parties."""
    politicians = load_json(f"{DATA_DIR}/answers/nationalrat_members.json")
    questionnaire = load_json(f"{DATA_DIR}/questionnaire/questionnaire.json")

    question_ids: list[QuestionId] = []
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        for q in cat.get("questions", []):
            question_ids.append(q["id"])

    flagship_names: set[str] = set()
    for fam in families.values():
        f = fam.get("flagship")
        if f:
            flagship_names.add(f)
    flagships = [m for m in all_models if m["name"] in flagship_names]

    party_pols: dict[PartyName, list[dict[str, Any]]] = {p: [] for p in PARTY_ORDER}
    for pol in politicians:
        party = pol.get("party", {}).get("abbreviation", pol.get("partyAbbreviation"))
        if party in party_pols:
            party_pols[party].append(pol)

    def get_answer_map(entity: dict[str, Any]) -> dict[QuestionId, int]:
        m: dict[QuestionId, int] = {}
        for a in entity.get("answers", []):
            qid, val = a.get("questionId"), a.get("value", -1)
            if val != -1:
                m[qid] = val
        return m

    def agreement(llm_map: dict[QuestionId, int], pol_map: dict[QuestionId, int]) -> Percentage | None:
        overlap = set(llm_map.keys()) & set(pol_map.keys()) & set(question_ids)
        if not overlap:
            return None
        sq_diffs = [(llm_map[q] - pol_map[q]) ** 2 for q in overlap]
        return 100 * (1 - np.mean(sq_diffs) / 10000)

    rows: list[dict[str, Any]] = []
    for m in flagships:
        llm_map = get_answer_map(m)
        scores: dict[PartyName, Percentage] = {}
        for party in PARTY_ORDER:
            party_scores: list[Percentage] = []
            for pol in party_pols[party]:
                pol_map = get_answer_map(pol)
                a = agreement(llm_map, pol_map)
                if a is not None:
                    party_scores.append(a)
            scores[party] = round(np.mean(party_scores), 1) if party_scores else 0
        rows.append({
            "name": m["name"],
            "display": m.get("display", m["name"].split("/")[-1]),
            "country": m.get("country", ""),
            "open_source": m.get("open_source", False),
            "scores": scores
        })

    rows.sort(key=lambda r: r["scores"]["SP"], reverse=True)

    return {
        "parties": PARTY_ORDER,
        "models": rows
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating blog data files...")
    pca_1d = load_json(f"{OUTPUT_DIR}/smartvote_pca_1d.json")
    all_models = load_json(f"{DATA_DIR}/answers/all_model_answers.json")
    families = load_json(f"{DATA_DIR}/model_families.json")

    timeline = generate_timeline(pca_1d, all_models)
    save_json(f"{OUTPUT_DIR}/timeline_data.json", timeline)
    print(f"  -> timeline_data.json ({sum(len(c['models']) for c in timeline['companies'].values())} points)")

    refusal = generate_refusal(all_models)
    save_json(f"{OUTPUT_DIR}/refusal_data.json", refusal)
    with_refusal = [r for r in refusal if r["rate"] > 0]
    print(f"  -> refusal_data.json ({len(refusal)} models, {len(with_refusal)} with refusals)")

    heatmap = generate_agreement_heatmap(all_models, families)
    save_json(f"{OUTPUT_DIR}/agreement_heatmap.json", heatmap)
    print(f"  -> agreement_heatmap.json ({len(heatmap['models'])} flagship models)")

    print("Done.")


if __name__ == "__main__":
    main()
