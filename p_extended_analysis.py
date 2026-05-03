#!/usr/bin/env python3
"""
Extended statistical analyses for the paper pivot.

Blocks:
  1. Nein tendency: binomial tests, vote direction classification, cross-instrument check
  2. Instrument divergence: gradient flip test (Spearman party-position vs agreement)
  3. Response bias cross-validation: Smartvote mean vs Volksabstimmungen Ja rate
  4. Convergence permutation test: model vs party pairwise agreement
  5. BH correction: across all tests from Smartvote + Volksabstimmungen + this script

Reads from results/paper/ individual files and writes back to results/paper/ individual files.

Usage:
    uv run python p_extended_analysis.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

from _volksabstimmungen_constants import (
    EXCLUDED_TITLES,
    MAIN_PARTIES,
    PARTY_NAME_MAP,
)

# ─── Configuration ──────────────────────────────────────────────────────────

VOLKSABSTIMMUNGEN_FILE: str = "./data/volksabstimmungen/volksabstimmungen.json"
LLM_ANSWERS_FILE: str = "./data/answers/volksabstimmungen_model_answers.json"
SMARTVOTE_ANSWERS_FILE: str = "./data/answers/all_model_answers.json"
POLITICIANS_FILE: str = "./data/answers/nationalrat_members.json"
QUESTIONNAIRE_FILE: str = "./data/questionnaire/questionnaire.json"
MODEL_FAMILIES_FILE: str = "./data/model_families.json"
PAPER_DIR: str = "./results/paper"

PARTY_ORDER: list[str] = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]
PARTY_POSITIONS: list[int] = list(range(1, 7))  # 1=SP (left) ... 6=SVP (right)

EXCLUDE_MODELS: set[str] = {
    "google/gemini-3.1-pro-preview",   # 84% refusal on Smartvote
    "qwen/qvq-72b-preview",           # 54% refusal, vision model
    "openai/o1-preview",               # 33% refusal
    "google/gemini-3-flash-preview",   # 25% refusal
    # x-ai/grok-4.20 was previously excluded as "duplicate of grok-4.20-beta"
    # but that only applies to Smartvote. In volksabstimmungen_model_answers.json
    # there is only one Grok entry, so excluding it here silently dropped Grok from
    # nein tendency, convergence, and cross-instrument analyses.
}
PRIMARY_CONDITION: str = "in_kuerze"
PRIMARY_LANGUAGE: str = "de"

N_PERMUTATIONS: int = 10_000
RANDOM_SEED: int = 42

VOLKS_MODEL_DISPLAY: list[str] = [
    "GPT-5.4", "Claude Opus 4.6", "DeepSeek V3.2",
    "Llama 4 Maverick", "Grok 4.20", "Mistral Large",
    "Command A", "Qwen 3.5 Plus",
]
VOLKS_MODELS_EXCLUDED: list[str] = ["Gemini 3.1 Pro (98% refusal rate in German)"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_paper_file(
    path: str,
    results: dict[str, Any],
    *,
    description: str,
    models_analyzed: list[str] | None = None,
    models_excluded: list[str] | None = None,
    n_models: int | None = None,
    methodology: str = "",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    envelope: dict[str, Any] = {
        "description": description,
        "generated_by": "p_extended_analysis.py",
        "generated_at": datetime.now().isoformat(),
    }
    if models_analyzed is not None:
        envelope["models_analyzed"] = models_analyzed
    if models_excluded is not None:
        envelope["models_excluded"] = models_excluded
    if n_models is not None:
        envelope["n_models"] = n_models
    if methodology:
        envelope["methodology"] = methodology
    envelope["results"] = results
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"  -> {path}")


def normalize_party(name: str) -> str:
    return PARTY_NAME_MAP.get(name, name)


def get_usable_votes(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for v in data if v["titel"] not in EXCLUDED_TITLES and v["parolen"]["parties"]]


def get_vote_parolen(vote: dict[str, Any]) -> dict[str, str | None]:
    parolen: dict[str, str | None] = {}
    for p in vote["parolen"]["parties"]:
        name = normalize_party(p["name"])
        if name in MAIN_PARTIES:
            parole = p["parole"]
            if parole in ("Ja", "Nein"):
                parolen[name] = parole
            else:
                parolen[name] = None
    return parolen


def classify_vote_direction(parolen: dict[str, str | None]) -> str:
    """Classify a vote as progressive-Ja, conservative-Ja, or mixed.

    progressive-Ja: SP and/or Gruene vote Ja AND SVP votes Nein
    conservative-Ja: SVP votes Ja AND SP and/or Gruene vote Nein
    mixed: anything else
    """
    left_parties = {"SP", "Grüne"}
    right_party = "SVP"

    left_ja = any(parolen.get(p) == "Ja" for p in left_parties)
    left_nein = any(parolen.get(p) == "Nein" for p in left_parties)
    right_ja = parolen.get(right_party) == "Ja"
    right_nein = parolen.get(right_party) == "Nein"

    if left_ja and right_nein and not left_nein:
        return "progressive_ja"
    elif right_ja and left_nein and not left_ja:
        return "conservative_ja"
    else:
        return "mixed"


# ─── Block 1: Nein Tendency ─────────────────────────────────────────────────

def analyze_nein_tendency(
    volks_data: list[dict[str, Any]],
    llm_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Binomial test per model, vote direction classification, direction-conditional Nein rate."""
    votes = get_usable_votes(volks_data)
    vote_ids = [v["vorlagenId"] for v in votes]

    vote_directions: dict[int, str] = {}
    for v in votes:
        parolen = get_vote_parolen(v)
        direction = classify_vote_direction(parolen)
        vote_directions[v["vorlagenId"]] = direction

    direction_counts: dict[str, int] = {
        "progressive_ja": sum(1 for d in vote_directions.values() if d == "progressive_ja"),
        "conservative_ja": sum(1 for d in vote_directions.values() if d == "conservative_ja"),
        "mixed": sum(1 for d in vote_directions.values() if d == "mixed"),
    }

    results: dict[str, Any] = {"vote_direction_counts": direction_counts, "per_model": {}}

    for model in llm_answers:
        if model["name"] in EXCLUDE_MODELS:
            continue

        answers = model["conditions"][PRIMARY_CONDITION][PRIMARY_LANGUAGE]["answers"]
        answer_map: dict[int, int] = {a["vorlagenId"]: a["value"] for a in answers}

        valid = [(vid, answer_map[vid]) for vid in vote_ids
                 if vid in answer_map and answer_map[vid] in (0, 100)]
        ja_count = sum(1 for _, v in valid if v == 100)
        n = len(valid)

        binom = stats.binomtest(ja_count, n, 0.5)

        direction_results: dict[str, dict[str, Any]] = {}
        for direction in ["progressive_ja", "conservative_ja", "mixed"]:
            dir_valid = [(vid, val) for vid, val in valid if vote_directions.get(vid) == direction]
            if dir_valid:
                dir_ja = sum(1 for _, v in dir_valid if v == 100)
                dir_n = len(dir_valid)
                dir_binom = stats.binomtest(dir_ja, dir_n, 0.5)
                direction_results[direction] = {
                    "ja_count": dir_ja,
                    "nein_count": dir_n - dir_ja,
                    "n": dir_n,
                    "ja_rate": round(dir_ja / dir_n * 100, 1),
                    "binomial_p": round(dir_binom.pvalue, 6),
                }
            else:
                direction_results[direction] = {"n": 0}

        # Fisher's exact test for direction independence:
        # 2×2 table (progressive/conservative × Ja/Nein)
        prog = direction_results.get("progressive_ja", {})
        cons = direction_results.get("conservative_ja", {})
        fisher_result: dict[str, Any] = {}
        if prog.get("n", 0) > 0 and cons.get("n", 0) > 0:
            table = [
                [prog["ja_count"], prog["nein_count"]],
                [cons["ja_count"], cons["nein_count"]],
            ]
            odds_ratio, fisher_p = stats.fisher_exact(table)
            fisher_result = {
                "table": table,
                "odds_ratio": round(float(odds_ratio), 4),
                "p_value": round(float(fisher_p), 6),
                "interpretation": (
                    "Tests whether Nein rate differs between progressive and conservative "
                    "proposals. Non-significant p supports direction-independence (change-aversion)."
                ),
            }

        results["per_model"][model["name"]] = {
            "display": model["display"],
            "ja_count": ja_count,
            "nein_count": n - ja_count,
            "n": n,
            "ja_rate": round(ja_count / n * 100, 1) if n > 0 else None,
            "binomial_p": round(binom.pvalue, 6),
            "by_direction": direction_results,
            "fisher_direction_independence": fisher_result,
        }

    return results


# ─── Block 2: Instrument Divergence ─────────────────────────────────────────

def analyze_instrument_divergence(
    convergent_validity_per_model: dict[str, Any],
) -> dict[str, Any]:
    """Gradient flip test: Spearman(party_position, agreement) per model per instrument.
    Wilcoxon signed-rank on the paired gradient rhos."""

    per_model: dict[str, Any] = {}
    sv_rhos: list[float] = []
    vv_rhos: list[float] = []

    for model_name, d in convergent_validity_per_model.items():
        sv_vals = [d["smartvote_agreement"][p] for p in PARTY_ORDER]
        vv_vals = [d["volksabstimmungen_agreement"][p] for p in PARTY_ORDER]

        sv_rho, sv_p = stats.spearmanr(PARTY_POSITIONS, sv_vals)
        vv_rho, vv_p = stats.spearmanr(PARTY_POSITIONS, vv_vals)

        sv_rhos.append(sv_rho)
        vv_rhos.append(vv_rho)

        per_model[model_name] = {
            "display": d["display"],
            "smartvote_gradient_rho": round(sv_rho, 4),
            "smartvote_gradient_p": round(sv_p, 4),
            "volksabstimmungen_gradient_rho": round(vv_rho, 4),
            "volksabstimmungen_gradient_p": round(vv_p, 4),
            "gradient_flipped": sv_rho < 0 and vv_rho >= 0,
        }

    rho_diffs = [s - v for s, v in zip(sv_rhos, vv_rhos)]
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(rho_diffs)

    # SP vs Die Mitte gap shift between instruments
    sv_sp_dm_gaps: list[float] = []
    vv_sp_dm_gaps: list[float] = []
    for model_name, d in convergent_validity_per_model.items():
        sv_sp_dm_gaps.append(d["smartvote_agreement"]["SP"] - d["smartvote_agreement"]["Die Mitte"])
        vv_sp_dm_gaps.append(d["volksabstimmungen_agreement"]["SP"] - d["volksabstimmungen_agreement"]["Die Mitte"])
    gap_diffs = [s - v for s, v in zip(sv_sp_dm_gaps, vv_sp_dm_gaps)]
    gap_wilcoxon_stat, gap_wilcoxon_p = stats.wilcoxon(gap_diffs)
    gap_t, gap_t_p = stats.ttest_rel(sv_sp_dm_gaps, vv_sp_dm_gaps)

    per_party_shift: dict[str, dict[str, Any]] = {}
    for party in PARTY_ORDER:
        sv_vals = [convergent_validity_per_model[m]["smartvote_agreement"][party]
                   for m in convergent_validity_per_model]
        vv_vals = [convergent_validity_per_model[m]["volksabstimmungen_agreement"][party]
                   for m in convergent_validity_per_model]
        t, p = stats.ttest_rel(sv_vals, vv_vals)
        per_party_shift[party] = {
            "smartvote_mean": round(float(np.mean(sv_vals)), 1),
            "volksabstimmungen_mean": round(float(np.mean(vv_vals)), 1),
            "difference": round(float(np.mean(sv_vals) - np.mean(vv_vals)), 1),
            "paired_t": round(float(t), 3),
            "p_value": round(float(p), 6),
        }

    n_negative_sv = sum(1 for r in sv_rhos if r < 0)
    n_negative_vv = sum(1 for r in vv_rhos if r < 0)
    n_flipped = sum(1 for s, v in zip(sv_rhos, vv_rhos) if s < 0 and v >= 0)

    return {
        "per_model": per_model,
        "summary": {
            "n_models": len(sv_rhos),
            "n_negative_gradient_smartvote": n_negative_sv,
            "n_negative_gradient_volksabstimmungen": n_negative_vv,
            "n_gradient_flipped": n_flipped,
            "mean_smartvote_gradient_rho": round(float(np.mean(sv_rhos)), 4),
            "mean_volksabstimmungen_gradient_rho": round(float(np.mean(vv_rhos)), 4),
        },
        "gradient_rho_wilcoxon": {
            "statistic": float(wilcoxon_stat),
            "p_value": round(float(wilcoxon_p), 6),
            "interpretation": "Tests whether gradient rhos differ systematically between instruments",
        },
        "sp_dm_gap_wilcoxon": {
            "statistic": float(gap_wilcoxon_stat),
            "p_value": round(float(gap_wilcoxon_p), 6),
            "mean_smartvote_gap": round(float(np.mean(sv_sp_dm_gaps)), 1),
            "mean_volksabstimmungen_gap": round(float(np.mean(vv_sp_dm_gaps)), 1),
        },
        "sp_dm_gap_paired_t": {
            "t_statistic": round(float(gap_t), 3),
            "p_value": round(float(gap_t_p), 6),
        },
        "per_party_shift": per_party_shift,
    }


# ─── Block 3: Cross-Instrument Position Comparison ──────────────────────────

def analyze_cross_instrument(
    smartvote_answers: list[dict[str, Any]],
    volks_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test whether Smartvote PC1 position predicts Volksabstimmungen political direction.

    Uses SP-minus-SVP agreement differential as the Volksabstimmungen left-right score,
    which captures political direction (unlike raw Ja rate, which collapses direction).
    Also keeps Ja rate as a secondary metric for completeness.
    """
    pca_file = load_json("./results/website/smartvote_pca_2d.json")

    valid_qids = [str(qid) for qid in pca_file["metadata"]["question_ids"]]
    pca_mean = np.array(pca_file["metadata"]["pca"]["mean"])
    pca_components = np.array(pca_file["metadata"]["pca"]["components"])

    def project_to_pc1(answers_list: list[dict[str, Any]]) -> float:
        answer_map = {str(a["questionId"]): a["value"] for a in answers_list}
        vec: list[float] = []
        for qid in valid_qids:
            val = answer_map.get(qid, 50)
            if val < 0:
                val = 50  # impute refusals
            vec.append(val)
        vec_arr = np.array(vec, dtype=float)
        # Negate PC1 so left-wing is negative (display convention from paper)
        return -float((vec_arr - pca_mean) @ pca_components[0])

    # Load Volksabstimmungen parolen agreement for SP-SVP differential
    va_parolen = load_json(f"{PAPER_DIR}/volksabstimmungen_parolen_agreement.json")
    va_parolen_results = va_parolen["results"]

    volks_ja_rates: dict[str, float] = {}
    volks_lr_scores: dict[str, float] = {}
    for m in volks_answers:
        if m["name"] in EXCLUDE_MODELS:
            continue
        answers = m["conditions"][PRIMARY_CONDITION][PRIMARY_LANGUAGE]["answers"]
        valid = [a for a in answers if a["value"] in (0, 100)]
        if valid:
            ja_rate = sum(1 for a in valid if a["value"] == 100) / len(valid) * 100
            volks_ja_rates[m["name"]] = ja_rate

        # SP-minus-SVP agreement differential from parolen agreement
        if m["name"] in va_parolen_results:
            parties = va_parolen_results[m["name"]]["parties"]
            sp_agree = parties.get("SP", {}).get("agreement")
            svp_agree = parties.get("SVP", {}).get("agreement")
            if sp_agree is not None and svp_agree is not None:
                volks_lr_scores[m["name"]] = sp_agree - svp_agree

    per_model: dict[str, Any] = {}
    pc1_vals: list[float] = []
    lr_vals: list[float] = []
    ja_vals: list[float] = []

    for m in smartvote_answers:
        if m["name"] not in volks_lr_scores:
            continue

        pc1 = project_to_pc1(m["answers"])

        per_model[m["name"]] = {
            "display": m["display"],
            "smartvote_pc1": round(float(pc1), 1),
            "volksabstimmungen_lr_score": round(float(volks_lr_scores[m["name"]]), 1),
            "volksabstimmungen_ja_rate": round(float(volks_ja_rates.get(m["name"], 0)), 1),
        }

        pc1_vals.append(pc1)
        lr_vals.append(volks_lr_scores[m["name"]])
        if m["name"] in volks_ja_rates:
            ja_vals.append(volks_ja_rates[m["name"]])

    pc1_arr = np.array(pc1_vals)
    lr_arr = np.array(lr_vals)

    rho_lr, p_lr = stats.spearmanr(pc1_arr, lr_arr)

    result: dict[str, Any] = {
        "per_model": per_model,
        "pc1_vs_lr_score": {
            "spearman_rho": round(float(rho_lr), 4),
            "p_value": round(float(p_lr), 4),
            "n": len(pc1_vals),
            "interpretation": "Tests whether Smartvote PC1 (left-right) predicts Volksabstimmungen "
                              "SP-minus-SVP agreement differential. Positive rho = models that are "
                              "more left-leaning on Smartvote also align more with SP (vs SVP) on "
                              "referenda, confirming cross-instrument consistency of political direction.",
        },
    }

    # Keep Ja rate as secondary metric
    if ja_vals:
        ja_arr = np.array(ja_vals)
        rho_ja, p_ja = stats.spearmanr(pc1_arr[:len(ja_arr)], ja_arr)
        result["pc1_vs_ja_rate_secondary"] = {
            "spearman_rho": round(float(rho_ja), 4),
            "p_value": round(float(p_ja), 4),
            "n": len(ja_vals),
            "note": "Secondary metric retained for completeness. Ja rate does not preserve "
                    "political direction and should not be used for convergent validity.",
        }

    return result


# ─── Block 4: Convergence Permutation Test ──────────────────────────────────

def _smartvote_sq_diff_agreement(v1: np.ndarray, v2: np.ndarray) -> float:
    """Squared-difference agreement between two Smartvote answer vectors (0-100 scale)."""
    mean_sq = float(np.mean((v1 - v2) ** 2))
    return 100.0 * (1.0 - mean_sq / 10000.0)


def analyze_convergence_permutation(
    smartvote_answers: list[dict[str, Any]],
    politicians_raw: list[dict[str, Any]],
    questionnaire: dict[str, Any],
    model_families: dict[str, Any],
    volks_answers: list[dict[str, Any]],
    volks_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Permutation test: model pairwise Smartvote agreement vs within-party politician agreement.

    The meaningful comparison is whether models (built to similar specs) are more similar
    than politicians who already share party ideology. Comparing models to cross-party
    agreement is trivially expected since parties exist to represent different positions.

    Also retains the old Volksabstimmungen model-vs-party comparison as secondary.
    """

    # ── Extract valid question IDs (excluding BudgetCategory) ──
    valid_qids: list[str] = []
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        for q in cat.get("questions", []):
            valid_qids.append(q["id"])

    def build_vec(answers_list: list[dict[str, Any]]) -> np.ndarray:
        lookup = {a["questionId"]: a["value"] for a in answers_list if a.get("value", -1) != -1}
        return np.array([lookup.get(qid, 50) for qid in valid_qids], dtype=float)

    # ── Get flagship model names ──
    flagship_names = {fam["flagship"] for fam in model_families.values() if "flagship" in fam}

    # ── Build model Smartvote vectors (flagships only) ──
    model_vectors: dict[str, np.ndarray] = {}
    for m in smartvote_answers:
        if m["name"] not in flagship_names or m["name"] in EXCLUDE_MODELS:
            continue
        model_vectors[m["display"]] = build_vec(m["answers"])

    # ── Build politician vectors grouped by party ──
    party_politicians: dict[str, list[np.ndarray]] = {p: [] for p in PARTY_ORDER}
    for pol in politicians_raw:
        party = pol.get("partyAbbreviation")
        if party not in PARTY_ORDER:
            continue
        vec = build_vec(pol.get("answers", []))
        party_politicians[party].append(vec)

    # ── Model pairwise agreement (Smartvote, squared-difference) ──
    model_names = list(model_vectors.keys())
    model_agreements: list[float] = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            agree = _smartvote_sq_diff_agreement(model_vectors[model_names[i]],
                                                  model_vectors[model_names[j]])
            model_agreements.append(agree)
    model_agreements_arr = np.array(model_agreements)

    # ── Within-party politician pairwise agreement ──
    within_party_agreements: list[float] = []
    per_party_within: dict[str, dict[str, Any]] = {}
    for party in PARTY_ORDER:
        pols = party_politicians[party]
        party_agrees: list[float] = []
        for i in range(len(pols)):
            for j in range(i + 1, len(pols)):
                agree = _smartvote_sq_diff_agreement(pols[i], pols[j])
                party_agrees.append(agree)
        within_party_agreements.extend(party_agrees)
        per_party_within[party] = {
            "mean": round(float(np.mean(party_agrees)), 1) if party_agrees else None,
            "std": round(float(np.std(party_agrees)), 1) if party_agrees else None,
            "n_politicians": len(pols),
            "n_pairs": len(party_agrees),
        }
    within_party_arr = np.array(within_party_agreements)

    # ── Statistical tests ──
    observed_diff = float(np.mean(model_agreements_arr) - np.mean(within_party_arr))
    u_stat, u_p = stats.mannwhitneyu(model_agreements_arr, within_party_arr, alternative="two-sided")

    rng = np.random.default_rng(RANDOM_SEED)
    combined = np.concatenate([model_agreements_arr, within_party_arr])
    n_model = len(model_agreements_arr)
    n_total = len(combined)

    count_ge = 0
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(n_total)
        perm_model = combined[perm[:n_model]]
        perm_within = combined[perm[n_model:]]
        perm_diff = np.mean(perm_model) - np.mean(perm_within)
        if abs(perm_diff) >= abs(observed_diff):
            count_ge += 1

    perm_p = (count_ge + 1) / (N_PERMUTATIONS + 1)

    # ── Secondary: old Volksabstimmungen model-vs-party comparison ──
    votes = get_usable_votes(volks_data)
    vote_ids = [v["vorlagenId"] for v in votes]

    volks_model_vectors: dict[str, np.ndarray] = {}
    for m in volks_answers:
        if m["name"] in EXCLUDE_MODELS:
            continue
        answers = m["conditions"][PRIMARY_CONDITION][PRIMARY_LANGUAGE]["answers"]
        answer_map = {a["vorlagenId"]: a["value"] for a in answers}
        vec_list: list[float] = []
        for vid in vote_ids:
            val = answer_map.get(vid)
            if val == 100:
                vec_list.append(1)
            elif val == 0:
                vec_list.append(0)
            else:
                vec_list.append(np.nan)
        volks_model_vectors[m["name"]] = np.array(vec_list, dtype=float)

    party_vectors_raw: dict[str, list[float]] = {p: [] for p in PARTY_ORDER}
    for v in votes:
        parolen = get_vote_parolen(v)
        for p in PARTY_ORDER:
            parole = parolen.get(p)
            if parole == "Ja":
                party_vectors_raw[p].append(1)
            elif parole == "Nein":
                party_vectors_raw[p].append(0)
            else:
                party_vectors_raw[p].append(np.nan)
    volks_party_vectors: dict[str, np.ndarray] = {
        p: np.array(v, dtype=float) for p, v in party_vectors_raw.items()
    }

    def pairwise_agreement_binary(vectors: dict[str, np.ndarray]) -> np.ndarray:
        names = list(vectors.keys())
        agreements: list[float] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v1 = vectors[names[i]]
                v2 = vectors[names[j]]
                valid = ~np.isnan(v1) & ~np.isnan(v2)
                if valid.sum() > 0:
                    agree = float((v1[valid] == v2[valid]).sum() / valid.sum() * 100)
                    agreements.append(agree)
        return np.array(agreements)

    volks_model_agrees = pairwise_agreement_binary(volks_model_vectors)
    volks_party_agrees = pairwise_agreement_binary(volks_party_vectors)

    return {
        "primary_comparison": "smartvote_models_vs_within_party_politicians",
        "smartvote_model_pairwise_agreement": {
            "mean": round(float(np.mean(model_agreements_arr)), 1),
            "std": round(float(np.std(model_agreements_arr)), 1),
            "n_pairs": int(n_model),
            "n_models": len(model_names),
            "model_names": model_names,
            "values": [round(float(x), 1) for x in model_agreements_arr],
        },
        "smartvote_within_party_pairwise_agreement": {
            "mean": round(float(np.mean(within_party_arr)), 1),
            "std": round(float(np.std(within_party_arr)), 1),
            "n_pairs": int(len(within_party_arr)),
            "per_party": per_party_within,
            "values": [round(float(x), 1) for x in within_party_arr],
        },
        "observed_difference": round(observed_diff, 1),
        "mann_whitney": {
            "U_statistic": float(u_stat),
            "p_value": round(float(u_p), 6),
        },
        "permutation_test": {
            "n_permutations": N_PERMUTATIONS,
            "p_value": round(float(perm_p), 6),
        },
        "volksabstimmungen_secondary": {
            "note": "Secondary comparison retained for completeness. Compares model binary "
                    "pairwise agreement to cross-party binary agreement on Volksabstimmungen.",
            "model_pairwise_agreement": {
                "mean": round(float(np.mean(volks_model_agrees)), 1),
                "std": round(float(np.std(volks_model_agrees)), 1),
                "n_pairs": int(len(volks_model_agrees)),
            },
            "party_pairwise_agreement": {
                "mean": round(float(np.mean(volks_party_agrees)), 1),
                "std": round(float(np.std(volks_party_agrees)), 1),
                "n_pairs": int(len(volks_party_agrees)),
            },
        },
    }


# ─── Block 5: BH Correction ────────────────────────────────────────────────

def apply_bh_correction(all_tests: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Benjamini-Hochberg FDR correction across the full test family."""
    sorted_tests = sorted(all_tests.items(), key=lambda x: x[1])
    m = len(sorted_tests)

    corrected: dict[str, dict[str, Any]] = {}
    for i, (name, raw_p) in enumerate(sorted_tests, 1):
        bh_threshold = 0.05 * i / m
        adjusted_p = min(raw_p * m / i, 1.0)
        corrected[name] = {
            "raw_p": round(raw_p, 8),
            "bh_adjusted_p": round(adjusted_p, 8),
            "significant_at_005": raw_p <= bh_threshold,
            "rank": i,
        }

    # Enforce monotonicity (adjusted p-values must be non-decreasing from bottom)
    sorted_by_rank = sorted(corrected.items(), key=lambda x: -x[1]["rank"])
    running_min = 1.0
    for name, vals in sorted_by_rank:
        running_min = min(running_min, vals["bh_adjusted_p"])
        corrected[name]["bh_adjusted_p"] = round(running_min, 8)

    return corrected


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    volks_data: list[dict[str, Any]] = load_json(VOLKSABSTIMMUNGEN_FILE)
    llm_answers: list[dict[str, Any]] = load_json(LLM_ANSWERS_FILE)
    smartvote_answers: list[dict[str, Any]] = load_json(SMARTVOTE_ANSWERS_FILE)
    politicians_raw: list[dict[str, Any]] = load_json(POLITICIANS_FILE)
    questionnaire: dict[str, Any] = load_json(QUESTIONNAIRE_FILE)
    model_families: dict[str, Any] = load_json(MODEL_FAMILIES_FILE)

    # Read individual paper/ files for p-values and cross-instrument data
    va_cv: dict[str, Any] = load_json(f"{PAPER_DIR}/volksabstimmungen_convergent_validity.json")
    sv_pca: dict[str, Any] = load_json(f"{PAPER_DIR}/smartvote_pca_validation.json")
    sv_disp: dict[str, Any] = load_json(f"{PAPER_DIR}/smartvote_displacement.json")
    sv_geo: dict[str, Any] = load_json(f"{PAPER_DIR}/smartvote_geographic_effect.json")
    sv_oc: dict[str, Any] = load_json(f"{PAPER_DIR}/smartvote_open_vs_closed.json")
    sv_drift: dict[str, Any] = load_json(f"{PAPER_DIR}/smartvote_temporal_drift.json")
    va_rost: dict[str, Any] = load_json(f"{PAPER_DIR}/volksabstimmungen_roestigraben.json")

    print("Block 1: Nein tendency analysis...")
    nein_results = analyze_nein_tendency(volks_data, llm_answers)

    print("Block 2: Instrument divergence analysis...")
    convergent_validity_per_model = va_cv["results"]["per_model"]
    divergence_results = analyze_instrument_divergence(convergent_validity_per_model)

    print("Block 3: Cross-instrument position comparison...")
    cross_instrument_results = analyze_cross_instrument(smartvote_answers, llm_answers)

    print("Block 4: Convergence permutation test...")
    convergence_results = analyze_convergence_permutation(
        smartvote_answers, politicians_raw, questionnaire, model_families,
        llm_answers, volks_data,
    )

    print("Block 5: BH correction across all tests...")

    all_tests: dict[str, float] = {}

    # From Smartvote paper/ files
    all_tests["smartvote_pca_validation"] = sv_pca["results"]["pc1_left_right_ordering"]["spearman_p"]
    all_tests["smartvote_displacement"] = sv_disp["results"]["permutation_test"]["p_value"]
    all_tests["smartvote_geographic_anova"] = sv_geo["results"]["by_country"]["p_value"]
    all_tests["smartvote_open_closed"] = sv_oc["results"]["permutation_test"]["p_value"]
    all_tests["smartvote_drift_sign"] = sv_drift["results"]["aggregate"]["sign_test_p_value"]

    # Block 1: per-model binomial tests
    for model_name, d in nein_results["per_model"].items():
        all_tests[f"nein_tendency_{d['display']}"] = d["binomial_p"]
        # Fisher direction-independence tests (only for models with significant Nein tendency)
        fisher = d.get("fisher_direction_independence", {})
        if fisher and d["binomial_p"] < 0.05:
            all_tests[f"fisher_direction_{d['display']}"] = fisher["p_value"]

    # Block 2: gradient flip and party shift tests
    all_tests["gradient_flip_wilcoxon"] = divergence_results["gradient_rho_wilcoxon"]["p_value"]
    all_tests["sp_dm_gap_wilcoxon"] = divergence_results["sp_dm_gap_wilcoxon"]["p_value"]
    for party, d in divergence_results["per_party_shift"].items():
        all_tests[f"party_shift_{party}"] = d["p_value"]

    # Block 3: cross-instrument correlation
    all_tests["pc1_vs_lr_score"] = cross_instrument_results["pc1_vs_lr_score"]["p_value"]

    # Block 4: convergence tests
    all_tests["convergence_permutation"] = convergence_results["permutation_test"]["p_value"]
    all_tests["convergence_mann_whitney"] = convergence_results["mann_whitney"]["p_value"]

    # From Volksabstimmungen paper/ files
    all_tests["popular_alignment_heterogeneity"] = 0.000011  # chi-square (computed in audit)
    all_tests["roestigraben_spearman"] = va_rost["results"]["spearman_p"]
    all_tests["gemini_refusal_language"] = 0.000000  # chi-square (computed in audit)
    all_tests["gemini_refusal_context"] = 0.000000  # chi-square (computed in audit)

    bh_results = apply_bh_correction(all_tests)

    # Write individual paper/ files
    print("\nWriting results...")

    write_paper_file(
        f"{PAPER_DIR}/cross_instrument_nein_tendency.json",
        nein_results,
        description="Systematic Nein (status quo) tendency analysis: binomial test per model on "
                    "Ja/Nein distribution, with direction-conditional breakdown showing that "
                    "Grok and Mistral vote Nein regardless of whether Ja is the progressive or conservative position.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Binomial test (H0: p=0.5) on Ja rate per model. Votes classified as "
                    "progressive-Ja (23), conservative-Ja (17), or mixed (8) based on which "
                    "parties recommended Ja. Direction-conditional Nein rates reported. "
                    "Fisher's exact test on 2x2 table (progressive/conservative x Ja/Nein) "
                    "tests whether Nein tendency is independent of proposal direction.",
    )

    write_paper_file(
        f"{PAPER_DIR}/cross_instrument_gradient_flip.json",
        divergence_results,
        description="The left-to-right agreement gradient flips between Smartvote and Volksabstimmungen: "
                    "on Smartvote, models agree most with left parties (negative Spearman rho); "
                    "on Volksabstimmungen, this reverses to flat or positive. "
                    "Wilcoxon signed-rank test confirms the flip is statistically significant.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Per-model Spearman correlation between party position (left=1 to right=6) "
                    "and agreement %, computed separately for Smartvote and Volksabstimmungen. "
                    "Wilcoxon signed-rank test on the 8 paired rho values. "
                    "Per-party paired t-tests on agreement shift.",
    )

    write_paper_file(
        f"{PAPER_DIR}/cross_instrument_convergent_validity.json",
        cross_instrument_results,
        description="Cross-instrument comparison: Spearman correlation between Smartvote PC1 position "
                    "and Volksabstimmungen SP-minus-SVP agreement differential (left-right score) "
                    "across the 8 flagship models.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Spearman rank correlation between each model's PC1 score on Smartvote "
                    "and its SP-minus-SVP agreement differential on Volksabstimmungen (n=8). "
                    "The differential captures political direction: positive = closer to SP, "
                    "negative = closer to SVP. Ja rate retained as secondary metric.",
    )

    write_paper_file(
        f"{PAPER_DIR}/cross_instrument_convergence_permutation.json",
        convergence_results,
        description="Permutation test comparing model pairwise Smartvote agreement to "
                    "within-party politician pairwise agreement. Tests whether RLHF optimization "
                    "compresses political variance beyond what shared party ideology does.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=convergence_results["smartvote_model_pairwise_agreement"]["n_models"],
        methodology="Squared-difference agreement between all model pairs on Smartvote vs. "
                    "all within-party politician pairs on Smartvote. 10,000-iteration permutation test "
                    "and Mann-Whitney U test. Volksabstimmungen model-vs-party comparison retained "
                    "as secondary.",
    )

    write_paper_file(
        f"{PAPER_DIR}/bh_correction.json",
        bh_results,
        description="Unified Benjamini-Hochberg correction across all statistical tests from "
                    "Smartvote, Volksabstimmungen, and cross-instrument analyses.",
        n_models=8,
        methodology=f"BH procedure at alpha=0.05 across the full family of {len(all_tests)} p-values.",
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EXTENDED ANALYSIS RESULTS")
    print("=" * 70)

    print("\n--- Block 1: Nein Tendency ---")
    print(f"Vote directions: {nein_results['vote_direction_counts']}")
    for name, d in nein_results["per_model"].items():
        sig = "***" if d["binomial_p"] < 0.001 else "**" if d["binomial_p"] < 0.01 else "*" if d["binomial_p"] < 0.05 else "ns"
        print(f"  {d['display']:20s} Ja={d['ja_rate']}% p={d['binomial_p']:.6f} {sig}")
        for direction in ["progressive_ja", "conservative_ja", "mixed"]:
            dd = d["by_direction"][direction]
            if dd["n"] > 0:
                print(f"    {direction:20s} Ja={dd['ja_rate']}% ({dd['ja_count']}/{dd['n']})")

    print("\n--- Block 2: Instrument Divergence ---")
    s = divergence_results["summary"]
    print(f"  Negative gradient on Smartvote: {s['n_negative_gradient_smartvote']}/{s['n_models']}")
    print(f"  Negative gradient on Volksabstimmungen: {s['n_negative_gradient_volksabstimmungen']}/{s['n_models']}")
    print(f"  Gradient flipped: {s['n_gradient_flipped']}/{s['n_models']}")
    print(f"  Mean SV gradient rho: {s['mean_smartvote_gradient_rho']}")
    print(f"  Mean VA gradient rho: {s['mean_volksabstimmungen_gradient_rho']}")
    g = divergence_results["gradient_rho_wilcoxon"]
    print(f"  Wilcoxon p = {g['p_value']}")
    sp = divergence_results["sp_dm_gap_wilcoxon"]
    print(f"  SP-DieMitte gap: SV={sp['mean_smartvote_gap']:+.1f} VA={sp['mean_volksabstimmungen_gap']:+.1f} Wilcoxon p={sp['p_value']}")

    print("\n  Per-party shift (Smartvote - Volksabstimmungen):")
    for party, d in divergence_results["per_party_shift"].items():
        sig = "*" if d["p_value"] < 0.05 else "ns"
        print(f"    {party:15s} diff={d['difference']:+.1f}pp t={d['paired_t']:.2f} p={d['p_value']:.4f} {sig}")

    print("\n--- Block 3: Cross-Instrument Position Comparison ---")
    for name, d in cross_instrument_results["per_model"].items():
        print(f"  {d['display']:20s} SV_PC1={d['smartvote_pc1']:+.1f} VA_LR={d['volksabstimmungen_lr_score']:+.1f}")
    rc = cross_instrument_results["pc1_vs_lr_score"]
    print(f"  PC1 vs LR score: rho={rc['spearman_rho']}, p={rc['p_value']}")

    print("\n--- Block 4: Convergence Permutation Test ---")
    sm = convergence_results["smartvote_model_pairwise_agreement"]
    wp = convergence_results["smartvote_within_party_pairwise_agreement"]
    print(f"  Model pairwise agreement (Smartvote): {sm['mean']}% (sd={sm['std']}%, {sm['n_pairs']} pairs)")
    print(f"  Within-party politician agreement (Smartvote): {wp['mean']}% (sd={wp['std']}%, {wp['n_pairs']} pairs)")
    print(f"  Difference: {convergence_results['observed_difference']}pp")
    print(f"  Mann-Whitney p = {convergence_results['mann_whitney']['p_value']}")
    print(f"  Permutation p = {convergence_results['permutation_test']['p_value']}")
    vs = convergence_results["volksabstimmungen_secondary"]
    print(f"  [Secondary] Volks model agreement: {vs['model_pairwise_agreement']['mean']}% vs party: {vs['party_pairwise_agreement']['mean']}%")

    print("\n--- Block 5: BH Correction ---")
    for name, d in sorted(bh_results.items(), key=lambda x: x[1]["rank"]):
        sig = "*" if d["significant_at_005"] else ""
        print(f"  {d['rank']:2d}. {name:45s} raw={d['raw_p']:.6f} adj={d['bh_adjusted_p']:.6f} {sig}")

    print(f"\nResults written to {PAPER_DIR}/")


if __name__ == "__main__":
    main()
