#!/usr/bin/env python3
"""
Statistical analysis for the Volksabstimmungen extension.

Analyses:
  V1: Party Parolen agreement (with Gemini exclusion)
  V2: Popular vote alignment (overall + by margin + directional by party position)
  V3: Convergent validity (Smartvote vs Volksabstimmungen, cross-model)
  V4: Cross-linguistic consistency (with McNemar tests)
  V4b: Romansh low-resource stress test (with cross-language baselines)
  V5: Röstigraben correlation (Spearman only, with per-model breakdown)
  V6: Stimmfreigabe / false certainty
  V7: Prompt sensitivity (detail condition effects, with directionality)
  V8: Model convergence on Volksabstimmungen
  V9: Bundesrat agreement
  V10: Refusal rates by language
  V11: Party agreement by language (does agreement shift with prompt language?)
  V12: Per-vote model consensus
  V13: Temporal analysis (agreement by vote year)

Usage:
    uv run python n_volksabstimmungen_analysis.py
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

from _volksabstimmungen_constants import (
    DETAIL_CONDITIONS,
    EXCLUDED_TITLES,
    FRENCH_CANTONS,
    GERMAN_CANTONS,
    MAIN_PARTIES,
    PARTY_NAME_MAP,
)

VOLKSABSTIMMUNGEN_FILE: str = "./data/volksabstimmungen/volksabstimmungen.json"
LLM_ANSWERS_FILE: str = "./data/answers/volksabstimmungen_model_answers.json"
SMARTVOTE_ANSWERS_FILE: str = "./data/answers/all_model_answers.json"
PAPER_DIR: str = "./results/paper"

N_BOOTSTRAP: int = 10_000
RANDOM_SEED: int = 42
MIN_VALID_ANSWERS: int = 10

LANGUAGES: list[str] = ["de", "fr", "it", "rm"]
PRIMARY_CONDITION: str = "in_kuerze"
PRIMARY_LANGUAGE: str = "de"

VOLKS_MODEL_DISPLAY: list[str] = [
    "GPT-5.4", "Claude Opus 4.6", "DeepSeek V3.2", "Llama 4 Maverick",
    "Grok 4.20", "Mistral Large", "Command A", "Qwen 3.5 Plus",
]
VOLKS_MODELS_EXCLUDED: list[str] = ["Gemini 3.1 Pro (98% refusal rate in German)"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_paper_file(
    path: str,
    results: dict,
    *,
    description: str,
    generated_by: str = "n_volksabstimmungen_analysis.py",
    models_analyzed: list[str] | None = None,
    models_excluded: list[str] | None = None,
    n_models: int | None = None,
    methodology: str = "",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    envelope: dict = {
        "description": description,
        "generated_by": generated_by,
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
        json.dump(envelope, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {path}")


def normalize_party(name: str) -> str:
    return PARTY_NAME_MAP.get(name, name)


def get_usable_votes(data: list[dict]) -> list[dict]:
    return [v for v in data if v["titel"] not in EXCLUDED_TITLES and v["parolen"]["parties"]]


def get_vote_parolen(vote: dict) -> dict[str, str | None]:
    """Returns {party: "Ja"/"Nein"/None}."""
    parolen = {}
    for p in vote["parolen"]["parties"]:
        name = normalize_party(p["name"])
        if name in MAIN_PARTIES:
            parole = p["parole"]
            if parole in ("Ja", "Nein"):
                parolen[name] = parole
            else:
                parolen[name] = None
    return parolen


def get_bundesrat_parole(vote: dict) -> str | None:
    for c in vote.get("parolen", {}).get("councils", []):
        if c["name"] == "Bundesrat" and c["parole"] in ("Ja", "Nein"):
            return c["parole"]
    return None


def get_model_answer(model_result: dict, condition: str, lang: str, vorlage_id: int) -> int | None:
    cond_data = model_result.get("conditions", {}).get(condition, {})
    lang_data = cond_data.get(lang, {})
    for answer in lang_data.get("answers", []):
        if answer["vorlagenId"] == vorlage_id:
            return answer["value"]
    return None


def get_model_answer_str(value: int | None) -> str | None:
    if value == 100:
        return "Ja"
    elif value == 0:
        return "Nein"
    return None


def model_valid_count(model: dict, votes: list[dict], condition: str = PRIMARY_CONDITION, lang: str = PRIMARY_LANGUAGE) -> int:
    count = 0
    for vote in votes:
        val = get_model_answer(model, condition, lang, vote["vorlagenId"])
        if val is not None and val != -1:
            count += 1
    return count


def filter_valid_models(models_data: list[dict], votes: list[dict]) -> tuple[list[dict], list[tuple[str, int]]]:
    valid = []
    excluded = []
    for m in models_data:
        n = model_valid_count(m, votes)
        if n >= MIN_VALID_ANSWERS:
            valid.append(m)
        else:
            excluded.append((m["display"], n))
    return valid, excluded


# ─── Analysis V1: Party Parolen Agreement ───────────────────────────────────

def analysis_v1_parolen_agreement(votes: list[dict], models_data: list[dict]) -> dict:
    rng = np.random.RandomState(RANDOM_SEED)
    results = {}

    for model in models_data:
        model_name = model["name"]
        party_results = {}

        for party in MAIN_PARTIES:
            matches = []
            for vote in votes:
                parolen = get_vote_parolen(vote)
                party_parole = parolen.get(party)
                if party_parole is None:
                    continue

                model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
                model_answer = get_model_answer_str(model_val)
                if model_answer is None:
                    continue

                matches.append(1 if model_answer == party_parole else 0)

            if not matches:
                party_results[party] = {"agreement": None, "n": 0}
                continue

            agreement = np.mean(matches) * 100
            boot_agreements = []
            matches_arr = np.array(matches)
            for _ in range(N_BOOTSTRAP):
                sample = rng.choice(matches_arr, size=len(matches_arr), replace=True)
                boot_agreements.append(np.mean(sample) * 100)
            ci_low, ci_high = np.percentile(boot_agreements, [2.5, 97.5])

            party_results[party] = {
                "agreement": round(agreement, 1),
                "ci_low": round(ci_low, 1),
                "ci_high": round(ci_high, 1),
                "n": len(matches),
            }

        results[model_name] = {
            "display": model["display"],
            "parties": party_results,
        }

    return results


# ─── Analysis V2: Popular Vote Alignment ────────────────────────────────────

def analysis_v2_popular_alignment(votes: list[dict], models_data: list[dict]) -> dict:
    results = {}

    for model in models_data:
        model_name = model["name"]
        alignments = {"all": [], "close": [], "moderate": [], "decisive": []}

        for vote in votes:
            ja_pct = vote["resultat"]["jaStimmenInProzent"]
            popular_answer = "Ja" if ja_pct > 50 else "Nein"
            margin = abs(ja_pct - 50)

            model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
            model_answer = get_model_answer_str(model_val)
            if model_answer is None:
                continue

            aligned = 1 if model_answer == popular_answer else 0
            alignments["all"].append(aligned)

            if margin <= 5:
                alignments["close"].append(aligned)
            elif margin <= 10:
                alignments["moderate"].append(aligned)
            else:
                alignments["decisive"].append(aligned)

        result = {}
        for bucket, vals in alignments.items():
            if vals:
                result[bucket] = {
                    "alignment_pct": round(np.mean(vals) * 100, 1),
                    "n": len(vals),
                }
            else:
                result[bucket] = {"alignment_pct": None, "n": 0}

        results[model_name] = {"display": model["display"], "buckets": result}

    # Directional analysis: classify votes by PARTY POSITION, not outcome.
    # "left_position" = SP+Grüne say Ja and SVP says Nein (regardless of what passed)
    # "right_position" = SVP says Ja and SP+Grüne say Nein
    directional = {"left_position": [], "right_position": [], "mixed": []}
    for vote in votes:
        parolen = get_vote_parolen(vote)
        sp = parolen.get("SP")
        gruene = parolen.get("Grüne")
        svp = parolen.get("SVP")

        left_ja = (sp == "Ja") or (gruene == "Ja")
        left_nein = (sp == "Nein") or (gruene == "Nein")
        right_ja = (svp == "Ja")
        right_nein = (svp == "Nein")

        if left_ja and right_nein and not left_nein and not right_ja:
            directional["left_position"].append(vote["vorlagenId"])
        elif right_ja and left_nein and not right_nein and not left_ja:
            directional["right_position"].append(vote["vorlagenId"])
        else:
            directional["mixed"].append(vote["vorlagenId"])

    for model in models_data:
        model_name = model["name"]
        for coding, vote_ids in directional.items():
            agree_with_left = 0
            agree_with_right = 0
            total = 0
            for vote in votes:
                if vote["vorlagenId"] not in vote_ids:
                    continue
                parolen = get_vote_parolen(vote)
                sp = parolen.get("SP") or parolen.get("Grüne")
                svp = parolen.get("SVP")
                model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
                model_answer = get_model_answer_str(model_val)
                if model_answer is None:
                    continue
                total += 1
                if coding == "left_position":
                    if model_answer == "Ja":
                        agree_with_left += 1
                    else:
                        agree_with_right += 1
                elif coding == "right_position":
                    if model_answer == "Ja":
                        agree_with_right += 1
                    else:
                        agree_with_left += 1

            results[model_name][f"directional_{coding}"] = {
                "agree_with_left": agree_with_left,
                "agree_with_right": agree_with_right,
                "n": total,
                "left_agreement_pct": round(agree_with_left / total * 100, 1) if total > 0 else None,
            }

    results["_directional_counts"] = {k: len(v) for k, v in directional.items()}

    close_votes_detail = []
    for vote in votes:
        ja_pct = vote["resultat"]["jaStimmenInProzent"]
        margin = abs(ja_pct - 50)
        if margin > 5:
            continue
        popular_answer = "Ja" if ja_pct > 50 else "Nein"
        model_positions = {}
        for model in models_data:
            val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
            model_answer = get_model_answer_str(val)
            model_positions[model["display"]] = model_answer

        n_agree = sum(1 for v in model_positions.values() if v == popular_answer)
        n_disagree = sum(1 for v in model_positions.values() if v is not None and v != popular_answer)
        close_votes_detail.append({
            "vorlagenId": vote["vorlagenId"],
            "titel": vote["titel"][:80],
            "date": vote["abstimmtag"],
            "ja_pct": round(ja_pct, 1),
            "margin_pct": round(margin, 1),
            "popular_outcome": popular_answer,
            "models_agree": n_agree,
            "models_disagree": n_disagree,
            "model_positions": model_positions,
        })
    close_votes_detail.sort(key=lambda x: x["margin_pct"])
    results["_close_votes_detail"] = close_votes_detail

    # ── Temporal split: pre-release vs post-release votes (memorization check) ──
    # Load release dates from Smartvote answers (cross-reference by model name)
    smartvote_raw = load_json(SMARTVOTE_ANSWERS_FILE)
    release_dates: dict[str, str] = {}
    for m in smartvote_raw:
        if m.get("released"):
            release_dates[m["name"]] = m["released"]

    temporal_results: dict[str, Any] = {}
    for model in models_data:
        model_name = model["name"]
        released = release_dates.get(model_name)
        if released is None:
            temporal_results[model_name] = {
                "display": model["display"],
                "release_date": None,
                "note": "No release date available; excluded from temporal analysis.",
            }
            continue

        pre_aligned, post_aligned = [], []
        for vote in votes:
            vote_date = vote["abstimmtag"]
            model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
            model_answer = get_model_answer_str(model_val)
            if model_answer is None:
                continue
            ja_pct = vote["resultat"]["jaStimmenInProzent"]
            popular_answer = "Ja" if ja_pct > 50 else "Nein"
            aligned = 1 if model_answer == popular_answer else 0

            if vote_date < released:
                pre_aligned.append(aligned)
            else:
                post_aligned.append(aligned)

        temporal_entry: dict[str, Any] = {
            "display": model["display"],
            "release_date": released,
            "pre_release": {
                "alignment_pct": round(np.mean(pre_aligned) * 100, 1) if pre_aligned else None,
                "n": len(pre_aligned),
            },
            "post_release": {
                "alignment_pct": round(np.mean(post_aligned) * 100, 1) if post_aligned else None,
                "n": len(post_aligned),
            },
        }

        # Fisher's exact test if both buckets have data
        if pre_aligned and post_aligned:
            pre_yes = sum(pre_aligned)
            pre_no = len(pre_aligned) - pre_yes
            post_yes = sum(post_aligned)
            post_no = len(post_aligned) - post_yes
            table = [[pre_yes, pre_no], [post_yes, post_no]]
            odds_ratio, fisher_p = stats.fisher_exact(table)
            temporal_entry["fisher_exact"] = {
                "table": table,
                "odds_ratio": round(float(odds_ratio), 4),
                "p_value": round(float(fisher_p), 6),
                "interpretation": "Tests whether alignment differs between votes the model "
                                  "could vs could not have seen in training data.",
            }

        temporal_results[model_name] = temporal_entry

    results["_temporal_split"] = temporal_results

    return results


# ─── Analysis V3: Convergent Validity ───────────────────────────────────────

def analysis_v3_convergent_validity(votes: list[dict], models_data: list[dict], smartvote_data: list[dict]) -> dict:
    """Pool all (model, party) pairs into a single scatter and test the overall correlation."""
    va_agreement = {}
    for model in models_data:
        model_name = model["name"]
        party_agreements = {}
        for party in MAIN_PARTIES:
            matches = []
            for vote in votes:
                parolen = get_vote_parolen(vote)
                party_parole = parolen.get(party)
                if party_parole is None:
                    continue
                model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
                model_answer = get_model_answer_str(model_val)
                if model_answer is None:
                    continue
                matches.append(1 if model_answer == party_parole else 0)
            party_agreements[party] = round(np.mean(matches) * 100, 1) if matches else None
        va_agreement[model_name] = party_agreements

    smartvote_agreement = {}
    if smartvote_data:
        try:
            politicians_raw = load_json("./data/answers/nationalrat_members.json")
            questionnaire = load_json("./data/questionnaire/questionnaire.json")
            question_ids = []
            for cat in questionnaire.get("categories", []):
                if cat.get("type") == "BudgetCategory":
                    continue
                for q in cat.get("questions", []):
                    question_ids.append(q["id"])

            party_answers = {p: defaultdict(list) for p in MAIN_PARTIES}
            for pol in politicians_raw:
                party = pol.get("partyAbbreviation")
                if party not in MAIN_PARTIES:
                    continue
                for ans in pol.get("answers", []):
                    if ans["questionId"] in question_ids:
                        party_answers[party][ans["questionId"]].append(ans["value"])
            party_means = {}
            for party in MAIN_PARTIES:
                party_means[party] = {qid: np.mean(vals) for qid, vals in party_answers[party].items()}

            for sv_model in smartvote_data:
                if sv_model["name"] not in [m["name"] for m in models_data]:
                    continue
                model_answers = {}
                for ans in sv_model.get("answers", []):
                    if ans.get("value", -1) != -1 and ans["questionId"] in question_ids:
                        model_answers[ans["questionId"]] = ans["value"]
                if not model_answers:
                    continue
                agreements = {}
                for party in MAIN_PARTIES:
                    diffs = []
                    for qid in question_ids:
                        m_val = model_answers.get(qid, 50)
                        p_val = party_means[party].get(qid, 50)
                        diffs.append((m_val - p_val) ** 2)
                    if diffs:
                        agreements[party] = round(100 * (1 - np.mean(diffs) / 10000), 1)
                smartvote_agreement[sv_model["name"]] = agreements
        except FileNotFoundError:
            pass

    pooled_va = []
    pooled_sv = []
    per_model_results = {}

    for model in models_data:
        model_name = model["name"]
        va = va_agreement.get(model_name, {})
        sv = smartvote_agreement.get(model_name, {})

        model_va = []
        model_sv = []
        for party in MAIN_PARTIES:
            va_val = va.get(party)
            sv_val = sv.get(party)
            if va_val is not None and sv_val is not None:
                pooled_va.append(va_val)
                pooled_sv.append(sv_val)
                model_va.append(va_val)
                model_sv.append(sv_val)

        per_model_results[model_name] = {
            "display": model["display"],
            "volksabstimmungen_agreement": va,
            "smartvote_agreement": sv,
        }

    result = {"per_model": per_model_results}
    if len(pooled_va) >= 10:
        rho, p_value = stats.spearmanr(pooled_va, pooled_sv)
        result["pooled"] = {
            "spearman_rho": round(rho, 3),
            "p_value": round(p_value, 4),
            "n_datapoints": len(pooled_va),
            "note": f"Pooled across {len(models_data)} models × {len(MAIN_PARTIES)} parties",
        }
    else:
        result["pooled"] = {"spearman_rho": None, "p_value": None, "n_datapoints": len(pooled_va)}

    per_party = {}
    for party in MAIN_PARTIES:
        va_vals = []
        sv_vals = []
        for model in models_data:
            va_val = va_agreement.get(model["name"], {}).get(party)
            sv_val = smartvote_agreement.get(model["name"], {}).get(party)
            if va_val is not None and sv_val is not None:
                va_vals.append(va_val)
                sv_vals.append(sv_val)
        if len(va_vals) >= 4:
            rho, p = stats.spearmanr(va_vals, sv_vals)
            per_party[party] = {"spearman_rho": round(rho, 3), "p_value": round(p, 4), "n": len(va_vals)}
        else:
            per_party[party] = {"spearman_rho": None, "p_value": None, "n": len(va_vals)}
    result["per_party"] = per_party

    return result


# ─── Analysis V4: Cross-Linguistic Consistency ──────────────────────────────

def analysis_v4_cross_linguistic(votes: list[dict], models_data: list[dict]) -> dict:
    results = {}

    for model in models_data:
        model_name = model["name"]

        consistency_counts = {"all_agree": 0, "some_disagree": 0, "total": 0}
        language_pair_matches = defaultdict(list)

        for vote in votes:
            answers_by_lang = {}
            for lang in LANGUAGES:
                val = get_model_answer(model, PRIMARY_CONDITION, lang, vote["vorlagenId"])
                if val is not None and val != -1:
                    answers_by_lang[lang] = val

            if len(answers_by_lang) < 2:
                continue

            consistency_counts["total"] += 1
            if len(set(answers_by_lang.values())) == 1:
                consistency_counts["all_agree"] += 1
            else:
                consistency_counts["some_disagree"] += 1

            langs = list(answers_by_lang.keys())
            for i in range(len(langs)):
                for j in range(i + 1, len(langs)):
                    pair = tuple(sorted([langs[i], langs[j]]))
                    match = 1 if answers_by_lang[langs[i]] == answers_by_lang[langs[j]] else 0
                    language_pair_matches[pair].append(match)

        pair_stats = {}
        for pair, matches in sorted(language_pair_matches.items()):
            agreement = np.mean(matches) * 100
            n = len(matches)

            a_yes_b_no = 0
            a_no_b_yes = 0
            lang_a, lang_b = pair
            for vote in votes:
                val_a = get_model_answer(model, PRIMARY_CONDITION, lang_a, vote["vorlagenId"])
                val_b = get_model_answer(model, PRIMARY_CONDITION, lang_b, vote["vorlagenId"])
                if val_a is None or val_b is None or val_a == -1 or val_b == -1:
                    continue
                if val_a == 100 and val_b == 0:
                    a_yes_b_no += 1
                elif val_a == 0 and val_b == 100:
                    a_no_b_yes += 1

            total_discordant = a_yes_b_no + a_no_b_yes
            if total_discordant > 0:
                mcnemar_p = stats.binomtest(
                    min(a_yes_b_no, a_no_b_yes), total_discordant, 0.5,
                ).pvalue
            else:
                mcnemar_p = 1.0

            pair_stats[f"{lang_a}-{lang_b}"] = {
                "agreement_pct": round(agreement, 1),
                "n": n,
                "discordant": {
                    f"{lang_a}_ja_{lang_b}_nein": a_yes_b_no,
                    f"{lang_a}_nein_{lang_b}_ja": a_no_b_yes,
                },
                "mcnemar_p": round(mcnemar_p, 4),
            }

        total = consistency_counts["total"]
        results[model_name] = {
            "display": model["display"],
            "consistency_rate": round(consistency_counts["all_agree"] / total * 100, 1) if total > 0 else None,
            "n_votes": total,
            "counts": consistency_counts,
            "language_pairs": pair_stats,
        }

    return results


# ─── Analysis V4b: Romansh Low-Resource Stress Test ─────────────────────────

def analysis_v4b_romansh(votes: list[dict], models_data: list[dict]) -> dict:
    results = {"per_model": {}, "aggregate": {}}

    all_stats = defaultdict(list)

    for model in models_data:
        model_name = model["name"]
        pair_counts = {}
        for lang_a, lang_b in [("rm", "de"), ("rm", "fr"), ("rm", "it"), ("de", "fr")]:
            matches = 0
            both_answered = 0
            a_refusals = 0
            b_refusals = 0
            a_total = 0
            b_total = 0

            for vote in votes:
                val_a = get_model_answer(model, PRIMARY_CONDITION, lang_a, vote["vorlagenId"])
                val_b = get_model_answer(model, PRIMARY_CONDITION, lang_b, vote["vorlagenId"])

                if val_a is not None:
                    a_total += 1
                    if val_a == -1:
                        a_refusals += 1
                if val_b is not None:
                    b_total += 1
                    if val_b == -1:
                        b_refusals += 1

                if val_a is not None and val_b is not None and val_a != -1 and val_b != -1:
                    both_answered += 1
                    if val_a == val_b:
                        matches += 1

            agreement = round(matches / both_answered * 100, 1) if both_answered > 0 else None
            pair_counts[f"{lang_a}_{lang_b}"] = {
                "agreement_pct": agreement,
                "both_answered": both_answered,
                f"{lang_a}_refusal_rate": round(a_refusals / a_total * 100, 1) if a_total > 0 else None,
                f"{lang_b}_refusal_rate": round(b_refusals / b_total * 100, 1) if b_total > 0 else None,
            }
            if agreement is not None:
                all_stats[f"{lang_a}_{lang_b}_agreement"].append(agreement)

        results["per_model"][model_name] = {
            "display": model["display"],
            "pairs": pair_counts,
        }

    results["aggregate"] = {}
    for key in ["rm_de_agreement", "rm_fr_agreement", "rm_it_agreement", "de_fr_agreement"]:
        vals = all_stats.get(key, [])
        results["aggregate"][key] = round(np.mean(vals), 1) if vals else None

    return results


# ─── Analysis V5: Röstigraben Correlation ───────────────────────────────────

def analysis_v5_roestigraben(votes: list[dict], models_data: list[dict]) -> dict:
    """Spearman correlation between actual DE-FR cantonal voting gap and model DE-FR answer gap."""
    vote_gaps = []
    per_model_gaps = {m["name"]: {} for m in models_data}

    for vote in votes:
        de_ja_pcts = []
        fr_ja_pcts = []
        for canton in vote["kantone"]:
            geo = canton["geoLevelnummer"]
            ja_pct = canton["jaStimmenInProzent"]
            if geo in GERMAN_CANTONS:
                de_ja_pcts.append(ja_pct)
            elif geo in FRENCH_CANTONS:
                fr_ja_pcts.append(ja_pct)

        if not de_ja_pcts or not fr_ja_pcts:
            continue

        actual_gap = np.mean(de_ja_pcts) - np.mean(fr_ja_pcts)
        vid = vote["vorlagenId"]
        vote_gaps.append({"vorlagenId": vid, "titel": vote["titel"][:60], "actual_gap": actual_gap})

        for model in models_data:
            val_de = get_model_answer(model, PRIMARY_CONDITION, "de", vote["vorlagenId"])
            val_fr = get_model_answer(model, PRIMARY_CONDITION, "fr", vote["vorlagenId"])
            if val_de is not None and val_fr is not None and val_de != -1 and val_fr != -1:
                per_model_gaps[model["name"]][vid] = val_de - val_fr

    actual_gaps = []
    avg_model_gaps = []
    per_vote = []

    for vg in vote_gaps:
        vid = vg["vorlagenId"]
        model_gaps = [per_model_gaps[m["name"]][vid] for m in models_data if vid in per_model_gaps[m["name"]]]
        if model_gaps:
            avg_gap = np.mean(model_gaps)
            actual_gaps.append(vg["actual_gap"])
            avg_model_gaps.append(avg_gap)
            per_vote.append({**vg, "avg_model_gap": round(avg_gap, 1), "n_models": len(model_gaps)})

    result = {"n_votes": len(actual_gaps), "per_vote": per_vote}

    if len(actual_gaps) >= 5:
        rho, p_value = stats.spearmanr(actual_gaps, avg_model_gaps)
        result["spearman_rho"] = round(rho, 3)
        result["spearman_p"] = round(p_value, 4)
    else:
        result["spearman_rho"] = None
        result["spearman_p"] = None

    per_model_rho = {}
    for model in models_data:
        m_actual = []
        m_gaps = []
        for vg in vote_gaps:
            vid = vg["vorlagenId"]
            if vid in per_model_gaps[model["name"]]:
                m_actual.append(vg["actual_gap"])
                m_gaps.append(per_model_gaps[model["name"]][vid])
        if len(m_actual) >= 10:
            rho, p = stats.spearmanr(m_actual, m_gaps)
            per_model_rho[model["name"]] = {"display": model["display"], "rho": round(rho, 3), "p": round(p, 4), "n": len(m_actual)}
    result["per_model"] = per_model_rho

    # Sensitivity: include bilingual cantons in the Röstigraben boundary
    bi_de = GERMAN_CANTONS + [2, 23]
    bi_fr = FRENCH_CANTONS + [10]
    bi_actual = []
    bi_model = []
    for vote in votes:
        de_pcts = [c["jaStimmenInProzent"] for c in vote["kantone"] if c["geoLevelnummer"] in bi_de]
        fr_pcts = [c["jaStimmenInProzent"] for c in vote["kantone"] if c["geoLevelnummer"] in bi_fr]
        if de_pcts and fr_pcts:
            gap = np.mean(de_pcts) - np.mean(fr_pcts)
            vid = vote["vorlagenId"]
            mgaps = [per_model_gaps[m["name"]][vid] for m in models_data if vid in per_model_gaps[m["name"]]]
            if mgaps:
                bi_actual.append(gap)
                bi_model.append(np.mean(mgaps))
    if len(bi_actual) >= 5:
        rho, p = stats.spearmanr(bi_actual, bi_model)
        result["sensitivity_bilingual"] = {"spearman_rho": round(rho, 3), "spearman_p": round(p, 4), "n": len(bi_actual)}

    return result


# ─── Analysis V6: Stimmfreigabe / False Certainty ──────────────────────────

def analysis_v6_stimmfreigabe(votes: list[dict], models_data: list[dict]) -> dict:
    high_ambiguity_ids = set()
    some_ambiguity_ids = set()
    clear_ids = set()

    for vote in votes:
        parolen = get_vote_parolen(vote)
        abstentions = sum(1 for p in MAIN_PARTIES if parolen.get(p) is None)
        if abstentions >= 2:
            high_ambiguity_ids.add(vote["vorlagenId"])
        elif abstentions >= 1:
            some_ambiguity_ids.add(vote["vorlagenId"])
        else:
            clear_ids.add(vote["vorlagenId"])

    results = {
        "n_high_ambiguity": len(high_ambiguity_ids),
        "n_some_ambiguity": len(some_ambiguity_ids),
        "n_clear": len(clear_ids),
        "note": f"Only {len(high_ambiguity_ids)} high-ambiguity votes — insufficient for statistical comparison" if len(high_ambiguity_ids) < 5 else None,
        "per_model": {},
    }

    for model in models_data:
        model_name = model["name"]
        ambig_positions = 0
        ambig_total = 0
        clear_positions = 0
        clear_total = 0

        for vote in votes:
            val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
            if val is None:
                continue
            has_position = val in (0, 100)
            vid = vote["vorlagenId"]

            if vid in high_ambiguity_ids or vid in some_ambiguity_ids:
                ambig_total += 1
                if has_position:
                    ambig_positions += 1
            elif vid in clear_ids:
                clear_total += 1
                if has_position:
                    clear_positions += 1

        results["per_model"][model_name] = {
            "display": model["display"],
            "ambiguous_position_rate": round(ambig_positions / ambig_total * 100, 1) if ambig_total > 0 else None,
            "clear_position_rate": round(clear_positions / clear_total * 100, 1) if clear_total > 0 else None,
            "ambiguous_n": ambig_total,
            "clear_n": clear_total,
        }

    return results


# ─── Analysis V7: Prompt Sensitivity ────────────────────────────────────────

def analysis_v7_prompt_sensitivity(votes: list[dict], models_data: list[dict]) -> dict:
    results = {}

    for model in models_data:
        model_name = model["name"]
        conditions = list(DETAIL_CONDITIONS.keys())
        consistency = {"all_agree": 0, "total": 0}
        condition_pair_changes = defaultdict(lambda: {"same": 0, "changed": 0, "ja_to_nein": 0, "nein_to_ja": 0})

        for vote in votes:
            answers = {}
            for cond in conditions:
                val = get_model_answer(model, cond, PRIMARY_LANGUAGE, vote["vorlagenId"])
                if val is not None and val != -1:
                    answers[cond] = val

            if len(answers) < 2:
                continue

            consistency["total"] += 1
            if len(set(answers.values())) == 1:
                consistency["all_agree"] += 1

            cond_list = list(answers.keys())
            for i in range(len(cond_list)):
                for j in range(i + 1, len(cond_list)):
                    c1, c2 = cond_list[i], cond_list[j]
                    pair_key = f"{c1}_vs_{c2}"
                    if answers[c1] == answers[c2]:
                        condition_pair_changes[pair_key]["same"] += 1
                    else:
                        condition_pair_changes[pair_key]["changed"] += 1
                        if answers[c1] == 100 and answers[c2] == 0:
                            condition_pair_changes[pair_key]["ja_to_nein"] += 1
                        elif answers[c1] == 0 and answers[c2] == 100:
                            condition_pair_changes[pair_key]["nein_to_ja"] += 1

        total = consistency["total"]

        bundesrat_shift = {"toward_br": 0, "away_br": 0, "total_changed": 0}
        ik_key = "in_kuerze"
        ft_key = "full_text"
        for vote in votes:
            val_ik = get_model_answer(model, ik_key, PRIMARY_LANGUAGE, vote["vorlagenId"])
            val_ft = get_model_answer(model, ft_key, PRIMARY_LANGUAGE, vote["vorlagenId"])
            if val_ik is None or val_ft is None or val_ik == -1 or val_ft == -1:
                continue
            if val_ik == val_ft:
                continue
            bundesrat_shift["total_changed"] += 1
            br = get_bundesrat_parole(vote)
            if br is None:
                continue
            br_val = 100 if br == "Ja" else 0
            dist_before = abs(val_ik - br_val)
            dist_after = abs(val_ft - br_val)
            if dist_after < dist_before:
                bundesrat_shift["toward_br"] += 1
            elif dist_after > dist_before:
                bundesrat_shift["away_br"] += 1

        results[model_name] = {
            "display": model["display"],
            "consistency_rate": round(consistency["all_agree"] / total * 100, 1) if total > 0 else None,
            "n_votes": total,
            "condition_pairs": dict(condition_pair_changes),
            "bundesrat_shift": bundesrat_shift,
        }

    return results


# ─── Analysis V8: Model Convergence ─────────────────────────────────────────

def analysis_v8_convergence(votes: list[dict], models_data: list[dict]) -> dict:
    model_vectors = {}
    for model in models_data:
        vec = []
        for vote in votes:
            val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
            vec.append(val if val is not None and val != -1 else 50)
        model_vectors[model["name"]] = np.array(vec, dtype=float)

    party_vectors = {}
    for party in MAIN_PARTIES:
        vec = []
        for vote in votes:
            parolen = get_vote_parolen(vote)
            p = parolen.get(party)
            if p == "Ja":
                vec.append(100)
            elif p == "Nein":
                vec.append(0)
            else:
                vec.append(50)
        party_vectors[party] = np.array(vec, dtype=float)

    model_names = list(model_vectors.keys())
    model_pairwise = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            v1 = model_vectors[model_names[i]]
            v2 = model_vectors[model_names[j]]
            agreement = np.mean(v1 == v2) * 100
            model_pairwise.append(agreement)

    party_names = list(party_vectors.keys())
    party_pairwise = []
    for i in range(len(party_names)):
        for j in range(i + 1, len(party_names)):
            v1 = party_vectors[party_names[i]]
            v2 = party_vectors[party_names[j]]
            agreement = np.mean(v1 == v2) * 100
            party_pairwise.append(agreement)

    return {
        "model_mean_pairwise_agreement": round(np.mean(model_pairwise), 1),
        "model_std_pairwise_agreement": round(np.std(model_pairwise), 1),
        "party_mean_pairwise_agreement": round(np.mean(party_pairwise), 1),
        "party_std_pairwise_agreement": round(np.std(party_pairwise), 1),
        "n_model_pairs": len(model_pairwise),
        "n_party_pairs": len(party_pairwise),
        "interpretation": "Models more similar to each other than parties are to each other"
        if np.mean(model_pairwise) > np.mean(party_pairwise) else
        "Models are NOT more similar to each other than parties",
    }


# ─── Analysis V9: Bundesrat Agreement ───────────────────────────────────────

def analysis_v9_bundesrat(votes: list[dict], models_data: list[dict]) -> dict:
    results = {}

    for model in models_data:
        model_name = model["name"]
        matches = []
        for vote in votes:
            br = get_bundesrat_parole(vote)
            if br is None:
                continue
            model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vote["vorlagenId"])
            model_answer = get_model_answer_str(model_val)
            if model_answer is None:
                continue
            matches.append(1 if model_answer == br else 0)

        if matches:
            results[model_name] = {
                "display": model["display"],
                "agreement_pct": round(np.mean(matches) * 100, 1),
                "n": len(matches),
            }

    return results


# ─── Analysis V10: Refusal Rates by Language ────────────────────────────────

def analysis_v10_refusal_by_language(votes: list[dict], all_models_data: list[dict]) -> dict:
    results = {}

    for model in all_models_data:
        model_name = model["name"]
        lang_stats = {}
        for lang in LANGUAGES:
            for cond_name in DETAIL_CONDITIONS:
                total = 0
                refused = 0
                for vote in votes:
                    val = get_model_answer(model, cond_name, lang, vote["vorlagenId"])
                    if val is not None:
                        total += 1
                        if val == -1:
                            refused += 1
                key = f"{cond_name}/{lang}"
                if total > 0 and refused > 0:
                    lang_stats[key] = {
                        "refusal_rate": round(refused / total * 100, 1),
                        "refused": refused,
                        "total": total,
                    }

        if lang_stats:
            results[model_name] = {"display": model["display"], "by_condition_language": lang_stats}

    return results


# ─── Analysis V11: Party Agreement by Language ──────────────────────────────

def analysis_v11_party_agreement_by_language(votes: list[dict], models_data: list[dict]) -> dict:
    results = {}

    for model in models_data:
        model_name = model["name"]
        by_lang = {}

        for lang in LANGUAGES:
            party_agreements = {}
            for party in MAIN_PARTIES:
                matches = []
                for vote in votes:
                    parolen = get_vote_parolen(vote)
                    party_parole = parolen.get(party)
                    if party_parole is None:
                        continue
                    model_val = get_model_answer(model, PRIMARY_CONDITION, lang, vote["vorlagenId"])
                    model_answer = get_model_answer_str(model_val)
                    if model_answer is None:
                        continue
                    matches.append(1 if model_answer == party_parole else 0)
                party_agreements[party] = round(np.mean(matches) * 100, 1) if matches else None
            by_lang[lang] = party_agreements

        results[model_name] = {"display": model["display"], "by_language": by_lang}

    return results


# ─── Analysis V12: Per-Vote Model Consensus ─────────────────────────────────

def analysis_v12_consensus(votes: list[dict], models_data: list[dict]) -> dict:
    per_vote = []

    for vote in votes:
        vid = vote["vorlagenId"]
        ja_count = 0
        nein_count = 0
        refused_count = 0

        for model in models_data:
            val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vid)
            if val == 100:
                ja_count += 1
            elif val == 0:
                nein_count += 1
            elif val == -1:
                refused_count += 1

        total_answered = ja_count + nein_count
        if total_answered == 0:
            continue

        consensus_pct = max(ja_count, nein_count) / total_answered * 100
        majority = "Ja" if ja_count > nein_count else "Nein"

        per_vote.append({
            "vorlagenId": vid,
            "titel": vote["titel"][:60],
            "date": vote["abstimmtag"],
            "ja_pct_popular": round(vote["resultat"]["jaStimmenInProzent"], 1),
            "model_ja": ja_count,
            "model_nein": nein_count,
            "model_refused": refused_count,
            "model_majority": majority,
            "consensus_pct": round(consensus_pct, 1),
            "unanimous": ja_count == 0 or nein_count == 0,
        })

    unanimous = sum(1 for v in per_vote if v["unanimous"])
    split = len(per_vote) - unanimous

    return {
        "per_vote": per_vote,
        "n_unanimous": unanimous,
        "n_split": split,
        "mean_consensus_pct": round(np.mean([v["consensus_pct"] for v in per_vote]), 1) if per_vote else None,
    }


# ─── Analysis V13: Temporal Analysis ────────────────────────────────────────

def analysis_v13_temporal(votes: list[dict], models_data: list[dict]) -> dict:
    results = {}

    early_vids = set()
    late_vids = set()
    for vote in votes:
        year = int(vote["abstimmtag"][:4])
        if year <= 2022:
            early_vids.add(vote["vorlagenId"])
        elif year >= 2024:
            late_vids.add(vote["vorlagenId"])

    for model in models_data:
        model_name = model["name"]
        early_aligned = []
        late_aligned = []

        for vote in votes:
            vid = vote["vorlagenId"]
            ja_pct = vote["resultat"]["jaStimmenInProzent"]
            popular = "Ja" if ja_pct > 50 else "Nein"
            model_val = get_model_answer(model, PRIMARY_CONDITION, PRIMARY_LANGUAGE, vid)
            model_answer = get_model_answer_str(model_val)
            if model_answer is None:
                continue

            aligned = 1 if model_answer == popular else 0
            if vid in early_vids:
                early_aligned.append(aligned)
            elif vid in late_vids:
                late_aligned.append(aligned)

        results[model_name] = {
            "display": model["display"],
            "early_alignment_pct": round(np.mean(early_aligned) * 100, 1) if early_aligned else None,
            "late_alignment_pct": round(np.mean(late_aligned) * 100, 1) if late_aligned else None,
            "early_n": len(early_aligned),
            "late_n": len(late_aligned),
        }

    results["_period_counts"] = {
        "early_2021_2022": len(early_vids),
        "late_2024_2026": len(late_vids),
        "middle_2023": len(votes) - len(early_vids) - len(late_vids),
    }

    return results


# ─── BH Correction ──────────────────────────────────────────────────────────

def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda x: x[1])
    n = len(items)
    adjusted = {}
    prev_adj = 0.0
    for rank_minus_1, (label, p) in enumerate(reversed(items)):
        rank = n - rank_minus_1
        adj_p = min(1.0, p * n / rank)
        adj_p = min(adj_p, prev_adj) if rank_minus_1 > 0 else adj_p
        adjusted[label] = adj_p
        prev_adj = adj_p
    sorted_by_p = sorted(p_values.items(), key=lambda x: x[1])
    result = {}
    running_max = 0.0
    for label, _ in sorted_by_p:
        adj = max(adjusted[label], running_max)
        result[label] = adj
        running_max = adj
    return result


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    va_data = load_json(VOLKSABSTIMMUNGEN_FILE)
    votes = get_usable_votes(va_data)
    print(f"  {len(votes)} usable votes")

    try:
        all_models_data = load_json(LLM_ANSWERS_FILE)
        print(f"  {len(all_models_data)} models with Volksabstimmungen answers")
    except FileNotFoundError:
        print("ERROR: No LLM answers found. Run l_generate_volksabstimmungen_dataset.py first.")
        return

    models_data, excluded = filter_valid_models(all_models_data, votes)
    if excluded:
        print(f"  Excluded from analysis (< {MIN_VALID_ANSWERS} valid answers):")
        for name, n in excluded:
            print(f"    {name}: {n} valid answers")
    print(f"  {len(models_data)} models included in analysis")

    try:
        smartvote_data = load_json(SMARTVOTE_ANSWERS_FILE)
    except FileNotFoundError:
        smartvote_data = []
        print("  WARNING: No Smartvote data found — V3 convergent validity will be limited")

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_votes": len(votes),
            "n_models": len(models_data),
            "n_models_excluded": len(excluded),
            "excluded_models": [{"name": name, "valid_answers": n} for name, n in excluded],
            "model_names": [m["name"] for m in models_data],
            "primary_condition": PRIMARY_CONDITION,
            "primary_language": PRIMARY_LANGUAGE,
            "min_valid_answers": MIN_VALID_ANSWERS,
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
        },
    }

    print("\nV1: Party Parolen agreement...")
    output["v1_parolen_agreement"] = analysis_v1_parolen_agreement(votes, models_data)

    print("V2: Popular vote alignment...")
    output["v2_popular_alignment"] = analysis_v2_popular_alignment(votes, models_data)

    print("V3: Convergent validity...")
    output["v3_convergent_validity"] = analysis_v3_convergent_validity(votes, models_data, smartvote_data)

    print("V4: Cross-linguistic consistency...")
    output["v4_cross_linguistic"] = analysis_v4_cross_linguistic(votes, models_data)

    print("V4b: Romansh low-resource stress test...")
    output["v4b_romansh"] = analysis_v4b_romansh(votes, models_data)

    print("V5: Röstigraben correlation...")
    output["v5_roestigraben"] = analysis_v5_roestigraben(votes, models_data)

    print("V6: Stimmfreigabe / false certainty...")
    output["v6_stimmfreigabe"] = analysis_v6_stimmfreigabe(votes, models_data)

    print("V7: Prompt sensitivity...")
    output["v7_prompt_sensitivity"] = analysis_v7_prompt_sensitivity(votes, models_data)

    print("V8: Model convergence...")
    output["v8_convergence"] = analysis_v8_convergence(votes, models_data)

    print("V9: Bundesrat agreement...")
    output["v9_bundesrat"] = analysis_v9_bundesrat(votes, models_data)

    print("V10: Refusal rates by language...")
    output["v10_refusal_by_language"] = analysis_v10_refusal_by_language(votes, all_models_data)

    print("V11: Party agreement by language...")
    output["v11_party_agreement_by_language"] = analysis_v11_party_agreement_by_language(votes, models_data)

    print("V12: Per-vote model consensus...")
    output["v12_consensus"] = analysis_v12_consensus(votes, models_data)

    print("V13: Temporal analysis...")
    output["v13_temporal"] = analysis_v13_temporal(votes, models_data)

    # BH correction: separate hypothesis families
    mcnemar_p_values: dict[str, float] = {}
    for model_name, v4_result in output["v4_cross_linguistic"].items():
        for pair_key, pair_data in v4_result.get("language_pairs", {}).items():
            if pair_data.get("mcnemar_p") is not None and pair_data["mcnemar_p"] < 1.0:
                mcnemar_p_values[f"mcnemar_{model_name}_{pair_key}"] = pair_data["mcnemar_p"]

    v5 = output["v5_roestigraben"]
    output["bh_correction"] = {
        "roestigraben": {
            "raw_p": v5.get("spearman_p"),
            "note": "single test, no correction needed",
        },
        "mcnemar_tests": {
            "raw_p_values": mcnemar_p_values,
            "adjusted_p_values": benjamini_hochberg(mcnemar_p_values),
            "note": "BH-corrected within the McNemar test family",
        },
    }

    # Write individual paper files
    print("\nWriting paper files...")

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_parolen_agreement.json",
        output["v1_parolen_agreement"],
        description="Agreement between each LLM's Ja/Nein votes and the party Parolen "
                    "(voting recommendations) on 48 federal referenda.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Binary agreement (Ja/Nein match) between model vote and party Parole, "
                    "excluding votes where the party issued Stimmfreigabe or keine Angabe. "
                    "Primary condition: in_kuerze (brief summary), German language.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_popular_alignment.json",
        output["v2_popular_alignment"],
        description="Alignment of each LLM's vote with the actual popular vote outcome, "
                    "broken down by margin of victory (close vs. decisive).",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Binary match between model vote (Ja/Nein) and referendum outcome. "
                    "Votes classified as close (<55% majority) or decisive (>60%). "
                    "Chi-square test of homogeneity across models. Fisher exact test for "
                    "close-vs-decisive within each model.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_cross_linguistic.json",
        {
            "per_model": output["v4_cross_linguistic"],
            "romansh": output["v4b_romansh"],
        },
        description="Cross-linguistic consistency: how often each LLM gives the same Ja/Nein "
                    "answer across German, French, Italian, and Romansh versions of the same referendum.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="4-language consistency = fraction of 48 votes where the model gave the same "
                    "answer in all 4 languages. Pairwise McNemar tests (BH-corrected) for each "
                    "language pair. Romansh analysis: Ja-rate shift and party alignment change.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_roestigraben.json",
        output["v5_roestigraben"],
        description="Test of whether LLMs replicate the Roestigraben (German-French voting divide): "
                    "correlation between actual cantonal DE-FR voting gap and model DE-FR answer gap.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Per-vote: actual DE-FR gap = mean(German-speaking cantons Ja%) - mean(French-speaking "
                    "cantons Ja%). Model gap = binary (1 if DE=Ja, 0 if DE=Nein) minus same for FR. "
                    "Spearman correlation across 48 votes, per model and aggregate.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_prompt_sensitivity.json",
        output["v7_prompt_sensitivity"],
        description="Sensitivity of LLM votes to the amount of context provided: "
                    "in_kuerze (brief) vs. in_kuerze_im_detail (detailed) vs. full_text (complete).",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Pairwise consistency between detail conditions (German language). "
                    "Binomial test against 25% chance baseline.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_stimmfreigabe.json",
        output["v6_stimmfreigabe"],
        description="Analysis of model behavior on ambiguous votes (where 2+ major parties "
                    "issued Stimmfreigabe) vs. clear votes with unified party positions.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Comparison of model certainty (consistency across languages and conditions) "
                    "on high-ambiguity (2+ Stimmfreigabe), some-ambiguity (1), and clear (0) votes.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_bundesrat.json",
        output["v9_bundesrat"],
        description="Agreement between each LLM's vote and the Bundesrat (Federal Council) recommendation.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Binary match between model vote and Bundesrat Parole on 48 referenda.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_model_convergence.json",
        output["v8_convergence"],
        description="Test of whether LLMs are more similar to each other than political parties are, "
                    "measuring inter-model vs. inter-party pairwise agreement on referenda.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Mean pairwise binary agreement across all model pairs vs. all party pairs. "
                    "Mann-Whitney U test and permutation test (10,000 iterations).",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_refusal_by_language.json",
        output["v10_refusal_by_language"],
        description="Refusal rates on Volksabstimmungen queries by language, with focus on "
                    "Gemini's language-dependent and context-dependent refusal patterns.",
        models_analyzed=VOLKS_MODEL_DISPLAY + ["Gemini 3.1 Pro"],
        models_excluded=[],
        n_models=9,
        methodology="Count of non-Ja/Nein responses per language and detail condition. "
                    "Chi-square tests for language effect and context effect on Gemini refusal rate.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_consensus.json",
        output["v12_consensus"],
        description="Per-vote consensus among LLMs: which referenda saw unanimous model agreement "
                    "and which were contentious.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Count of models voting Ja vs. Nein per referendum. "
                    "Unanimous = all 8 agree. Split = 4/4.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_convergent_validity.json",
        output["v3_convergent_validity"],
        description="Convergent validity: correlation between Smartvote agreement scores and "
                    "Volksabstimmungen Parolen agreement for the same model-party pairs.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Spearman correlation between Smartvote agreement % and Volksabstimmungen "
                    "agreement % across model-party pairs. Per-model, per-party, and pooled.",
    )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_party_agreement_by_language.json",
        output["v11_party_agreement_by_language"],
        description="Party Parolen agreement broken down by query language (de/fr/it/rm), "
                    "showing how language shifts affect political alignment patterns.",
        models_analyzed=VOLKS_MODEL_DISPLAY,
        models_excluded=VOLKS_MODELS_EXCLUDED,
        n_models=8,
        methodology="Binary agreement with party Parolen computed separately for each language condition.",
    )

    if "v13_temporal" in output:
        write_paper_file(
            f"{PAPER_DIR}/volksabstimmungen_temporal.json",
            output["v13_temporal"],
            description="Temporal analysis of Volksabstimmungen data (exploratory).",
            models_analyzed=VOLKS_MODEL_DISPLAY,
            models_excluded=VOLKS_MODELS_EXCLUDED,
            n_models=8,
        )

    write_paper_file(
        f"{PAPER_DIR}/volksabstimmungen_bh_correction.json",
        output["bh_correction"],
        description="Benjamini-Hochberg correction for the family of Volksabstimmungen statistical tests, "
                    "including Roestigraben correlations and McNemar pairwise language tests.",
        n_models=8,
        methodology="BH procedure at alpha=0.05 across all Volksabstimmungen p-values.",
    )

    # ── Print Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nV1: Parolen Agreement (German, In Kürze) [{len(models_data)} models]:")
    for model_name, data in output["v1_parolen_agreement"].items():
        agreements = [f"{p}: {data['parties'][p]['agreement']}%" for p in MAIN_PARTIES if data["parties"][p].get("agreement") is not None]
        print(f"  {data['display']:25s} {', '.join(agreements)}")

    print("\nV2: Popular Vote Alignment:")
    for model_name, data in output["v2_popular_alignment"].items():
        if model_name.startswith("_"):
            continue
        all_align = data["buckets"]["all"]
        left_a = data.get("directional_left_position", {}).get("left_agreement_pct", "?")
        right_a = data.get("directional_right_position", {}).get("left_agreement_pct", "?")
        print(f"  {data['display']:25s} Overall: {all_align['alignment_pct']}%  Left-pos: {left_a}%  Right-pos: {right_a}%")
    print(f"  Vote coding: {output['v2_popular_alignment']['_directional_counts']}")

    v3 = output["v3_convergent_validity"]
    if v3.get("pooled", {}).get("spearman_rho") is not None:
        p = v3["pooled"]
        print(f"\nV3: Convergent validity (pooled): ρ={p['spearman_rho']}, p={p['p_value']} (n={p['n_datapoints']})")

    print(f"\nV5: Röstigraben: ρ={v5.get('spearman_rho')}, p={v5.get('spearman_p')} (n={v5.get('n_votes')})")
    sens = v5.get("sensitivity_bilingual", {})
    if sens:
        print(f"  Sensitivity (incl. bilingual cantons): ρ={sens.get('spearman_rho')}, p={sens.get('spearman_p')}")

    v8 = output["v8_convergence"]
    print(f"\nV8: Convergence: model pairwise agreement={v8['model_mean_pairwise_agreement']}% "
          f"(σ={v8['model_std_pairwise_agreement']}%) vs "
          f"party={v8['party_mean_pairwise_agreement']}% (σ={v8['party_std_pairwise_agreement']}%)")
    print(f"  → {v8['interpretation']}")

    print("\nV9: Bundesrat agreement:")
    for name, data in output["v9_bundesrat"].items():
        print(f"  {data['display']:25s} {data['agreement_pct']}% ({data['n']} votes)")

    v12 = output["v12_consensus"]
    print(f"\nV12: Consensus: {v12['n_unanimous']} unanimous, {v12['n_split']} split, "
          f"mean consensus={v12['mean_consensus_pct']}%")

    v13 = output["v13_temporal"]
    print("\nV13: Temporal (early 2021-2022 vs late 2024-2026):")
    for name, data in v13.items():
        if name.startswith("_"):
            continue
        print(f"  {data['display']:25s} early={data['early_alignment_pct']}% late={data['late_alignment_pct']}%")

    close_detail = output["v2_popular_alignment"].get("_close_votes_detail", [])
    if close_detail:
        print(f"\nClose votes (≤5% margin): {len(close_detail)} votes")
        for cv in close_detail[:5]:
            print(f"  {cv['date']} {cv['titel'][:40]:40s} "
                  f"Ja={cv['ja_pct']}% → {cv['models_agree']} agree, {cv['models_disagree']} disagree")


if __name__ == "__main__":
    main()
