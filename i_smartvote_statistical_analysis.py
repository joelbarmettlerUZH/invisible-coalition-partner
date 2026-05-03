#!/usr/bin/env python3
"""
Statistical analysis for the LLM Political Bias paper.

Usage:
    uv run python i_smartvote_statistical_analysis.py
"""
import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ─── Configuration ───────────────────────────────────────────────────────────

QUESTIONNAIRE_FILE: str = "./data/questionnaire/questionnaire.json"
POLITICIANS_FILE: str = "./data/answers/nationalrat_members.json"
ALL_MODELS_FILE: str = "./data/answers/all_model_answers.json"
PCA_FILE: str = "./results/website/smartvote_pca_2d.json"
PAPER_DIR: str = "./results/paper"

PARTY_ORDER: list[str] = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]
PARTY_LR_INDEX: dict[str, int] = {p: i for i, p in enumerate(PARTY_ORDER)}

EXCLUDE_MODELS: set[str] = {
    "google/gemini-3.1-pro-preview",  # 84% refusal on Smartvote
    "qwen/qvq-72b-preview",           # 54% refusal, vision model
    "openai/o1-preview",               # 33% refusal
    "google/gemini-3-flash-preview",   # 25% refusal
    "x-ai/grok-4.20",                  # duplicate of grok-4.20-beta
}

N_BOOTSTRAP: int = 10000
N_PERMUTATIONS: int = 10000
RANDOM_SEED: int = 42

DEFAULT_ANSWER: int = 50


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_paper_file(
    path: str,
    results: dict,
    *,
    description: str,
    generated_by: str = "i_smartvote_statistical_analysis.py",
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


def extract_valid_question_ids(questionnaire: dict) -> list[str]:
    valid = []
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        cat_sort = cat.get("sortorder", 0)
        for q in cat.get("questions", []):
            valid.append((cat_sort, q.get("sortorder", 0), q["id"]))
    valid.sort()
    return [qid for _, _, qid in valid]


def extract_category_question_ids(questionnaire: dict) -> dict[str, list[str]]:
    categories = {}
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        cat_name = cat["name"]
        qids = [q["id"] for q in cat.get("questions", [])]
        categories[cat_name] = qids
    return categories


def build_answer_vector(answers_list: list[dict], question_ids: list[str], default: int = DEFAULT_ANSWER) -> np.ndarray:
    lookup = {}
    for a in answers_list:
        val = a.get("value", -1)
        if val != -1:
            lookup[a["questionId"]] = val
    return np.array([lookup.get(qid, default) for qid in question_ids], dtype=float)


def load_all_data() -> tuple[list[str], list[dict], list[dict]]:
    questionnaire = load_json(QUESTIONNAIRE_FILE)
    question_ids = extract_valid_question_ids(questionnaire)
    politicians_raw = load_json(POLITICIANS_FILE)

    # Build politician data
    politicians = []
    for pol in politicians_raw:
        party = pol.get("partyAbbreviation")
        if party not in PARTY_ORDER:
            continue
        vec = build_answer_vector(pol.get("answers", []), question_ids)
        politicians.append({
            "name": f"{pol.get('firstname', '')} {pol.get('lastname', '')}".strip(),
            "party": party,
            "vector": vec,
            "_raw_answers": pol.get("answers", []),
        })

    # Build model data from consolidated file
    models = []
    raw_models = load_json(ALL_MODELS_FILE)
    question_ids_set = set(question_ids)
    for m in raw_models:
        if m["name"] in EXCLUDE_MODELS:
            continue
        answers = m.get("answers", [])
        valid_count = sum(1 for a in answers if a.get("value", -1) != -1)
        if valid_count < 10:  # skip models with almost no valid answers
            continue
        vec = build_answer_vector(answers, question_ids)
        valid_pca = sum(1 for a in answers if a.get("value", -1) != -1 and a["questionId"] in question_ids_set)
        models.append({
            "name": m["name"],
            "display": m.get("display", m["name"]),
            "batch": m.get("batch", "unknown"),
            "country": m.get("country"),
            "continent": m.get("continent"),
            "open_source": m.get("open_source"),
            "reasoning": m.get("reasoning"),
            "family": m.get("family"),
            "predecessor": m.get("predecessor"),
            "released": m.get("released"),
            "provider": m.get("provider"),
            "vector": vec,
            "valid_answers": valid_count,
            "valid_pca_answers": valid_pca,
            "total_questions": len(question_ids),
            "coverage": valid_pca / len(question_ids),
            "_raw_answers": answers,
        })

    return question_ids, politicians, models


# ─── PCA ─────────────────────────────────────────────────────────────────────

def fit_pca_on_politicians(politicians: list[dict], n_components: int = 2) -> tuple[PCA, np.ndarray]:
    X = np.vstack([p["vector"] for p in politicians])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    return pca, coords


def project_into_pca(pca: PCA, vectors: list[np.ndarray]) -> np.ndarray:
    X = np.vstack(vectors)
    return pca.transform(X)


# ─── Analysis 1: PCA Validation ─────────────────────────────────────────────

def analyze_pca_validity(pca: PCA, pol_coords: np.ndarray, politicians: list[dict]) -> dict:
    # Explained variance
    ev_ratio = pca.explained_variance_ratio_
    ev_cumulative = np.cumsum(ev_ratio)

    # Silhouette score: how well do parties separate in 2D PCA space?
    party_labels = [p["party"] for p in politicians]
    sil_score = silhouette_score(pol_coords, party_labels)

    # Per-party centroids and within-party std
    party_stats = {}
    for party in PARTY_ORDER:
        mask = [p["party"] == party for p in politicians]
        party_coords = pol_coords[mask]
        if len(party_coords) == 0:
            continue
        centroid = party_coords.mean(axis=0)
        std = party_coords.std(axis=0)
        party_stats[party] = {
            "centroid": centroid.tolist(),
            "std": std.tolist(),
            "n": int(len(party_coords)),
        }

    # Verify left-right ordering on PC1
    # If parties are correctly ordered, PC1 of centroids should be monotonic
    centroids_pc1 = [party_stats[p]["centroid"][0] for p in PARTY_ORDER if p in party_stats]
    # Check if monotonically increasing or decreasing
    diffs = np.diff(centroids_pc1)
    is_monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)
    # Spearman rank correlation with expected ordering
    from scipy.stats import spearmanr
    expected_order = list(range(len(centroids_pc1)))
    rho, p_val = spearmanr(centroids_pc1, expected_order)

    return {
        "explained_variance_ratio": ev_ratio.tolist(),
        "explained_variance_cumulative": ev_cumulative.tolist(),
        "total_explained_variance_2d": float(ev_cumulative[1]) if len(ev_cumulative) > 1 else float(ev_cumulative[0]),
        "silhouette_score": float(sil_score),
        "silhouette_interpretation": (
            "excellent" if sil_score > 0.5 else
            "good" if sil_score > 0.3 else
            "fair" if sil_score > 0.1 else
            "poor"
        ),
        "party_stats": party_stats,
        "pc1_left_right_ordering": {
            "is_monotonic": bool(is_monotonic),
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "interpretation": "PC1 captures left-right spectrum" if abs(rho) > 0.8 and p_val < 0.05 else "PC1 does not cleanly capture left-right",
        },
    }


# ─── Analysis 2: Progressive Bias Test ──────────────────────────────────────

def analyze_progressive_bias(pca: PCA, politicians: list[dict], models: list[dict], batch: str | None = None) -> dict:
    rng = np.random.RandomState(RANDOM_SEED)

    pol_coords = project_into_pca(pca, [p["vector"] for p in politicians])
    pol_centroid = pol_coords.mean(axis=0)

    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch]
    else:
        batch_models = list(models)
    if not batch_models:
        return {"error": f"No models found for batch {batch}"}

    model_coords = project_into_pca(pca, [m["vector"] for m in batch_models])
    model_centroid = model_coords.mean(axis=0)

    # Observed displacement
    displacement = model_centroid - pol_centroid
    displacement_magnitude = float(np.linalg.norm(displacement))

    # Permutation test: randomly partition all points into "model" and "politician"
    # groups and compute displacement. How often is random displacement >= observed?
    n_models = len(batch_models)
    all_coords = np.vstack([pol_coords, model_coords])
    observed_dist = displacement_magnitude

    count_ge = 0
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(len(all_coords))
        fake_model_coords = all_coords[perm[:n_models]]
        fake_pol_coords = all_coords[perm[n_models:]]
        fake_model_centroid = fake_model_coords.mean(axis=0)
        fake_pol_centroid = fake_pol_coords.mean(axis=0)
        fake_displacement = np.linalg.norm(fake_model_centroid - fake_pol_centroid)
        if fake_displacement >= observed_dist:
            count_ge += 1

    p_value = (count_ge + 1) / (N_PERMUTATIONS + 1)

    # Bootstrap CI on model centroid
    boot_centroids = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(model_coords), size=len(model_coords), replace=True)
        boot_centroids.append(model_coords[idx].mean(axis=0))
    boot_centroids = np.array(boot_centroids)
    ci_lower = np.percentile(boot_centroids, 2.5, axis=0)
    ci_upper = np.percentile(boot_centroids, 97.5, axis=0)

    # Which party centroid is closest to the model centroid?
    party_centroids = {}
    for party in PARTY_ORDER:
        mask = [p["party"] == party for p in politicians]
        pc = pol_coords[mask].mean(axis=0)
        party_centroids[party] = pc

    distances_to_parties = {
        party: float(np.linalg.norm(model_centroid - pc))
        for party, pc in party_centroids.items()
    }
    closest_party = min(distances_to_parties, key=distances_to_parties.get)

    # Per-model positions
    model_positions = []
    for m, coord in zip(batch_models, model_coords):
        dists = {party: float(np.linalg.norm(coord - party_centroids[party]))
                 for party in PARTY_ORDER}
        model_positions.append({
            "name": m["name"],
            "display": m["display"],
            "country": m.get("country"),
            "continent": m.get("continent"),
            "family": m.get("family"),
            "open_source": m.get("open_source"),
            "batch": m.get("batch"),
            "released": m.get("released"),
            "coords": coord.tolist(),
            "closest_party": min(dists, key=dists.get),
            "distances_to_parties": dists,
        })

    return {
        "batch": batch,
        "n_models": n_models,
        "politician_centroid": pol_centroid.tolist(),
        "model_centroid": model_centroid.tolist(),
        "displacement_vector": displacement.tolist(),
        "displacement_magnitude": displacement_magnitude,
        "permutation_test": {
            "p_value": float(p_value),
            "n_permutations": N_PERMUTATIONS,
            "significant_at_005": p_value < 0.05,
            "significant_at_001": p_value < 0.01,
        },
        "model_centroid_95ci": {
            "lower": ci_lower.tolist(),
            "upper": ci_upper.tolist(),
        },
        "closest_party_to_mean_llm": closest_party,
        "distances_to_party_centroids": distances_to_parties,
        "all_models_left_of_center": bool(all(c[0] < pol_centroid[0] for c in model_coords))
            if pol_centroid[0] > 0 else bool(all(c[0] > pol_centroid[0] for c in model_coords)),
        "model_positions": model_positions,
    }


# ─── Analysis 3: Agreement Scores with CIs ──────────────────────────────────

def analyze_agreement_scores(politicians: list[dict], models: list[dict], question_ids: list[str], batch: str | None = None) -> dict:
    rng = np.random.RandomState(RANDOM_SEED)

    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch]
    else:
        batch_models = list(models)
    if not batch_models:
        return {"error": f"No models found for batch {batch}"}

    # Group politicians by party
    party_pols = defaultdict(list)
    for p in politicians:
        party_pols[p["party"]].append(p["vector"])

    results = {}
    for model in batch_models:
        model_vec = model["vector"]
        model_results = {}

        for party in PARTY_ORDER:
            pol_vecs = party_pols[party]
            if not pol_vecs:
                continue

            # Compute agreement: 100 * (1 - mean_squared_diff / 10000)
            diffs = []
            for pv in pol_vecs:
                sq_diff = (pv - model_vec) ** 2
                diffs.append(sq_diff.mean())
            diffs = np.array(diffs)
            agreement = 100 * (1 - diffs.mean() / 10000)

            # Bootstrap CI on the agreement
            boot_agreements = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(len(diffs), size=len(diffs), replace=True)
                boot_mean = diffs[idx].mean()
                boot_agreements.append(100 * (1 - boot_mean / 10000))
            boot_agreements = np.array(boot_agreements)

            model_results[party] = {
                "agreement": float(np.clip(agreement, 0, 100)),
                "ci_lower": float(np.percentile(boot_agreements, 2.5)),
                "ci_upper": float(np.percentile(boot_agreements, 97.5)),
                "n_politicians": len(pol_vecs),
            }

        results[model["name"]] = {
            "display": model["display"],
            "agreements": model_results,
        }

    return {"batch": batch, "models": results}


# ─── Analysis 3b: Agreement Robustness (Absolute vs Squared Difference) ─────

def analyze_agreement_robustness(politicians: list[dict], models: list[dict], question_ids: list[str], batch: str | None = None) -> dict:
    """Test whether the left-right agreement gradient survives under absolute differences.

    The squared-difference metric (used throughout) penalizes extreme disagreements
    quadratically, which may inflate agreement with centrist parties. This robustness
    check computes agreement using absolute differences and checks whether the
    party-position gradient is preserved.
    """
    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch]
    else:
        batch_models = list(models)
    if not batch_models:
        return {"error": f"No models found for batch {batch}"}

    party_pols = defaultdict(list)
    for p in politicians:
        party_pols[p["party"]].append(p["vector"])

    party_positions = list(range(1, len(PARTY_ORDER) + 1))

    per_model: dict = {}
    gradient_preserved_count = 0
    sq_rhos: list[float] = []
    abs_rhos: list[float] = []

    for model in batch_models:
        mv = model["vector"]
        sq_agreements: list[float] = []
        abs_agreements: list[float] = []

        for party in PARTY_ORDER:
            pvecs = party_pols[party]
            if not pvecs:
                sq_agreements.append(float("nan"))
                abs_agreements.append(float("nan"))
                continue

            # Squared difference agreement (existing metric)
            sq_diffs = [float(np.mean((pv - mv) ** 2)) for pv in pvecs]
            sq_agree = 100.0 * (1.0 - np.mean(sq_diffs) / 10000.0)

            # Absolute difference agreement
            abs_diffs = [float(np.mean(np.abs(pv - mv))) for pv in pvecs]
            abs_agree = 100.0 * (1.0 - np.mean(abs_diffs) / 100.0)

            sq_agreements.append(float(np.clip(sq_agree, 0, 100)))
            abs_agreements.append(float(np.clip(abs_agree, 0, 100)))

        # Spearman correlation with party position
        sq_rho, sq_p = sp_stats.spearmanr(party_positions, sq_agreements)
        abs_rho, abs_p = sp_stats.spearmanr(party_positions, abs_agreements)

        same_sign = (sq_rho < 0 and abs_rho < 0) or (sq_rho > 0 and abs_rho > 0) or (sq_rho == 0 and abs_rho == 0)
        if same_sign:
            gradient_preserved_count += 1

        sq_rhos.append(sq_rho)
        abs_rhos.append(abs_rho)

        per_model[model["name"]] = {
            "display": model["display"],
            "squared_agreements": {p: round(a, 1) for p, a in zip(PARTY_ORDER, sq_agreements)},
            "absolute_agreements": {p: round(a, 1) for p, a in zip(PARTY_ORDER, abs_agreements)},
            "squared_gradient_rho": round(float(sq_rho), 4),
            "squared_gradient_p": round(float(sq_p), 4),
            "absolute_gradient_rho": round(float(abs_rho), 4),
            "absolute_gradient_p": round(float(abs_p), 4),
            "gradient_direction_preserved": same_sign,
        }

    # Cross-metric correlation: how similar are the two sets of agreement scores?
    all_sq = []
    all_abs = []
    for m in per_model.values():
        for p in PARTY_ORDER:
            all_sq.append(m["squared_agreements"][p])
            all_abs.append(m["absolute_agreements"][p])
    cross_rho, cross_p = sp_stats.spearmanr(all_sq, all_abs)

    return {
        "batch": batch,
        "n_models": len(batch_models),
        "gradient_preserved_count": gradient_preserved_count,
        "gradient_preserved_fraction": round(gradient_preserved_count / len(batch_models), 3),
        "mean_squared_gradient_rho": round(float(np.mean(sq_rhos)), 4),
        "mean_absolute_gradient_rho": round(float(np.mean(abs_rhos)), 4),
        "cross_metric_spearman": {
            "rho": round(float(cross_rho), 4),
            "p_value": round(float(cross_p), 6),
            "n_datapoints": len(all_sq),
            "note": "Correlation between squared-difference and absolute-difference "
                    "agreement scores across all model-party pairs.",
        },
        "per_model": per_model,
    }


# ─── Analysis 4: Geographic Effect ──────────────────────────────────────────

def analyze_geographic_effect(pca: PCA, models: list[dict], batch: str | None = None) -> dict:
    rng = np.random.RandomState(RANDOM_SEED)

    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch and m.get("country")]
    else:
        batch_models = [m for m in models if m.get("country")]
    if len(batch_models) < 4:
        return {"error": "Not enough models with country info"}

    coords = project_into_pca(pca, [m["vector"] for m in batch_models])
    countries = [m["country"] for m in batch_models]
    continents = [m["continent"] for m in batch_models]

    def permutation_anova_pc1(coords_pc1, labels, n_perm=N_PERMUTATIONS):
        """Permutation test for group differences on PC1."""
        groups = defaultdict(list)
        for c, label in zip(coords_pc1, labels):
            groups[label].append(c)

        # Only test if we have at least 2 groups with 2+ members
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(valid_groups) < 2:
            return {"error": "Not enough groups with 2+ members", "groups": {k: len(v) for k, v in groups.items()}}

        # Observed F-statistic (between-group variance / within-group variance)
        grand_mean = np.mean(coords_pc1)
        ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in valid_groups.values())
        ss_within = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in valid_groups.values())
        k = len(valid_groups)
        n = sum(len(v) for v in valid_groups.values())
        if ss_within == 0:
            return {"error": "Zero within-group variance"}
        observed_f = (ss_between / (k - 1)) / (ss_within / (n - k))

        # Permutation
        all_vals = np.array(coords_pc1)
        count_ge = 0
        for _ in range(n_perm):
            perm_labels = rng.permutation(labels)
            perm_groups = defaultdict(list)
            for c, label in zip(all_vals, perm_labels):
                perm_groups[label].append(c)
            perm_valid = {k: v for k, v in perm_groups.items() if len(v) >= 2}
            if len(perm_valid) < 2:
                continue
            perm_grand_mean = np.mean([v for vals in perm_valid.values() for v in vals])
            perm_ss_between = sum(len(v) * (np.mean(v) - perm_grand_mean) ** 2 for v in perm_valid.values())
            perm_ss_within = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in perm_valid.values())
            if perm_ss_within == 0:
                continue
            perm_f = (perm_ss_between / (k - 1)) / (perm_ss_within / (n - k))
            if perm_f >= observed_f:
                count_ge += 1

        p_value = (count_ge + 1) / (n_perm + 1)

        group_means = {k: {"mean_pc1": float(np.mean(v)), "std_pc1": float(np.std(v)), "n": len(v)}
                       for k, v in groups.items()}

        return {
            "f_statistic": float(observed_f),
            "p_value": float(p_value),
            "significant_at_005": p_value < 0.05,
            "group_stats": group_means,
        }

    pc1 = coords[:, 0]

    return {
        "batch": batch,
        "by_country": permutation_anova_pc1(pc1, countries),
        "by_continent": permutation_anova_pc1(pc1, continents),
    }


# ─── Analysis 5: Open-source vs Closed-source ───────────────────────────────

def analyze_open_vs_closed(pca: PCA, models: list[dict], batch: str | None = None) -> dict:
    rng = np.random.RandomState(RANDOM_SEED)

    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch and m.get("open_source") is not None]
    else:
        batch_models = [m for m in models if m.get("open_source") is not None]
    if len(batch_models) < 4:
        return {"error": "Not enough models"}

    coords = project_into_pca(pca, [m["vector"] for m in batch_models])
    labels = ["open" if m["open_source"] else "closed" for m in batch_models]

    open_coords = coords[[lbl == "open" for lbl in labels]]
    closed_coords = coords[[lbl == "closed" for lbl in labels]]

    if len(open_coords) < 2 or len(closed_coords) < 2:
        return {"error": "Not enough models in each group"}

    # Observed mean difference on PC1
    observed_diff = float(open_coords[:, 0].mean() - closed_coords[:, 0].mean())

    # Permutation test
    all_pc1 = coords[:, 0]
    n_open = len(open_coords)
    count_ge = 0
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(len(all_pc1))
        perm_open = all_pc1[perm[:n_open]]
        perm_closed = all_pc1[perm[n_open:]]
        perm_diff = perm_open.mean() - perm_closed.mean()
        if abs(perm_diff) >= abs(observed_diff):
            count_ge += 1

    p_value = (count_ge + 1) / (N_PERMUTATIONS + 1)

    return {
        "batch": batch,
        "open_source": {
            "n": int(len(open_coords)),
            "mean_pc1": float(open_coords[:, 0].mean()),
            "std_pc1": float(open_coords[:, 0].std()),
            "models": [m["display"] for m, lbl in zip(batch_models, labels) if lbl == "open"],
        },
        "closed_source": {
            "n": int(len(closed_coords)),
            "mean_pc1": float(closed_coords[:, 0].mean()),
            "std_pc1": float(closed_coords[:, 0].std()),
            "models": [m["display"] for m, lbl in zip(batch_models, labels) if lbl == "closed"],
        },
        "mean_difference_pc1": observed_diff,
        "permutation_test": {
            "p_value": float(p_value),
            "n_permutations": N_PERMUTATIONS,
            "significant_at_005": p_value < 0.05,
        },
    }


# ─── Analysis 6: Temporal Drift ─────────────────────────────────────────────

def analyze_drift(pca: PCA, models: list[dict]) -> dict:
    rng = np.random.RandomState(RANDOM_SEED)

    # Find pairs: 2026 models with predecessors in 2025
    model_map = {m["name"]: m for m in models}
    pairs = []
    for m in models:
        if m["batch"] == "2026" and m.get("predecessor"):
            pred = model_map.get(m["predecessor"])
            if pred:
                pairs.append((pred, m))

    if not pairs:
        return {"error": "No predecessor pairs found"}

    drift_results = []
    for old, new in pairs:
        old_coord = project_into_pca(pca, [old["vector"]])[0]
        new_coord = project_into_pca(pca, [new["vector"]])[0]
        drift_vec = new_coord - old_coord
        drift_mag = float(np.linalg.norm(drift_vec))

        # Bootstrap CI on drift magnitude via question resampling
        n_questions = len(old["vector"])
        boot_mags = []
        boot_drifts_pc1 = []
        for _ in range(N_BOOTSTRAP):
            # Sample questions with replacement and recompute PCA projection
            idx = rng.choice(n_questions, size=n_questions, replace=True)
            # Build resampled vectors and refit PCA projection on resampled questions
            boot_old_vec = old["vector"][idx]
            boot_new_vec = new["vector"][idx]
            # Project using resampled components of the PCA
            mean_sub = pca.mean_[idx]
            comp_sub = pca.components_[:, idx]
            boot_old_coord = np.dot(boot_old_vec - mean_sub, comp_sub.T)
            boot_new_coord = np.dot(boot_new_vec - mean_sub, comp_sub.T)
            boot_drift = boot_new_coord - boot_old_coord
            boot_mags.append(np.linalg.norm(boot_drift))
            boot_drifts_pc1.append(boot_drift[0])

        boot_mags = np.array(boot_mags)

        boot_drifts_pc1 = np.array(boot_drifts_pc1)
        drift_results.append({
            "old_model": old["display"],
            "new_model": new["display"],
            "old_name": old["name"],
            "new_name": new["name"],
            "old_coords": old_coord.tolist(),
            "new_coords": new_coord.tolist(),
            "drift_vector": drift_vec.tolist(),
            "drift_magnitude": drift_mag,
            "drift_ci_lower": float(np.percentile(boot_mags, 2.5)),
            "drift_ci_upper": float(np.percentile(boot_mags, 97.5)),
            "drift_pc1_ci_lower": float(np.percentile(boot_drifts_pc1, 2.5)),
            "drift_pc1_ci_upper": float(np.percentile(boot_drifts_pc1, 97.5)),
            "drift_direction_pc1": "left (more progressive)" if drift_vec[0] < 0 else "right (more conservative)",
        })

    # Aggregate: is there a systematic drift direction?
    drift_pc1 = [d["drift_vector"][0] for d in drift_results]
    mean_drift_pc1 = float(np.mean(drift_pc1))

    # Sign test: are more models moving left or right?
    n_left = sum(1 for d in drift_pc1 if d < 0)
    n_right = sum(1 for d in drift_pc1 if d > 0)
    from scipy.stats import binomtest
    sign_test = binomtest(n_left, n_left + n_right, 0.5, alternative='two-sided')

    return {
        "n_pairs": len(pairs),
        "pairs": drift_results,
        "aggregate": {
            "mean_drift_pc1": mean_drift_pc1,
            "mean_drift_direction": "left (more progressive)" if mean_drift_pc1 < 0 else "right (more conservative)",
            "n_moved_left": n_left,
            "n_moved_right": n_right,
            "sign_test_p_value": float(sign_test.pvalue),
            "systematic_drift": sign_test.pvalue < 0.05,
        },
    }


# ─── Analysis 7: Refusal Rate Analysis ──────────────────────────────────────

def analyze_refusal_rates(models: list[dict], batch: str | None = None) -> dict:
    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch]
    else:
        batch_models = list(models)

    refusal_data = []
    for m in batch_models:
        refusal_rate = 1.0 - m["coverage"]
        refusal_data.append({
            "name": m["name"],
            "display": m["display"],
            "provider": m.get("provider"),
            "refusal_rate": float(refusal_rate),
            "valid_answers": m["valid_pca_answers"],
            "total_questions": m["total_questions"],
            "is_moderated": m.get("is_moderated"),
        })

    # Sort by refusal rate
    refusal_data.sort(key=lambda x: x["refusal_rate"], reverse=True)

    # Summary stats
    rates = [d["refusal_rate"] for d in refusal_data]

    return {
        "batch": batch,
        "models": refusal_data,
        "summary": {
            "mean_refusal_rate": float(np.mean(rates)),
            "median_refusal_rate": float(np.median(rates)),
            "max_refusal_rate": float(np.max(rates)),
            "models_with_zero_refusals": sum(1 for r in rates if r == 0),
            "models_with_any_refusals": sum(1 for r in rates if r > 0),
        },
    }


# ─── Analysis 8: Deep Time-Series ───────────────────────────────────────────

def analyze_deep_timeseries(pca: PCA, models: list[dict]) -> dict:
    from scipy.stats import linregress

    # Group all models by family
    families = defaultdict(list)
    for m in models:
        if m.get("family") and m.get("released"):
            families[m["family"]].append(m)

    # Only analyze families with 3+ models (enough for regression)
    results = {}
    for family, members in families.items():
        if len(members) < 3:
            continue

        # Sort by release date
        members.sort(key=lambda m: m["released"])

        # Project into PCA
        coords = project_into_pca(pca, [m["vector"] for m in members])

        # Convert dates to numeric (days since first release)
        dates = [datetime.strptime(m["released"], "%Y-%m-%d") for m in members]
        days = [(d - dates[0]).days for d in dates]

        # Linear regression on PC1 over time
        pc1 = coords[:, 0]
        if len(set(days)) < 2:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(days, pc1)

        # Build timeline
        timeline = []
        for m, coord, date in zip(members, coords, dates):
            timeline.append({
                "name": m["name"],
                "display": m["display"],
                "released": m["released"],
                "coords": coord.tolist(),
                "pc1": float(coord[0]),
            })

        results[family] = {
            "display": members[0].get("provider", family),
            "n_models": len(members),
            "timeline": timeline,
            "trend": {
                "slope_per_year": float(slope * 365),  # PC1 units per year
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "significant_at_005": p_value < 0.05,
                "direction": "trending progressive" if slope < 0 else "trending conservative",
                "interpretation": (
                    f"{'Significant' if p_value < 0.05 else 'Non-significant'} "
                    f"{'leftward' if slope < 0 else 'rightward'} trend "
                    f"(R²={r_value**2:.3f}, p={p_value:.3f})"
                ),
            },
        }

    return results


# ─── Analysis 9: Imputation Sensitivity ───────────────────────────────────────

def analyze_imputation_sensitivity(politicians: list[dict], models: list[dict], question_ids: list[str]) -> dict:
    imputation_values = [0, 25, 50, 75, 100]
    results = {}
    for imp_val in imputation_values:
        # Rebuild model vectors with different imputation
        alt_models = []
        for m in models:
            vec = build_answer_vector(m.get("_raw_answers", []), question_ids, default=imp_val)
            alt_models.append({**m, "vector": vec})
        alt_pols = []
        for p in politicians:
            vec = build_answer_vector(p.get("_raw_answers", []), question_ids, default=imp_val)
            alt_pols.append({**p, "vector": vec})

        pca_alt, _ = fit_pca_on_politicians(alt_pols)
        key = f"imputation_{imp_val}"
        result = analyze_progressive_bias(pca_alt, alt_pols, alt_models, batch=None)
        if "error" not in result:
            results[key] = {
                "imputation_value": imp_val,
                "displacement_magnitude": result["displacement_magnitude"],
                "p_value": result["permutation_test"]["p_value"],
                "significant_at_005": result["permutation_test"]["significant_at_005"],
            }
        else:
            results[key] = {"imputation_value": imp_val, "error": result["error"]}
    return results


# ─── Analysis 10: Per-Category Political Profiles ────────────────────────────

def analyze_category_profiles(questionnaire: dict, politicians: list[dict], models: list[dict], batch: str | None = None) -> dict:
    """Per-category 1D PCA with SP < SVP orientation ensures politically meaningful axes
    regardless of question framing.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    categories = extract_category_question_ids(questionnaire)

    if batch is not None:
        batch_models = [m for m in models if m["batch"] == batch]
    else:
        batch_models = list(models)

    # Group politicians by party
    party_pols = defaultdict(list)
    for p in politicians:
        party_pols[p["party"]].append(p)

    def get_answer_map(entity):
        m = {}
        for a in entity.get("_raw_answers", entity.get("answers", [])):
            qid, val = a.get("questionId"), a.get("value", -1)
            if val != -1:
                m[qid] = val
        return m

    def build_category_vector(entity, qids, default=DEFAULT_ANSWER):
        """Build answer vector for a subset of questions."""
        am = get_answer_map(entity)
        return np.array([am.get(qid, default) for qid in qids], dtype=float)

    results = {}
    for cat_name, qids in categories.items():
        n_questions = len(qids)

        # Build politician matrix for this category
        pol_vecs = np.array([build_category_vector(p, qids) for p in politicians])

        # Fit 1D PCA on politicians
        pca_cat = PCA(n_components=1)
        pol_pc1 = pca_cat.fit_transform(pol_vecs).flatten()
        explained_var = float(pca_cat.explained_variance_ratio_[0])

        # Orient so SP centroid < SVP centroid (left < right)
        sp_mask = np.array([p["party"] == "SP" for p in politicians])
        svp_mask = np.array([p["party"] == "SVP" for p in politicians])
        sp_centroid = pol_pc1[sp_mask].mean()
        svp_centroid = pol_pc1[svp_mask].mean()
        flip = 1.0
        if sp_centroid > svp_centroid:
            flip = -1.0
            pol_pc1 = -pol_pc1
            sp_centroid = -sp_centroid
            svp_centroid = -svp_centroid

        # Party centroids on PC1
        party_pc1 = {}
        for party in PARTY_ORDER:
            mask = np.array([p["party"] == party for p in politicians])
            if mask.any():
                party_pc1[party] = float(pol_pc1[mask].mean())

        # Parliamentary centroid and SD
        parl_centroid = float(pol_pc1.mean())
        parl_std = float(pol_pc1.std(ddof=1))

        # Project LLMs into category PCA space
        llm_vecs = np.array([build_category_vector(m, qids) for m in batch_models])
        llm_pc1 = (pca_cat.transform(llm_vecs).flatten()) * flip
        llm_centroid = float(llm_pc1.mean())
        llm_std = float(llm_pc1.std(ddof=1)) if len(llm_pc1) > 1 else 0.0

        # Displacement from parliamentary centroid
        displacement = llm_centroid - parl_centroid
        displacement_sd = displacement / parl_std if parl_std > 0 else 0.0

        # Closest party
        dists = {p: abs(llm_centroid - pc1) for p, pc1 in party_pc1.items()}
        closest_party = min(dists, key=dists.get) if dists else None

        # Permutation test: is LLM centroid significantly displaced from parliament?
        all_pc1 = np.concatenate([pol_pc1, llm_pc1])
        n_llm = len(llm_pc1)
        observed_diff = abs(llm_centroid - parl_centroid)
        count_ge = 0
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(len(all_pc1))
            perm_llm = all_pc1[perm[:n_llm]]
            perm_parl = all_pc1[perm[n_llm:]]
            perm_diff = abs(perm_llm.mean() - perm_parl.mean())
            if perm_diff >= observed_diff:
                count_ge += 1
        p_value = (count_ge + 1) / (N_PERMUTATIONS + 1)

        # Bootstrap CI on LLM centroid
        boot_centroids = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(llm_pc1), size=len(llm_pc1), replace=True)
            boot_centroids.append(float(llm_pc1[idx].mean()))
        ci_lower = float(np.percentile(boot_centroids, 2.5))
        ci_upper = float(np.percentile(boot_centroids, 97.5))

        results[cat_name] = {
            "n_questions": n_questions,
            "explained_variance_pc1": round(explained_var, 3),
            "party_pc1": {p: round(v, 1) for p, v in party_pc1.items()},
            "parliamentary_centroid": round(parl_centroid, 1),
            "parliamentary_std": round(parl_std, 1),
            "llm_centroid": round(llm_centroid, 1),
            "llm_std": round(llm_std, 1),
            "llm_ci_lower": round(ci_lower, 1),
            "llm_ci_upper": round(ci_upper, 1),
            "displacement": round(displacement, 1),
            "displacement_sd": round(displacement_sd, 2),
            "closest_party": closest_party,
            "permutation_p": round(p_value, 4),
            "significant_at_005": p_value < 0.05,
        }

    # Benjamini-Hochberg correction across categories
    raw_ps = {cat: r["permutation_p"] for cat, r in results.items()}
    adj_ps = benjamini_hochberg(raw_ps)
    for cat in results:
        results[cat]["adjusted_p"] = round(adj_ps[cat], 4)
        results[cat]["significant_after_correction"] = adj_ps[cat] < 0.05

    return {
        "batch": batch,
        "n_models": len(batch_models),
        "n_categories": len(results),
        "categories": results,
    }


# ─── Effect Sizes ────────────────────────────────────────────────────────────

def compute_effect_sizes(results: dict) -> dict:
    effect_sizes = {}

    # Progressive bias: Cohen's d (LLM centroid vs politician centroid)
    pb = results.get("progressive_bias", {})
    if "error" not in pb and "model_positions" in pb:
        model_pc1 = [m["coords"][0] for m in pb["model_positions"]]
        if model_pc1:
            mean_diff = np.mean(model_pc1) - pb["politician_centroid"][0]
            sd = np.std(model_pc1, ddof=1) if len(model_pc1) > 1 else 1.0
            effect_sizes["progressive_bias_cohens_d"] = float(mean_diff / sd) if sd > 0 else float("inf")

    # Open vs closed: Cohen's d
    oc = results.get("open_vs_closed", {})
    if "error" not in oc:
        n1 = oc["open_source"]["n"]
        n2 = oc["closed_source"]["n"]
        s1 = oc["open_source"]["std_pc1"]
        s2 = oc["closed_source"]["std_pc1"]
        # Pooled SD
        if n1 > 1 and n2 > 1:
            pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            if pooled_sd > 0:
                effect_sizes["open_vs_closed_cohens_d"] = float(oc["mean_difference_pc1"] / pooled_sd)

    # Geographic ANOVA: eta-squared
    ge = results.get("geographic_effect", {})
    for geo_key in ["by_country", "by_continent"]:
        geo = ge.get(geo_key, {})
        if "error" not in geo and "f_statistic" in geo:
            # Reconstruct eta-squared from group stats
            group_stats = geo.get("group_stats", {})
            if group_stats:
                all_means = [g["mean_pc1"] for g in group_stats.values()]
                all_ns = [g["n"] for g in group_stats.values()]
                grand_mean = np.average(all_means, weights=all_ns)
                ss_between = sum(n * (m - grand_mean)**2 for m, n in zip(all_means, all_ns))
                ss_within = sum(g["n"] * g["std_pc1"]**2 for g in group_stats.values())
                ss_total = ss_between + ss_within
                if ss_total > 0:
                    effect_sizes[f"geographic_{geo_key}_eta_squared"] = float(ss_between / ss_total)

    # Drift: mean magnitude ± SD
    dr = results.get("drift", {})
    if "error" not in dr and "pairs" in dr:
        mags = [p["drift_magnitude"] for p in dr["pairs"]]
        effect_sizes["drift_mean_magnitude"] = float(np.mean(mags))
        effect_sizes["drift_std_magnitude"] = float(np.std(mags, ddof=1)) if len(mags) > 1 else 0.0
        # Individual drift magnitudes and directions
        effect_sizes["drift_individual"] = [
            {"family": f"{p['old_model']} → {p['new_model']}",
             "magnitude": p["drift_magnitude"],
             "direction_pc1": p["drift_direction_pc1"]}
            for p in dr["pairs"]
        ]

    return effect_sizes


def compute_sign_test_power(n_pairs: int, alpha: float = 0.05) -> dict:
    from scipy.stats import binom
    # Under H0: p=0.5, two-sided test
    # Critical value: smallest k such that P(X <= k | n, 0.5) <= alpha/2
    # Power: P(reject H0 | p=0.75)
    k_crit = binom.ppf(alpha / 2, n_pairs, 0.5)
    # Power under alternative p=0.75
    power_75 = float(binom.cdf(k_crit, n_pairs, 0.75) + 1 - binom.cdf(n_pairs - k_crit, n_pairs, 0.75))
    # Power under alternative p=0.65
    power_65 = float(binom.cdf(k_crit, n_pairs, 0.65) + 1 - binom.cdf(n_pairs - k_crit, n_pairs, 0.65))
    return {
        "n_pairs": n_pairs,
        "alpha": alpha,
        "power_at_p75": power_75,
        "power_at_p65": power_65,
        "interpretation": f"With n={n_pairs} pairs, the sign test has {power_75:.1%} power to detect a 75/25 split and {power_65:.1%} power to detect a 65/35 split at α={alpha}.",
    }


# ─── Multiple Comparison Correction ──────────────────────────────────────────

def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
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
    # Re-sort by original order and enforce monotonicity from smallest to largest
    sorted_by_p = sorted(p_values.items(), key=lambda x: x[1])
    result = {}
    running_max = 0.0
    for label, _ in sorted_by_p:
        adj = max(adjusted[label], running_max)
        result[label] = adj
        running_max = adj
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    question_ids, politicians, models = load_all_data()
    print(f"  {len(politicians)} politicians, {len(models)} models across all batches")

    # Deduplicate models (same name+batch)
    seen = set()
    unique_models = []
    for m in models:
        key = (m["name"], m["batch"])
        if key not in seen:
            seen.add(key)
            unique_models.append(m)
    models = unique_models
    print(f"  {len(models)} unique models after deduplication")

    print("Fitting PCA on politicians...")
    pca, pol_coords = fit_pca_on_politicians(politicians)

    results = {}

    print("1. PCA Validation...")
    results["pca_validation"] = analyze_pca_validity(pca, pol_coords, politicians)

    # All models (after exclusions) for main analyses
    cross_sectional = models  # no batch filtering — pool all models
    print(f"  {len(cross_sectional)} models (all batches pooled)")

    print("2. Progressive Bias Test (cross-sectional, pooled)...")
    results["progressive_bias"] = analyze_progressive_bias(pca, politicians, cross_sectional, batch=None)

    print("3. Agreement Scores with CIs...")
    results["agreement_scores"] = analyze_agreement_scores(politicians, cross_sectional, question_ids, batch=None)

    print("3b. Agreement Robustness (Absolute vs Squared Difference)...")
    results["agreement_robustness"] = analyze_agreement_robustness(politicians, cross_sectional, question_ids, batch=None)

    print("4. Geographic Effect...")
    results["geographic_effect"] = analyze_geographic_effect(pca, cross_sectional, batch=None)

    print("5. Open vs Closed Source...")
    results["open_vs_closed"] = analyze_open_vs_closed(pca, cross_sectional, batch=None)

    print("6. Temporal Drift...")
    results["drift"] = analyze_drift(pca, models)  # all models for drift (uses predecessor pairs)

    print("7. Refusal Rates...")
    results["refusal_rates"] = analyze_refusal_rates(cross_sectional, batch=None)

    print("8. Deep Time-Series...")
    results["deep_timeseries"] = analyze_deep_timeseries(pca, models)

    print("9. Imputation Sensitivity...")
    results["imputation_sensitivity"] = analyze_imputation_sensitivity(politicians, cross_sectional, question_ids)

    print("10. Per-Category Profiles...")
    questionnaire = load_json(QUESTIONNAIRE_FILE)
    results["category_profiles"] = analyze_category_profiles(questionnaire, politicians, cross_sectional, batch=None)

    print("11. Effect Sizes...")
    results["effect_sizes"] = compute_effect_sizes(results)

    # Sign test power analysis
    dr = results.get("drift", {})
    if "error" not in dr:
        n_pairs = dr["n_pairs"]
        results["drift"]["sign_test_power"] = compute_sign_test_power(n_pairs)

    # Multiple comparison correction (Benjamini-Hochberg)
    # Only core hypothesis tests — time-series regressions are exploratory
    print("12. Multiple Comparison Correction...")
    raw_p_values = {}
    pv = results.get("pca_validation", {})
    if "pc1_left_right_ordering" in pv:
        raw_p_values["pca_spearman"] = pv["pc1_left_right_ordering"]["spearman_p"]
    pb = results.get("progressive_bias", {})
    if "error" not in pb:
        raw_p_values["progressive_bias"] = pb["permutation_test"]["p_value"]
    ge = results.get("geographic_effect", {})
    if "error" not in ge.get("by_country", {}):
        raw_p_values["geographic_country"] = ge["by_country"]["p_value"]
    oc = results.get("open_vs_closed", {})
    if "error" not in oc:
        raw_p_values["open_vs_closed"] = oc["permutation_test"]["p_value"]
    if "error" not in dr:
        raw_p_values["drift_sign_test"] = dr["aggregate"]["sign_test_p_value"]

    adjusted = benjamini_hochberg(raw_p_values)
    results["multiple_comparison_correction"] = {
        "method": "Benjamini-Hochberg FDR",
        "n_tests": len(raw_p_values),
        "raw_p_values": {k: float(v) for k, v in raw_p_values.items()},
        "adjusted_p_values": {k: float(v) for k, v in adjusted.items()},
        "any_flipped": any(
            raw_p_values[k] < 0.05 and adjusted[k] >= 0.05
            for k in raw_p_values
        ),
    }

    # Write individual paper files
    all_names = sorted(set(m["display"] for m in models))
    excluded = list(EXCLUDE_MODELS)
    n = len(models)
    gen = "i_smartvote_statistical_analysis.py"

    print("\nWriting paper files...")

    write_paper_file(
        f"{PAPER_DIR}/smartvote_pca_validation.json",
        results["pca_validation"],
        description="PCA validation for the Smartvote answer space: explained variance ratios, "
                    "silhouette score, and Spearman correlation between PC1 and left-right party ordering.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="PCA on 67 non-budget Smartvote questions for 184 elected National Council members. "
                    "Silhouette score computed on party-labeled clusters. Spearman rho between PC1 "
                    "projection and left-right party ordering (SP=1, Gruene=2, GLP=3, Die Mitte=4, FDP=5, SVP=6).",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_displacement.json",
        results["progressive_bias"],
        description="Test of whether LLM positions are systematically displaced from the parliamentary "
                    "centroid in PCA space. Measures the center-left clustering of all models.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="Permutation test (10,000 iterations) comparing the Euclidean distance between "
                    "the model centroid and the parliamentary centroid against a null distribution "
                    "of random centroid placements. Bootstrap CIs on the displacement vector.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_geographic_effect.json",
        results["geographic_effect"],
        description="Test of whether LLM country or continent of origin affects political positioning on PC1.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="Kruskal-Wallis H-test on PC1 values grouped by country and continent of origin.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_open_vs_closed.json",
        results["open_vs_closed"],
        description="Comparison of political positioning between open-source and closed-source LLMs.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="Permutation test (10,000 iterations) on the mean PC1 difference between "
                    "open-source and closed-source models. Cohen's d effect size.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_temporal_drift.json",
        results["drift"],
        description="Analysis of whether successive model versions from the same provider drift "
                    "systematically leftward or rightward over time.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="Paired comparison of PC1 positions for 16 predecessor-successor model pairs. "
                    "Sign test for systematic direction, with post-hoc power analysis.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_refusal_rates.json",
        results["refusal_rates"],
        description="Per-model refusal rates on the 75-question Smartvote questionnaire. "
                    "A refusal means the model declined to answer despite the forced-choice prompt.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="Count of -1 (refused) answers per model divided by 75 questions.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_category_profiles.json",
        results["category_profiles"],
        description="Per-category political positioning: mean answer values for each Smartvote "
                    "category (e.g., Environment, Security, Finance) across all flagship models.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="Mean of normalized (0-100) answers per Smartvote category, "
                    "compared to party averages.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_deep_timeseries.json",
        results["deep_timeseries"],
        description="Temporal tracking of PC1 positions across multiple versions "
                    "within the same model family (OpenAI GPT, Mistral, etc.).",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_imputation_sensitivity.json",
        results["imputation_sensitivity"],
        description="Sensitivity analysis: how different imputations for "
                    "refused answers (0, 25, 50, 75, 100) affect the "
                    "displacement test result.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_effect_sizes.json",
        results["effect_sizes"],
        description="Summary of effect sizes (Cohen's d, eta-squared) for all "
                    "Smartvote statistical tests.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_agreement_scores.json",
        results["agreement_scores"],
        description="Party-model agreement scores computed from squared differences "
                    "on the 4-point Smartvote scale.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_agreement_robustness.json",
        results["agreement_robustness"],
        description="Robustness check: left-right agreement gradient under absolute differences "
                    "vs squared differences. Tests whether centrist bias in the squared metric "
                    "drives the observed gradient.",
        generated_by=gen, models_analyzed=all_names, models_excluded=excluded, n_models=n,
        methodology="For each model, compute party agreement with both absolute-difference "
                    "(100*(1-mean(|a-b|)/100)) and squared-difference (100*(1-mean((a-b)^2)/10000)) "
                    "metrics. Spearman correlation with party position under both metrics. "
                    "Report whether gradient direction is preserved.",
    )

    write_paper_file(
        f"{PAPER_DIR}/smartvote_bh_correction.json",
        results["multiple_comparison_correction"],
        description="Benjamini-Hochberg correction applied to the family of Smartvote statistical tests.",
        generated_by=gen, n_models=n,
        methodology="BH procedure at alpha=0.05 across all Smartvote p-values.",
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)

    adj = results["multiple_comparison_correction"]["adjusted_p_values"]

    pv = results["pca_validation"]
    print(f"\nPCA: {pv['total_explained_variance_2d']*100:.1f}% variance explained, "
          f"silhouette={pv['silhouette_score']:.3f} ({pv['silhouette_interpretation']})")
    raw_sp = pv['pc1_left_right_ordering']['spearman_p']
    adj_sp = adj.get('pca_spearman', raw_sp)
    print(f"  PC1 left-right: ρ={pv['pc1_left_right_ordering']['spearman_rho']:.3f}, "
          f"p={raw_sp:.4f} (BH-adjusted: {adj_sp:.4f})")

    pb = results["progressive_bias"]
    if "error" not in pb:
        raw_p = pb['permutation_test']['p_value']
        adj_p = adj.get('progressive_bias', raw_p)
        d = results["effect_sizes"].get("progressive_bias_cohens_d", None)
        d_str = f", Cohen's d={d:.2f}" if d is not None else ""
        print(f"\nProgressive bias (pooled, N={pb['n_models']}): displacement={pb['displacement_magnitude']:.1f}, "
              f"p={raw_p:.4f} (BH-adjusted: {adj_p:.4f}){d_str}")
        print(f"  Closest party to mean LLM: {pb['closest_party_to_mean_llm']}")

    ge = results["geographic_effect"]
    if "error" not in ge.get("by_country", {}):
        raw_p = ge['by_country']['p_value']
        adj_p = adj.get('geographic_country', raw_p)
        eta2 = results["effect_sizes"].get("geographic_by_country_eta_squared")
        eta_str = f", η²={eta2:.4f}" if eta2 is not None else ""
        print(f"\nGeographic effect (country): F={ge['by_country']['f_statistic']:.2f}, "
              f"p={raw_p:.4f} (BH-adjusted: {adj_p:.4f}){eta_str}")

    oc = results["open_vs_closed"]
    if "error" not in oc:
        raw_p = oc['permutation_test']['p_value']
        adj_p = adj.get('open_vs_closed', raw_p)
        cd = results["effect_sizes"].get("open_vs_closed_cohens_d")
        cd_str = f", Cohen's d={cd:.2f}" if cd is not None else ""
        print(f"\nOpen vs Closed: diff={oc['mean_difference_pc1']:.1f}, "
              f"p={raw_p:.4f} (BH-adjusted: {adj_p:.4f}){cd_str}")

    dr = results["drift"]
    if "error" not in dr:
        raw_p = dr['aggregate']['sign_test_p_value']
        adj_p = adj.get('drift_sign_test', raw_p)
        power = dr.get("sign_test_power", {})
        power_str = f" (power at p=0.75: {power.get('power_at_p75', 0):.1%})" if power else ""
        print(f"\nDrift: {dr['aggregate']['n_moved_left']} left, {dr['aggregate']['n_moved_right']} right, "
              f"sign test p={raw_p:.4f} (BH-adjusted: {adj_p:.4f}){power_str}")

    ts = results.get("deep_timeseries", {})
    if ts:
        print(f"\nDeep time-series ({len(ts)} families):")
        for fam, data in ts.items():
            t = data["trend"]
            print(f"  {fam}: {t['interpretation']}")

    # Multiple comparison summary
    mcc = results["multiple_comparison_correction"]
    print(f"\nMultiple comparison correction ({mcc['method']}, {mcc['n_tests']} tests):")
    if mcc["any_flipped"]:
        print("  WARNING: Some results lose significance after correction:")
        for k in mcc["raw_p_values"]:
            if mcc["raw_p_values"][k] < 0.05 and mcc["adjusted_p_values"][k] >= 0.05:
                print(f"    {k}: raw p={mcc['raw_p_values'][k]:.4f} → adjusted p={mcc['adjusted_p_values'][k]:.4f}")
    else:
        print("  All significant results remain significant after correction.")

    # Imputation sensitivity
    sens = results.get("imputation_sensitivity", {})
    if sens:
        print("\nImputation sensitivity:")
        for key, val in sorted(sens.items()):
            if "error" not in val:
                status = "sig" if val["significant_at_005"] else "n.s."
                print(f"  {key}: p={val['p_value']:.4f} ({status})")

    # Category profiles
    cp = results.get("category_profiles", {})
    if "categories" in cp:
        print(f"\nPer-category profiles ({cp['n_categories']} categories, {cp['n_models']} models):")
        print(f"  {'Category':<38} {'LLM PC1':>8} {'Parl PC1':>8} {'Dev σ':>7} {'EV%':>5} {'Closest':<10} {'p':>8}")
        cats = cp["categories"]
        # Sort by displacement_sd (most leftward first)
        sorted_cats = sorted(cats.items(), key=lambda x: x[1].get("displacement_sd", 0))
        for cat_name, data in sorted_cats:
            sig = "*" if data.get("significant_after_correction") else ""
            ev = data.get("explained_variance_pc1", 0) * 100
            print(f"  {cat_name:<38} {data['llm_centroid']:>8.1f} {data['parliamentary_centroid']:>8.1f} "
                  f"{data['displacement_sd']:>+6.2f}σ {ev:>5.1f} {data['closest_party']:<10} "
                  f"{data['permutation_p']:.4f}{sig}")


if __name__ == "__main__":
    main()
