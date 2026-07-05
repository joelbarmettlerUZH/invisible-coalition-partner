"""
Generate publication-quality figures for the FAccT paper.

Usage:
    cd projects/llm-political-bias
    uv run python j_smartvote_paper_figures.py

Outputs PDF figures to paper/figures/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from datetime import datetime

# Publication settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RESULTS_DIR = "results"
FIGURES_DIR = "paper/figures"
DATA_DIR = "data"

PARTY_ORDER = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]
PARTY_LABELS = {
    "SP": "SP",
    "Grüne": "Greens",
    "GLP": "GLP",
    "Die Mitte": "Centre",
    "FDP": "FDP",
    "SVP": "SVP",
}

# 8 usable Volksabstimmungen flagship model IDs (Gemini excluded due to 98% refusal)
# These IDs match all_model_answers.json
VOLKSABSTIMMUNGEN_FLAGSHIP_IDS = {
    "openai/gpt-5.4",
    "anthropic/claude-opus-4.6",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
    "x-ai/grok-4.20",
    "mistralai/mistral-large-2512",
    "cohere/command-a",
    "qwen/qwen3.5-plus-02-15",
}

# Fixed display-name order for heatmaps (matches Volksabstimmungen left-party agreement order)
HEATMAP_MODEL_ORDER = [
    "Claude Opus 4.6",
    "GPT-5.4",
    "Qwen 3.5 Plus",
    "Mistral Large",
    "Llama 4 Maverick",
    "Grok 4.20",
    "Command A",
    "DeepSeek V3.2",
]

# Smartvote display names differ from Volksabstimmungen for some models;
# normalize to shorter canonical forms so both heatmaps use the same labels.
DISPLAY_NAME_OVERRIDES = {
    "Mistral Large (Dec 2025)": "Mistral Large",
    "DeepSeek v3.2": "DeepSeek V3.2",
}

# Country display and colors for strip plot
COUNTRY_COLORS = {
    "USA": "#3c78d8",
    "China": "#cc0000",
    "France": "#6aa84f",
    "Canada": "#e69138",
    "Israel": "#8e7cc3",
}


# ── Shared data loading ──────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_shared():
    """Load PCA models, politician points, all model answers, and family config."""
    pca_1d = load_json(f"{RESULTS_DIR}/website/smartvote_pca_1d.json")
    pca_2d = load_json(f"{RESULTS_DIR}/website/smartvote_pca_2d.json")
    all_models = load_json(f"{DATA_DIR}/answers/all_model_answers.json")
    families = load_json(f"{DATA_DIR}/model_families.json")
    return pca_1d, pca_2d, all_models, families


def project(model, pca_data, ndim=2):
    """Project a model's answers into PCA space (1D or 2D). Returns negated PC1.

    PC1 is negated so that left-wing parties appear on the left side of figures.
    This is a display convention only; PCA components are sign-invariant.
    """
    pca_info = pca_data["metadata"]["pca"]
    mean = np.array(pca_info["mean"])
    components = np.array(pca_info["components"])
    question_ids = pca_data["metadata"]["question_ids"]

    lookup = {}
    for ans in model.get("answers", []):
        qid, val = ans.get("questionId"), ans.get("value", 50)
        if qid and val != -1:
            lookup[qid] = val
    vec = np.array([lookup.get(qid, 50) for qid in question_ids], dtype=float)
    coords = np.dot(vec - mean, components.T).flatten()
    # Negate PC1 so left-wing parties appear on the left
    coords[0] = -coords[0]
    return coords[:ndim]


def get_flagships(all_models, families):
    """Return list of flagship model dicts, one per family."""
    flagship_names = set()
    for fam in families.values():
        f = fam.get("flagship")
        if f:
            flagship_names.add(f)
    return [m for m in all_models if m["name"] in flagship_names]


def get_drift_pairs(all_models, families):
    """Return list of (old_model, new_model) dicts for families with drift pairs."""
    pairs = []
    models_by_name = {m["name"]: m for m in all_models}
    for fam_key, fam in families.items():
        drift = fam.get("drift")
        if not drift or len(drift) != 2:
            continue
        old = models_by_name.get(drift[0])
        new = models_by_name.get(drift[1])
        if old and new:
            pairs.append((old, new, fam.get("display", fam_key)))
    return pairs


def get_pol_points(pca_data):
    """Return politician points from PCA data (already projected)."""
    return pca_data.get("points", [])


# ── Figure 1: 2D PCA scatter ─────────────────────────────────────────────────

def fig_pca_2d(pca_2d, flagships):
    """2D PCA scatter: politicians colored by party, flagship LLMs as labeled diamonds."""
    pol_points = get_pol_points(pca_2d)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Plot politicians by party
    for party in PARTY_ORDER:
        pts = [p for p in pol_points if p["party"] == party]
        if not pts:
            continue
        xs = [-p["coords"][0] for p in pts]
        ys = [p["coords"][1] for p in pts]
        color = pts[0]["color"]
        ax.scatter(xs, ys, c=color, s=15, alpha=0.5, label=PARTY_LABELS[party],
                   edgecolors="none", zorder=2)

    # Project and plot flagship LLMs
    llm_coords = []
    for m in flagships:
        c = project(m, pca_2d, ndim=2)
        llm_coords.append(c)
        ax.scatter(c[0], c[1], c="black", s=30, marker="D", zorder=3,
                   edgecolors="white", linewidths=0.5)

    llm_coords = np.array(llm_coords)

    # Convex hull around LLMs
    if len(llm_coords) >= 3:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(llm_coords)
        hull_pts = llm_coords[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.08, color="black", zorder=1)
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="black", linewidth=0.5,
                linestyle="--", alpha=0.4, zorder=1)

    # Centroids
    pol_cx = np.mean([-p["coords"][0] for p in pol_points])
    pol_cy = np.mean([p["coords"][1] for p in pol_points])
    ax.scatter(pol_cx, pol_cy, c="gray", s=80, marker="+", linewidths=1.5, zorder=4)
    ax.scatter(llm_coords[:, 0].mean(), llm_coords[:, 1].mean(),
               c="black", s=80, marker="+", linewidths=1.5, zorder=4)

    # Compute explained variance ratios dynamically from PCA data
    ev = pca_2d["metadata"]["pca"]["explained_variance"]
    # Approximate total variance: use raw eigenvalues and estimate remaining from data
    # If noise_variance not available, compute ratio from first two components relative to total
    total_var = sum(pca_2d["metadata"]["pca"].get("total_variance", ev))
    if total_var == sum(ev):
        # total_variance not stored — fall back to loading from paper analysis file
        try:
            sv_pca_val = load_json(f"{RESULTS_DIR}/paper/smartvote_pca_validation.json")
            ev_ratio = sv_pca_val["results"]["explained_variance_ratio"]
        except (FileNotFoundError, KeyError):
            ev_ratio = [e / sum(ev) for e in ev]
    else:
        ev_ratio = [e / total_var for e in ev]
    ax.set_xlabel(f"PC1 ({ev_ratio[0]*100:.1f}% variance) \u2014 Left \u2190 \u2192 Right")
    ax.set_ylabel(f"PC2 ({ev_ratio[1]*100:.1f}% variance)")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.axhline(y=0, color="gray", linewidth=0.3, linestyle=":")
    ax.axvline(x=0, color="gray", linewidth=0.3, linestyle=":")

    fig.savefig(f"{FIGURES_DIR}/fig-pca-2d.pdf")
    plt.close(fig)
    print("  -> fig-pca-2d.pdf")


# ── Figure 2: 1D PCA strip (parties + country groups) ────────────────────────

def fig_pca_1d(pca_1d, flagships):
    """1D strip: rows for parties, then rows for country-of-origin groups."""
    pol_points = get_pol_points(pca_1d)

    # Group flagships by country
    country_order = ["USA", "China", "France", "Canada"]
    country_models = {c: [] for c in country_order}
    for m in flagships:
        c = m.get("country")
        if c in country_models:
            country_models[c].append(m)
        else:
            # Israel etc. — fold into nearest group or skip
            country_models.setdefault(c, []).append(m)
            if c not in country_order:
                country_order.append(c)

    n_party_rows = len(PARTY_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Party rows
    for i, party in enumerate(PARTY_ORDER):
        pts = [p for p in pol_points if p["party"] == party]
        if not pts:
            continue
        xs = [-p["coords"][0] for p in pts]
        color = pts[0]["color"]
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(xs))
        ax.scatter(xs, [i + j for j in jitter], c=color, s=10, alpha=0.4,
                   edgecolors="none", zorder=2)
        ax.scatter(np.mean(xs), i, c=color, s=60, marker="|", linewidths=2, zorder=3)

    # Separator line
    sep_y = n_party_rows - 0.5 + 0.5
    ax.axhline(y=sep_y, color="gray", linewidth=0.5, linestyle="-", alpha=0.3)

    # Country rows
    for j, country in enumerate(country_order):
        row = n_party_rows + 1 + j
        models = country_models.get(country, [])
        if not models:
            continue
        color = COUNTRY_COLORS.get(country, "gray")
        xs = []
        for m in models:
            c = project(m, pca_1d, ndim=1)
            xs.append(c[0])
        jitter = np.random.RandomState(42 + j).uniform(-0.15, 0.15, len(xs))
        ax.scatter(xs, [row + jj for jj in jitter], c=color, s=20, marker="D",
                   zorder=3, edgecolors="white", linewidths=0.3)
        ax.scatter(np.mean(xs), row, c=color, s=60, marker="|", linewidths=2, zorder=3)

    # Labels
    labels = [PARTY_LABELS[p] for p in PARTY_ORDER] + [""] + country_order
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("PC1 \u2014 Left \u2190 \u2192 Right")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle=":")
    ax.invert_yaxis()

    fig.savefig(f"{FIGURES_DIR}/fig-pca-1d.pdf")
    plt.close(fig)
    print("  -> fig-pca-1d.pdf")


# ── Figure 3: Agreement heatmap (flagships only) ─────────────────────────────

def fig_agreement_heatmap(pca_1d, all_models):
    """Heatmap: 8 Volksabstimmungen flagship models (rows) x parties (columns)."""
    # Filter to only the 8 usable Volksabstimmungen flagships
    flagships = [m for m in all_models if m["name"] in VOLKSABSTIMMUNGEN_FLAGSHIP_IDS]

    # We need to compute agreement scores on-the-fly for flagships
    politicians = load_json(f"{DATA_DIR}/answers/nationalrat_members.json")
    questionnaire = load_json(f"{DATA_DIR}/questionnaire/questionnaire.json")

    # Get question IDs (exclude BudgetCategory)
    question_ids = []
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        for q in cat.get("questions", []):
            question_ids.append(q["id"])

    # Group politicians by party
    party_pols = {p: [] for p in PARTY_ORDER}
    for pol in politicians:
        party = pol.get("party", {}).get("abbreviation", pol.get("partyAbbreviation"))
        if party in party_pols:
            party_pols[party].append(pol)

    def get_answer_map(entity):
        m = {}
        for a in entity.get("answers", []):
            qid, val = a.get("questionId"), a.get("value", -1)
            if val != -1:
                m[qid] = val
        return m

    def agreement(llm_map, pol_map):
        """Squared-difference agreement between an LLM and a politician."""
        overlap = set(llm_map.keys()) & set(pol_map.keys()) & set(question_ids)
        if not overlap:
            return None
        sq_diffs = [(llm_map[q] - pol_map[q]) ** 2 for q in overlap]
        return 100 * (1 - np.mean(sq_diffs) / 10000)

    # Compute agreement for each flagship × party
    model_data = {}
    for m in flagships:
        llm_map = get_answer_map(m)
        row = []
        for party in PARTY_ORDER:
            scores = []
            for pol in party_pols[party]:
                pol_map = get_answer_map(pol)
                a = agreement(llm_map, pol_map)
                if a is not None:
                    scores.append(a)
            row.append(np.mean(scores) if scores else 0)
        display = m.get("display", m["name"].split("/")[-1])
        display = DISPLAY_NAME_OVERRIDES.get(display, display)
        model_data[display] = {
            "row": row,
            "country": m.get("country", "?"),
            "os": "OS" if m.get("open_source") else "CS",
        }

    # Sort by fixed display-name order (matching Volksabstimmungen heatmap)
    ordered_displays = [d for d in HEATMAP_MODEL_ORDER if d in model_data]
    # Add any remaining models not in the fixed order
    for d in model_data:
        if d not in ordered_displays:
            ordered_displays.append(d)

    model_names = ordered_displays
    matrix = np.array([model_data[d]["row"] for d in ordered_displays])
    model_countries = [model_data[d]["country"] for d in ordered_displays]
    model_os = [model_data[d]["os"] for d in ordered_displays]

    # Y-labels: display name only (country/license tags omitted for legibility)
    ylabels = list(model_names)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=40, vmax=95, aspect="auto")

    ax.set_xticks(range(len(PARTY_ORDER)))
    ax.set_xticklabels([PARTY_LABELS[p] for p in PARTY_ORDER])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8.2)

    for i in range(len(model_names)):
        for j in range(len(PARTY_ORDER)):
            val = matrix[i, j]
            color = "white" if val < 65 or val > 88 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Agreement (%)", fontsize=12)

    fig.savefig(f"{FIGURES_DIR}/fig-agreement-heatmap.pdf")
    plt.close(fig)
    print("  -> fig-agreement-heatmap.pdf")


# ── Figure 4: Drift arrow plot (top drifts only) ─────────────────────────────

def fig_drift(pca_2d, drift_pairs):
    """Arrow plot showing largest drifts between oldest and newest model per family."""
    pol_points = get_pol_points(pca_2d)

    # Load bootstrap CIs from temporal drift analysis if available
    drift_cis = {}
    try:
        sv_drift = load_json(f"{RESULTS_DIR}/paper/smartvote_temporal_drift.json")
        for pair in sv_drift.get("results", {}).get("drift_2025_2026", {}).get("pairs", []):
            key = (pair["old_name"], pair["new_name"])
            drift_cis[key] = {
                "ci_lower": pair.get("drift_ci_lower"),
                "ci_upper": pair.get("drift_ci_upper"),
            }
    except (FileNotFoundError, KeyError):
        pass

    # Project all drift pairs
    projected_pairs = []
    for old_m, new_m, family_name in drift_pairs:
        old_c = project(old_m, pca_2d, ndim=2)
        new_c = project(new_m, pca_2d, ndim=2)
        magnitude = np.sqrt((new_c[0] - old_c[0])**2 + (new_c[1] - old_c[1])**2)
        ci_key = (old_m.get("display", old_m["name"]), new_m.get("display", new_m["name"]))
        ci = drift_cis.get(ci_key, {})
        projected_pairs.append({
            "old": old_c, "new": new_c,
            "old_name": old_m.get("display", old_m["name"]),
            "new_name": new_m.get("display", new_m["name"]),
            "family": family_name,
            "magnitude": magnitude,
            "direction": "left" if new_c[0] < old_c[0] else "right",
            "ci_lower": ci.get("ci_lower"),
            "ci_upper": ci.get("ci_upper"),
        })

    # Keep only top 5 by magnitude
    projected_pairs.sort(key=lambda p: p["magnitude"], reverse=True)
    top_pairs = projected_pairs[:5]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Party centroids for context
    for party in PARTY_ORDER:
        pts = [p for p in pol_points if p["party"] == party]
        if not pts:
            continue
        cx = np.mean([-p["coords"][0] for p in pts])
        cy = np.mean([p["coords"][1] for p in pts])
        color = pts[0]["color"]
        ax.scatter(cx, cy, c=color, s=100, alpha=0.3, marker="s", zorder=1)
        ax.annotate(PARTY_LABELS[party], (cx, cy), fontsize=9, alpha=0.5,
                    ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    for pair in top_pairs:
        old, new = pair["old"], pair["new"]
        color = "#2166ac" if pair["direction"] == "left" else "#b2182b"

        arrow = FancyArrowPatch(
            (old[0], old[1]), (new[0], new[1]),
            arrowstyle="-|>", mutation_scale=10,
            color=color, linewidth=1.2, zorder=3
        )
        ax.add_patch(arrow)

        # Draw magnitude CI as a thin bracket at the arrow endpoint
        if pair.get("ci_lower") is not None and pair.get("ci_upper") is not None:
            # Show CI as a small error bar at the new endpoint
            ci_lo = pair["ci_lower"]
            ci_hi = pair["ci_upper"]
            ax.plot(new[0], new[1], marker="o", markersize=2.5, color=color, zorder=4)
            # Text annotation with CI
            ci_text = f"[{ci_lo:.0f},{ci_hi:.0f}]"
            ax.annotate(ci_text, (new[0], new[1]), fontsize=5.2,
                        xytext=(4, -6), textcoords="offset points", color=color, alpha=0.7)

        # Label at arrow origin, rotated to match its angle
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Flip text angle so it's always readable (not upside down)
        display_angle = angle
        flipped = False
        if display_angle > 90:
            display_angle -= 180
            flipped = True
        elif display_angle < -90:
            display_angle += 180
            flipped = True
        # Company name only (e.g., "OpenAI GPT" -> "OpenAI", "xAI Grok" -> "xAI")
        company = pair["family"].split()[0] if pair["family"] else pair["family"]
        # When angle is flipped, text reads right-to-left, so use ha="right"
        # to anchor text end at the arrow origin
        h_align = "right" if flipped else "left"
        ax.text(old[0], old[1], company, fontsize=7.5, color=color,
                ha=h_align, va="bottom", rotation=display_angle,
                rotation_mode="anchor",
                transform_rotates_text=True)

    legend_elements = [
        Line2D([0], [0], color="#2166ac", linewidth=1.5, label="Moved left (progressive)"),
        Line2D([0], [0], color="#b2182b", linewidth=1.5, label="Moved right (conservative)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10.5, bbox_to_anchor=(0.0, 1.0))

    ax.set_xlabel("PC1 \u2014 Left \u2190 \u2192 Right")
    ax.set_ylabel("PC2")
    ax.axhline(y=0, color="gray", linewidth=0.3, linestyle=":")
    ax.axvline(x=0, color="gray", linewidth=0.3, linestyle=":")

    fig.savefig(f"{FIGURES_DIR}/fig-drift.pdf")
    plt.close(fig)
    print("  -> fig-drift.pdf")


# ── Figure 5: Timeline ───────────────────────────────────────────────────────

def fig_timeline(pca_1d, all_models):
    """Timeline: release date (y) vs PC1 position (x), one line per company."""
    import matplotlib.dates as mdates

    # Map families to companies (one line per company)
    FAMILY_TO_COMPANY = {
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
    COMPANY_CONFIG = {
        "OpenAI": {"color": "#10a37f"},
        "xAI": {"color": "#1da1f2"},
        "Mistral": {"color": "#ff7000"},
        "Anthropic": {"color": "#d4a574"},
    }

    # Collect all models belonging to tracked companies
    company_data = {}
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

        pc1 = project(model, pca_1d, ndim=1)[0]
        date = datetime.strptime(released, "%Y-%m-%d")
        size = model.get("size") or 0
        display = model.get("display", model["name"].split("/")[-1])
        company_data[company].append({
            "name": display, "pc1": pc1, "date": date, "size": size,
        })

    # For each company: sort by date, deduplicate same-date releases (keep largest)
    for company in company_data:
        points = company_data[company]
        points.sort(key=lambda p: (p["date"], -p["size"]))
        # Deduplicate: for same date, keep only the one with largest size
        deduped = []
        seen_dates = set()
        for p in points:
            date_str = p["date"].strftime("%Y-%m-%d")
            if date_str not in seen_dates:
                seen_dates.add(date_str)
                deduped.append(p)
        company_data[company] = deduped

    # Party centroids for reference
    pol_points = get_pol_points(pca_1d)
    party_centroids = {}
    for party in PARTY_ORDER:
        pts = [p for p in pol_points if p.get("party") == party]
        if pts:
            party_centroids[party] = -np.mean([p["coords"][0] for p in pts])

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    for party in PARTY_ORDER:
        if party in party_centroids:
            ax.axvline(x=party_centroids[party], color="gray", linewidth=0.3,
                       linestyle=":", alpha=0.5, zorder=0)
            ax.text(party_centroids[party], 0.01, PARTY_LABELS[party],
                    transform=ax.get_xaxis_transform(), fontsize=7.5, ha="center",
                    color="gray", alpha=0.6)

    for company, config in COMPANY_CONFIG.items():
        if company not in company_data or not company_data[company]:
            continue
        points = company_data[company]
        xs = [p["pc1"] for p in points]
        ys = [p["date"] for p in points]
        ax.plot(xs, ys, color=config["color"], linewidth=1.2,
                marker="o", markersize=4, zorder=3, label=company)
        for p in points:
            ax.annotate(p["name"], (p["pc1"], p["date"]), fontsize=6.8,
                        xytext=(4, 0), textcoords="offset points", va="center",
                        color=config["color"], alpha=0.8)

    ax.set_xlabel("PC1 (1D) \u2014 Left \u2190 \u2192 Right")
    ax.set_ylabel("Release Date")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.invert_yaxis()

    fig.savefig(f"{FIGURES_DIR}/fig-timeline.pdf")
    plt.close(fig)
    print("  -> fig-timeline.pdf")


# ── Figure 6: Refusal rates ──────────────────────────────────────────────────

def fig_refusal(all_models):
    """Bar chart of refusal rates across all models with any refusals."""
    questionnaire = load_json(f"{DATA_DIR}/questionnaire/questionnaire.json")
    question_ids = set()
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        for q in cat.get("questions", []):
            question_ids.add(q["id"])
    n_questions = len(question_ids)

    # Compute refusal rates for all models (explicit refusals only, not missing answers)
    refusal_data = []
    for m in all_models:
        answers = m.get("answers", [])
        relevant = [a for a in answers if a.get("questionId") in question_ids]
        explicit_refusals = sum(1 for a in relevant if a.get("value") == -1)
        missing = n_questions - len(relevant)
        rate = explicit_refusals / n_questions if n_questions > 0 else 0
        refusal_data.append({
            "display": m.get("display", m["name"].split("/")[-1]),
            "rate": rate,
            "explicit_refusals": explicit_refusals,
            "missing_answers": missing,
        })

    # Sort and show all with rate > 0, plus a few zero-rate for context
    refusal_data.sort(key=lambda x: x["rate"], reverse=True)
    # Keep models with refusals + top 5 zero-rate models
    with_refusal = [d for d in refusal_data if d["rate"] > 0]
    without = [d for d in refusal_data if d["rate"] == 0][:5]
    to_show = with_refusal + without

    names = [d["display"] for d in to_show]
    rates = [d["rate"] * 100 for d in to_show]

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, len(to_show) * 0.18)))
    colors = ["#d32f2f" if r > 5 else "#ff9800" if r > 0 else "#4caf50" for r in rates]
    bars = ax.barh(range(len(names)), rates, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Refusal Rate (%)")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)

    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.1f}%", va="center", fontsize=9)

    ax.text(0.95, 0.02,
            f"Showing {len(with_refusal)} models with refusals + {len(without)} zero-rate for reference.\n"
            f"{len(refusal_data) - len(to_show)} additional models with 0% refusal omitted.",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            style="italic", color="gray")

    fig.savefig(f"{FIGURES_DIR}/fig-refusal.pdf")
    plt.close(fig)
    print("  -> fig-refusal.pdf")


# ── Figure 7: Per-category political profiles ──────────────────────────────

# English translations for category names
CATEGORY_LABELS = {
    "Sozialstaat & Familie": "Social Welfare",
    "Gesundheit": "Healthcare",
    "Bildung": "Education",
    "Migration & Integration": "Migration",
    "Gesellschaft & Ethik": "Society & Ethics",
    "Umweltschutz": "Environment",
    "Finanzen & Steuern": "Finance & Tax",
    "Wirtschaft & Arbeit": "Economy & Labor",
    "Energie & Verkehr": "Energy & Transport",
    "Demokratie, Medien & Digitalisierung": "Democracy & Media",
    "Sicherheit & Armee": "Security & Military",
    "Werthaltungen": "Values",
    "Aussenbeziehungen": "Foreign Relations",
}


def fig_category_profiles(all_models, families):
    """Dot plot: per-category PCA (PC1) showing party and LLM positions on left-right axis."""
    from sklearn.decomposition import PCA as SkPCA

    questionnaire = load_json(f"{DATA_DIR}/questionnaire/questionnaire.json")
    politicians = load_json(f"{DATA_DIR}/answers/nationalrat_members.json")
    sv_cat = load_json(f"{RESULTS_DIR}/paper/smartvote_category_profiles.json")

    # Get category -> question IDs mapping
    categories = {}
    for cat in questionnaire.get("categories", []):
        if cat.get("type") == "BudgetCategory":
            continue
        categories[cat["name"]] = [q["id"] for q in cat["questions"]]

    # Get flagship models
    flagships = get_flagships(all_models, families)

    # Get statistical results for sorting and significance
    cat_stats = sv_cat.get("results", {}).get("categories", {})

    # Sort categories by displacement_sd (most leftward first)
    sorted_cats = sorted(categories.keys(),
                         key=lambda c: cat_stats.get(c, {}).get("displacement_sd", 0))

    def get_answer_map(entity):
        m = {}
        for a in entity.get("answers", []):
            qid, val = a.get("questionId"), a.get("value", -1)
            if val != -1:
                m[qid] = val
        return m

    def build_vec(entity, qids, default=50):
        am = get_answer_map(entity)
        return np.array([am.get(qid, default) for qid in qids], dtype=float)

    # Group politicians by party
    party_pols = {p: [] for p in PARTY_ORDER}
    for pol in politicians:
        party = pol.get("party", {}).get("abbreviation", pol.get("partyAbbreviation"))
        if party in party_pols:
            party_pols[party].append(pol)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    n_cats = len(sorted_cats)
    y_positions = list(range(n_cats))

    for i, cat_name in enumerate(sorted_cats):
        qids = categories[cat_name]
        y = i

        # Build politician matrix and fit 1D PCA for this category
        pol_vecs = np.array([build_vec(pol, qids) for pol in politicians])
        pca_cat = SkPCA(n_components=1)
        pol_pc1 = pca_cat.fit_transform(pol_vecs).flatten()

        # Orient so SP < SVP (left < right)
        sp_mask = np.array([pol.get("party", {}).get("abbreviation",
                   pol.get("partyAbbreviation")) == "SP" for pol in politicians])
        svp_mask = np.array([pol.get("party", {}).get("abbreviation",
                    pol.get("partyAbbreviation")) == "SVP" for pol in politicians])
        flip = 1.0
        if pol_pc1[sp_mask].mean() > pol_pc1[svp_mask].mean():
            flip = -1.0
            pol_pc1 = -pol_pc1

        # Parliamentary centroid line
        ax.axvline(x=0, color="gray", linewidth=0.3, linestyle=":", alpha=0.3, zorder=0)

        # Plot party centroids
        for party in PARTY_ORDER:
            mask = np.array([pol.get("party", {}).get("abbreviation",
                    pol.get("partyAbbreviation")) == party for pol in politicians])
            if mask.any():
                party_mean = pol_pc1[mask].mean()
                # Get party color from first matching politician
                pols = party_pols[party]
                color = pols[0].get("party", {}).get("color",
                         pols[0].get("partyColor", "gray")) if pols else "gray"
                ax.scatter(party_mean, y, c=color, s=25, alpha=0.6,
                          edgecolors="white", linewidths=0.3, zorder=2)

        # Project flagship LLMs
        llm_vecs = np.array([build_vec(m, qids) for m in flagships])
        llm_pc1 = pca_cat.transform(llm_vecs).flatten() * flip

        llm_mean = llm_pc1.mean()
        llm_q25 = np.percentile(llm_pc1, 25)
        llm_q75 = np.percentile(llm_pc1, 75)

        # IQR whisker
        ax.plot([llm_q25, llm_q75], [y, y],
                color="black", linewidth=1.5, alpha=0.3, zorder=3)
        ax.scatter(llm_mean, y, c="black", s=40, marker="D",
                  edgecolors="white", linewidths=0.5, zorder=4)

    # Y-axis labels (English translations)
    labels = [CATEGORY_LABELS.get(c, c) for c in sorted_cats]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Category PC1 \u2014 Left \u2190 \u2192 Right")
    ax.invert_yaxis()

    # Place significance markers after xlim is finalized
    xlim = ax.get_xlim()
    for i, cat_name in enumerate(sorted_cats):
        cs = cat_stats.get(cat_name, {})
        if cs.get("significant_after_correction"):
            ax.text(xlim[1] + 2, i, "*", fontsize=13.5, ha="left", va="center",
                    fontweight="bold", color="black", clip_on=False)

    # Add light grid
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)

    # Legend: party dots + LLM diamond
    legend_elements = []
    for party in PARTY_ORDER:
        pols = party_pols[party]
        if pols:
            color = pols[0].get("party", {}).get("color",
                     pols[0].get("partyColor", "gray"))
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=5, label=PARTY_LABELS[party]))
    legend_elements.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
               markersize=5, label="LLM mean"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              ncol=2, framealpha=0.9)

    fig.savefig(f"{FIGURES_DIR}/fig-category-profiles.pdf")
    plt.close(fig)
    print("  -> fig-category-profiles.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...")
    pca_1d, pca_2d, all_models, families = load_shared()
    flagships = get_flagships(all_models, families)
    drift_pairs = get_drift_pairs(all_models, families)

    print(f"  Flagships: {len(flagships)} models")
    print(f"  Drift pairs: {len(drift_pairs)} pairs")

    fig_pca_2d(pca_2d, flagships)
    fig_pca_1d(pca_1d, flagships)
    fig_agreement_heatmap(pca_1d, all_models)
    fig_drift(pca_2d, drift_pairs)
    fig_timeline(pca_1d, all_models)
    fig_refusal(all_models)
    fig_category_profiles(all_models, families)
    print("Done.")


if __name__ == "__main__":
    main()
