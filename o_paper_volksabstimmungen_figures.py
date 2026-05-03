"""
Generate publication-quality figures for the Volksabstimmungen extension.

Usage:
    uv run python o_paper_volksabstimmungen_figures.py

Outputs PDF figures to paper/figures/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _volksabstimmungen_constants import MAIN_PARTIES, PARTY_COLORS, FLAGSHIP_MODELS

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FIGURES_DIR = "paper/figures"
PAPER_DIR = "results/paper"
DATA_DIR = "data"

PARTY_LABELS = {
    "SP": "SP",
    "Grüne": "Greens",
    "GLP": "GLP",
    "Die Mitte": "Centre",
    "FDP": "FDP",
    "SVP": "SVP",
}

# Fixed model order for heatmaps — must match j_smartvote_paper_figures.py
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

# Model metadata lookup built from FLAGSHIP_MODELS constant
_FLAGSHIP_META = {m["name"]: m for m in FLAGSHIP_MODELS}


def _model_label(name: str, display: str) -> str:
    """Format model label with country and license tag, matching Smartvote heatmap style."""
    meta = _FLAGSHIP_META.get(name, {})
    country = meta.get("country", "")
    license_tag = "OS" if meta.get("open_source") else "CS"
    if country:
        return f"{display}  [{country}, {license_tag}]"
    return display


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Figure 1: Volksabstimmungen Agreement Heatmap ───────────────────────────

def fig_volksabstimmungen_heatmap(analysis):
    """Agreement heatmap with country/license tags matching the Smartvote heatmap."""
    v1 = analysis["v1_parolen_agreement"]
    model_names = [k for k in v1 if not k.startswith("_")]

    display_to_name = {v1[n]["display"]: n for n in model_names}
    ordered = [display_to_name[d] for d in HEATMAP_MODEL_ORDER if d in display_to_name]
    for n in model_names:
        if n not in ordered:
            ordered.append(n)
    model_names = ordered

    matrix = []
    labels = []
    for name in model_names:
        row = []
        for party in MAIN_PARTIES:
            val = v1[name]["parties"].get(party, {}).get("agreement")
            row.append(val if val is not None else 50)
        matrix.append(row)
        labels.append(_model_label(name, v1[name]["display"]))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    # Same colorscale bounds as Smartvote heatmap for direct comparison
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=40, vmax=95)

    ax.set_xticks(range(len(MAIN_PARTIES)))
    ax.set_xticklabels([PARTY_LABELS.get(p, p) for p in MAIN_PARTIES])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(MAIN_PARTIES)):
            val = matrix[i, j]
            color = "white" if val < 55 or val > 82 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_xlabel("Party")
    ax.set_title("Volksabstimmungen Agreement (%)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Agreement (%)")

    fig.savefig(f"{FIGURES_DIR}/fig-volksabstimmungen-heatmap.pdf")
    plt.close(fig)
    print("  Saved fig-volksabstimmungen-heatmap.pdf")


# ── Figure 2: Popular Vote Alignment ────────────────────────────────────────

def fig_popular_alignment(analysis):
    """Model alignment with popular vote by margin bucket."""
    v2 = analysis["v2_popular_alignment"]
    model_names = [k for k in v2 if not k.startswith("_")]

    buckets = ["close", "moderate", "decisive", "all"]
    bucket_labels = ["Close\n(\u22645%)", "Moderate\n(5\u201310%)", "Decisive\n(>10%)", "Overall"]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(buckets))
    width = 0.8 / len(model_names)

    for i, name in enumerate(model_names):
        vals = []
        for bucket in buckets:
            data = v2[name]["buckets"].get(bucket, {})
            vals.append(data.get("alignment_pct", 0) or 0)
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=v2[name]["display"], alpha=0.85)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Alignment with popular majority (%)")
    ax.set_title("Model Alignment with Referendum Outcomes")
    ax.legend(fontsize=5, ncol=3, loc="lower left")
    ax.set_ylim(0, 100)

    fig.savefig(f"{FIGURES_DIR}/fig-popular-alignment.pdf")
    plt.close(fig)
    print("  Saved fig-popular-alignment.pdf")


# ── Figure 3: Röstigraben Scatter ──────────────────────────────────────────

def fig_roestigraben(analysis):
    """Scatter: actual DE-FR cantonal gap vs model DE-FR answer gap."""
    v5 = analysis["v5_roestigraben"]
    per_vote = v5.get("per_vote", [])

    if not per_vote:
        print("  SKIP fig-roestigraben.pdf \u2014 no data")
        return

    actual = [v["actual_gap"] for v in per_vote]
    model = [v["avg_model_gap"] for v in per_vote]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(actual, model, s=20, alpha=0.7, c="#3c78d8", edgecolors="white", linewidths=0.3)

    if len(actual) >= 3:
        z = np.polyfit(actual, model, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(actual), max(actual), 100)
        ax.plot(x_range, p(x_range), "r-", linewidth=1, alpha=0.7)

    rho = v5.get("spearman_rho")
    p_val = v5.get("spearman_p")
    if rho is not None:
        ax.text(0.05, 0.95, f"Spearman \u03c1 = {rho:.3f}\np = {p_val:.4f}",
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Actual R\u00f6stigraben (DE \u2212 FR canton Ja%)")
    ax.set_ylabel("Model language gap (DE \u2212 FR prompt answer)")
    ax.set_title("R\u00f6stigraben: Actual vs. Model Language Gap")

    fig.savefig(f"{FIGURES_DIR}/fig-roestigraben.pdf")
    plt.close(fig)
    print("  Saved fig-roestigraben.pdf")


# ── Figure 4: Language Consistency ──────────────────────────────────────────

def fig_language_consistency(analysis):
    """Bar chart: per-model cross-linguistic consistency rate."""
    v4 = analysis["v4_cross_linguistic"]
    model_names = sorted(v4.keys(), key=lambda k: v4[k].get("consistency_rate", 0) or 0, reverse=True)

    if not model_names:
        print("  SKIP fig-language-consistency.pdf \u2014 no data")
        return

    displays = [v4[n]["display"] for n in model_names]
    rates = [v4[n].get("consistency_rate", 0) or 0 for n in model_names]

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.barh(range(len(displays)), rates, color="#3c78d8", alpha=0.8)

    ax.set_yticks(range(len(displays)))
    ax.set_yticklabels(displays)
    ax.set_xlabel("4-language consistency rate (%)")
    ax.set_title("Cross-Linguistic Consistency")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    for i, v in enumerate(rates):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=6)

    fig.savefig(f"{FIGURES_DIR}/fig-language-consistency.pdf")
    plt.close(fig)
    print("  Saved fig-language-consistency.pdf")


# ── Figure 5: Convergent Validity ───────────────────────────────────────────

def fig_convergent_validity(analysis):
    """Scatter: Smartvote vs Volksabstimmungen party agreement, colored by party."""
    v3 = analysis["v3_convergent_validity"]

    sv_points = []
    va_points = []
    party_labels_list = []

    per_model = v3.get("per_model", v3)
    for model_name, data in per_model.items():
        sv = data.get("smartvote_agreement", {})
        va = data.get("volksabstimmungen_agreement", {})
        if not sv or not va:
            continue
        for party in MAIN_PARTIES:
            sv_val = sv.get(party)
            va_val = va.get(party)
            if sv_val is not None and va_val is not None:
                sv_points.append(sv_val)
                va_points.append(va_val)
                party_labels_list.append(party)

    if len(sv_points) < 3:
        print("  SKIP fig-convergent-validity.pdf \u2014 insufficient data")
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    for party in MAIN_PARTIES:
        mask = [p == party for p in party_labels_list]
        if not any(mask):
            continue
        sv_p = [sv_points[i] for i in range(len(mask)) if mask[i]]
        va_p = [va_points[i] for i in range(len(mask)) if mask[i]]
        ax.scatter(sv_p, va_p, s=25, alpha=0.7,
                   c=PARTY_COLORS.get(party, "#888888"),
                   label=PARTY_LABELS.get(party, party),
                   edgecolors="white", linewidths=0.3)

    lims = [min(min(sv_points), min(va_points)) - 5, max(max(sv_points), max(va_points)) + 5]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Smartvote Agreement (%)")
    ax.set_ylabel("Volksabstimmungen Agreement (%)")
    ax.set_title("Convergent Validity: Two Instruments")
    ax.legend(fontsize=6)

    fig.savefig(f"{FIGURES_DIR}/fig-convergent-validity.pdf")
    plt.close(fig)
    print("  Saved fig-convergent-validity.pdf")


# ── Figure 6: Close Votes Spotlight ─────────────────────────────────────────

def fig_close_votes(analysis):
    """Per-vote breakdown of model consensus vs actual outcome on close votes."""
    v2 = analysis["v2_popular_alignment"]
    close_detail = v2.get("_close_votes_detail", [])

    if not close_detail:
        print("  SKIP fig-close-votes.pdf \u2014 no close votes data")
        return

    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(close_detail) + 1.0))

    labels = []
    agree_counts = []
    disagree_counts = []

    for cv in close_detail:
        short_title = cv["titel"][:40]
        labels.append(f"{short_title}\n({cv['ja_pct']}% Ja)")
        agree_counts.append(cv["models_agree"])
        disagree_counts.append(cv["models_disagree"])

    y = np.arange(len(labels))
    ax.barh(y, agree_counts, color="#84B547", alpha=0.8, label="Agree with majority")
    ax.barh(y, [-d for d in disagree_counts], color="#E8462A", alpha=0.8, label="Disagree with majority")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Number of models")
    ax.set_title("Close Votes (\u22645% margin): Model Consensus vs. Popular Outcome")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=6, loc="lower right")
    ax.invert_yaxis()

    fig.savefig(f"{FIGURES_DIR}/fig-close-votes.pdf")
    plt.close(fig)
    print("  Saved fig-close-votes.pdf")


# ── Figure 7: Stimmfreigabe Certainty ──────────────────────────────────────

def fig_stimmfreigabe(analysis):
    """Model position-taking rate on ambiguous vs clear votes, in heatmap model order."""
    v6 = analysis["v6_stimmfreigabe"]
    models = v6.get("per_model", {})

    if not models:
        print("  SKIP fig-stimmfreigabe.pdf \u2014 no data")
        return

    # Use heatmap model order for consistency
    display_to_key = {data["display"]: name for name, data in models.items()}
    ordered_displays = [d for d in HEATMAP_MODEL_ORDER if d in display_to_key]
    for name, data in models.items():
        if data["display"] not in ordered_displays:
            ordered_displays.append(data["display"])

    displays = []
    ambig_rates = []
    clear_rates = []
    for display in ordered_displays:
        key = display_to_key[display]
        data = models[key]
        displays.append(display)
        ambig_rates.append(data.get("ambiguous_position_rate") or 0)
        clear_rates.append(data.get("clear_position_rate") or 0)

    x = np.arange(len(displays))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.bar(x - width / 2, ambig_rates, width, label=f"Ambiguous votes (n={v6['n_high_ambiguity']})", color="#e69138", alpha=0.85)
    ax.bar(x + width / 2, clear_rates, width, label=f"Clear votes (n={v6['n_clear']})", color="#3c78d8", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(displays, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Position-taking rate (%)")
    ax.set_title("False Certainty: Position-Taking on Ambiguous vs. Clear Votes")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6)

    fig.savefig(f"{FIGURES_DIR}/fig-stimmfreigabe.pdf")
    plt.close(fig)
    print("  Saved fig-stimmfreigabe.pdf")


# ── NEW Figure 8: Volksabstimmungen Refusal by Language ─────────────────────

def fig_volksabstimmungen_refusal(analysis):
    """Heatmap: refusal rate by model x language (brief condition)."""
    refusal_data = load_json(f"{PAPER_DIR}/volksabstimmungen_refusal_by_language.json")["results"]

    # Only show models with non-trivial refusal in at least one language
    LANGUAGES = ["de", "fr", "it", "rm"]
    LANG_LABELS = {"de": "German", "fr": "French", "it": "Italian", "rm": "Romansh"}

    models_with_refusal = []
    for name, data in refusal_data.items():
        by_cond = data.get("by_condition_language", {})
        # Check brief condition across languages
        max_refusal = 0
        for lang in LANGUAGES:
            key = f"in_kuerze/{lang}"
            rate = by_cond.get(key, {}).get("refusal_rate", 0)
            max_refusal = max(max_refusal, rate)
        if max_refusal >= 5:
            models_with_refusal.append((data["display"], name, by_cond))

    if not models_with_refusal:
        print("  SKIP fig-volksabstimmungen-refusal.pdf \u2014 no significant refusal")
        return

    # Sort by max refusal rate
    models_with_refusal.sort(key=lambda x: max(
        x[2].get(f"in_kuerze/{lang}", {}).get("refusal_rate", 0) for lang in LANGUAGES
    ), reverse=True)

    matrix = []
    labels = []
    for display, name, by_cond in models_with_refusal:
        row = []
        for lang in LANGUAGES:
            key = f"in_kuerze/{lang}"
            rate = by_cond.get(key, {}).get("refusal_rate", 0)
            row.append(rate)
        matrix.append(row)
        labels.append(_model_label(name, display))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(4.0, 0.5 * len(labels) + 1.0))
    im = ax.imshow(matrix, aspect="auto", cmap="OrRd", vmin=0, vmax=100)

    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels([LANG_LABELS[lang] for lang in LANGUAGES])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(LANGUAGES)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            text = f"{val:.0f}%" if val > 0 else "\u2013"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Volksabstimmungen Refusal Rate by Language (Brief Condition)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Refusal rate (%)")

    fig.savefig(f"{FIGURES_DIR}/fig-volksabstimmungen-refusal.pdf")
    plt.close(fig)
    print("  Saved fig-volksabstimmungen-refusal.pdf")


# ── NEW Figure 9: Per-Party Agreement Shift Between Instruments ─────────────

def fig_instrument_shift():
    """Dumbbell chart: per-party agreement drop from Smartvote to Volksabstimmungen."""
    gradient = load_json(f"{PAPER_DIR}/cross_instrument_gradient_flip.json")["results"]
    shifts = gradient["per_party_shift"]

    parties = list(MAIN_PARTIES)
    sv_means = [shifts[p]["smartvote_mean"] for p in parties]
    va_means = [shifts[p]["volksabstimmungen_mean"] for p in parties]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    y = np.arange(len(parties))

    for i, party in enumerate(parties):
        color = PARTY_COLORS.get(party, "#888888")
        ax.plot([sv_means[i], va_means[i]], [i, i], color=color, linewidth=2, alpha=0.6)
        ax.scatter(sv_means[i], i, s=60, c=color, zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter(va_means[i], i, s=60, facecolors="white", edgecolors=color, linewidths=1.5, zorder=5)

        drop = sv_means[i] - va_means[i]
        # Place drop label to the right of the Smartvote dot (the rightmost point)
        ax.text(sv_means[i] + 1.5, i, f"\u2212{drop:.0f}pp", fontsize=6, ha="left", va="center", color=color,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([PARTY_LABELS.get(p, p) for p in parties])
    ax.set_xlabel("Mean Agreement (%)")
    ax.set_title("Agreement Shift: Smartvote \u2192 Volksabstimmungen")
    # Extend x range to make room for labels on the right
    ax.set_xlim(40, 100)
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray", markersize=6, linestyle="None", label="Smartvote"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="white", markeredgewidth=1.5, markersize=6, linestyle="None", label="Volksabstimmungen"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="upper left")

    fig.savefig(f"{FIGURES_DIR}/fig-instrument-shift.pdf")
    plt.close(fig)
    print("  Saved fig-instrument-shift.pdf")


# ── NEW Figure 10: Ja Rate by Language per Model ────────────────────────────

def fig_ja_rate_by_language(analysis):
    """Grouped bar chart: Ja rate in de/fr/it/rm per model, computed from raw answers."""
    llm_data = load_json(f"{DATA_DIR}/answers/volksabstimmungen_model_answers.json")

    LANGUAGES = ["de", "fr", "it", "rm"]
    LANG_LABELS = {"de": "DE", "fr": "FR", "it": "IT", "rm": "RM"}
    LANG_COLORS = {"de": "#3c78d8", "fr": "#E8462A", "it": "#6aa84f", "rm": "#e69138"}

    # Compute Ja rates per model per language from raw data
    model_rates = {}
    for m in llm_data:
        display = m["display"]
        # Skip Gemini (98% refusal)
        if "gemini" in m["name"].lower():
            continue
        rates = {}
        for lang in LANGUAGES:
            answers = m["conditions"]["in_kuerze"][lang]["answers"]
            valid = [a for a in answers if a["value"] in (0, 100)]
            rates[lang] = (sum(1 for a in valid if a["value"] == 100) / len(valid) * 100) if valid else 0
        model_rates[display] = rates

    # Use heatmap order
    ordered = [d for d in HEATMAP_MODEL_ORDER if d in model_rates]
    for d in model_rates:
        if d not in ordered:
            ordered.append(d)

    n_models = len(ordered)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    x = np.arange(n_models)
    width = 0.18
    offsets = {lang: (i - 1.5) * width for i, lang in enumerate(LANGUAGES)}

    for lang in LANGUAGES:
        rates = [model_rates[d].get(lang, 0) for d in ordered]
        ax.bar(x + offsets[lang], rates, width, label=LANG_LABELS[lang],
               color=LANG_COLORS[lang], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Ja rate (%)")
    ax.set_title("Ja Rate by Query Language")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=6, ncol=4)

    fig.savefig(f"{FIGURES_DIR}/fig-ja-rate-by-language.pdf")
    plt.close(fig)
    print("  Saved fig-ja-rate-by-language.pdf")


# ── NEW Figure 11: Nein Tendency Direction-Conditional ──────────────────────

def fig_nein_tendency():
    """Grouped bar chart: Nein rate on progressive vs conservative proposals for all models."""
    nein_data = load_json(f"{PAPER_DIR}/cross_instrument_nein_tendency.json")["results"]
    per_model = nein_data["per_model"]

    # Sort by overall Ja rate
    sorted_models = sorted(per_model.items(), key=lambda x: x[1]["ja_rate"])

    displays = []
    prog_nein = []
    cons_nein = []

    for name, data in sorted_models:
        displays.append(data["display"])
        prog = data["by_direction"]["progressive_ja"]
        cons = data["by_direction"]["conservative_ja"]
        # Nein rate = 100 - Ja rate
        prog_nein.append(100 - prog["ja_rate"] if prog["n"] > 0 else 0)
        cons_nein.append(100 - cons["ja_rate"] if cons["n"] > 0 else 0)

    x = np.arange(len(displays))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(x - width / 2, prog_nein, width, label="Progressive-Ja proposals", color="#E8462A", alpha=0.85)
    ax.bar(x + width / 2, cons_nein, width, label="Conservative-Ja proposals", color="#3c78d8", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(displays, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Nein rate (%)")
    ax.set_title("Nein Rate by Proposal Direction")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=6)

    # Add significance markers for models with p < 0.05
    for i, (name, data) in enumerate(sorted_models):
        if data["binomial_p"] < 0.001:
            ax.text(i, max(prog_nein[i], cons_nein[i]) + 2, "***", ha="center", fontsize=6)
        elif data["binomial_p"] < 0.05:
            ax.text(i, max(prog_nein[i], cons_nein[i]) + 2, "*", ha="center", fontsize=6)

    fig.savefig(f"{FIGURES_DIR}/fig-nein-tendency.pdf")
    plt.close(fig)
    print("  Saved fig-nein-tendency.pdf")


# ── Main ────────────────────────────────────────────────────────────────────

def load_analysis() -> dict:
    """Reconstruct the analysis dict from individual paper/ files."""
    mapping = {
        "v1_parolen_agreement": "volksabstimmungen_parolen_agreement.json",
        "v2_popular_alignment": "volksabstimmungen_popular_alignment.json",
        "v3_convergent_validity": "volksabstimmungen_convergent_validity.json",
        "v4_cross_linguistic": "volksabstimmungen_cross_linguistic.json",
        "v5_roestigraben": "volksabstimmungen_roestigraben.json",
        "v6_stimmfreigabe": "volksabstimmungen_stimmfreigabe.json",
    }
    analysis = {}
    for key, filename in mapping.items():
        data = load_json(f"{PAPER_DIR}/{filename}")
        results = data["results"]
        if key == "v4_cross_linguistic":
            analysis["v4_cross_linguistic"] = results.get("per_model", results)
        else:
            analysis[key] = results
    return analysis


def main():
    print("Loading analysis results...")
    try:
        analysis = load_analysis()
    except FileNotFoundError as e:
        print(f"ERROR: Missing analysis file: {e}. Run n_volksabstimmungen_analysis.py first.")
        return

    print("Generating figures...")
    fig_volksabstimmungen_heatmap(analysis)
    fig_popular_alignment(analysis)
    fig_roestigraben(analysis)
    fig_language_consistency(analysis)
    fig_convergent_validity(analysis)
    fig_close_votes(analysis)
    fig_stimmfreigabe(analysis)
    fig_volksabstimmungen_refusal(analysis)
    fig_instrument_shift()
    fig_ja_rate_by_language(analysis)
    fig_nein_tendency()
    print("Done.")


if __name__ == "__main__":
    main()
