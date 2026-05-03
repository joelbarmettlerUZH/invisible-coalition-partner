# The Invisible Coalition Partner

Replication code, data, and analysis pipeline for the paper:

> **The Invisible Coalition Partner: Auditing the Political Behaviour of Large Language Models Through Direct Democracy**
> Joel P. Barmettler, 2026.

This repository contains everything needed to reproduce every number, table, and figure in the paper. It also exposes the underlying datasets (LLM responses, parliament-member positions, federal referendum data) so other researchers can build on or contest our findings.

> **License:** Apache License 2.0. See [`LICENSE`](LICENSE).

---

## Table of contents

1. [What this study does](#what-this-study-does)
2. [Headline findings](#headline-findings)
3. [Repository layout](#repository-layout)
4. [Quick start: reproducing the analysis](#quick-start-reproducing-the-analysis)
5. [Pipeline scripts](#pipeline-scripts)
6. [Data sources](#data-sources)
7. [Data schemas](#data-schemas)
8. [Result files](#result-files)
9. [Models studied](#models-studied)
10. [Statistical methodology](#statistical-methodology)
11. [Practical notes for reproduction](#practical-notes-for-reproduction)
12. [Citing this work](#citing-this-work)

---

## What this study does

We audit nine flagship Large Language Models (LLMs) against Swiss democratic reality using **two independent instruments**:

1. **Smartvote**: a 75-question Voting Advice Application (VAA) used in the 2023 Swiss National Council election. We compare each model's positions to the answers of all 184 elected parliament members and the six main parties (SP, Grüne, GLP, Die Mitte, FDP, SVP).
2. **Volksabstimmungen**: 48 federal popular votes (referenda) decided by Swiss voters. For each vote we have multilingual texts, official party Parolen (recommendations), and actual cantonal results. We query each model on every vote in **four languages** (German, French, Italian, Romansh) under **three detail conditions** (~5,184 calls).

Three research questions:

1. **Do abstract VAAs predict how LLMs behave on concrete policy decisions?**
2. **Does the language of a political question change the answer?**
3. **Do LLMs represent the popular will?**

The Volksabstimmungen extension is, to our knowledge, the first audit of LLM political behaviour against a population of binding popular votes (rather than constructed surveys), and the first to systematically test query-language effects on political output across the four official languages of a country.

---

## Headline findings

| # | Finding | Confidence |
|---|---------|------------|
| 1 | The left-to-right party-agreement gradient on Smartvote **reverses** on Volksabstimmungen. Left-party agreement collapses; centre-right stays stable. | BULLETPROOF |
| 2 | Grok and Mistral exhibit a systematic **Nein bias** on Volksabstimmungen (status-quo aversion, not a left-right effect). | BULLETPROOF |
| 3 | Popular-vote alignment ranges from ~60% to ~98% across models; significantly heterogeneous. | BULLETPROOF |
| 4 | Cross-linguistic consistency ranges from ~50% to ~98%. Significant pairwise language effects only for Llama and Mistral after BH correction. No significant Röstigraben (DE-FR voting-gap) correlation. | STRONG |
| 5 | Gemini's refusal rate is sharply context- and language-dependent (98% in the brief German condition, ~9% with full French context). | BULLETPROOF |
| 6 | Detail level of the prompt does not systematically shift positions; consistency is 81-96%. | SOLID |
| 7 | Most models almost always commit to a position; only DeepSeek shows substantial Stimmfreigabe-like behaviour on ambiguous votes. | DESCRIPTIVE |

Every quantitative claim, p-value, effect size, and confidence interval in the paper is derived from a JSON file under `results/paper/`. There are no hardcoded numbers in the paper text that are not also present in those files.

---

## Repository layout

```
.
├── a_scrape_smartvote_questions.py     # Pipeline scripts (alphabetical = execution order)
├── b_scrape_smartvote_answers.py
├── c_generate_smartvote_dataset.py
├── d_create_smartvote_plots.py
├── e_smartvote_web_pca.py
├── f_smartvote_web_model_pca.py
├── g_smartvote_web_table.py
├── h_smartvote_web_agreement.py
├── i_smartvote_statistical_analysis.py
├── j_smartvote_paper_figures.py
├── k_scrape_volksabstimmungen.py
├── k_web_blog.py
├── l_generate_volksabstimmungen_dataset.py
├── m_volksabstimmungen_pca.py
├── n_volksabstimmungen_analysis.py
├── o_paper_volksabstimmungen_figures.py
├── p_extended_analysis.py
├── _volksabstimmungen_constants.py     # Shared constants (party names, language keywords, ...)
│
├── data/                               # Raw / scraped inputs (committed)
│   ├── questionnaire/questionnaire.json
│   ├── answers/
│   │   ├── nationalrat_members.json            # 184 elected MPs with Smartvote answers
│   │   ├── all_model_answers.json              # 69 LLMs × 75 questions
│   │   └── volksabstimmungen_model_answers.json  # 9 LLMs × 48 votes × 4 langs × 3 conditions
│   ├── volksabstimmungen/
│   │   └── volksabstimmungen.json              # 50 federal referenda (48 used)
│   └── model_families.json             # Family/flagship/drift-pair definitions
│
├── results/                            # Analysis outputs (committed)
│   ├── paper/                          # Per-test JSON files cited in the paper
│   ├── website/                        # JSON files consumed by the public-facing web visualisations
│   ├── volksabstimmungen_pca/          # PCA projections for Volksabstimmungen
│   └── README.md                       # Per-file documentation of the results directory
│
├── pyproject.toml                      # uv / pip project config (Python ≥ 3.13)
├── uv.lock                             # Pinned dependency versions
├── .env.example                        # Template; copy to .env and add your OpenRouter key
├── LICENSE                             # Apache 2.0
└── README.md                           # (this file)
```

> The `paper/` LaTeX sources are *not* included in this repository; this repo is the open-source code/data release. The compiled PDF will be available via arXiv.

---

## Quick start: reproducing the analysis

The repository ships with **all intermediate outputs already committed**, so you can reproduce every paper figure and statistic from the existing data without re-querying any LLM.

### Prerequisites

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip` + `venv`
- LaTeX (only if you want to recompile the paper itself; not in this repo)

### Install

```bash
# Clone, then:
uv sync          # creates .venv and installs pinned dependencies from uv.lock
```

If you do not use `uv`, you can install from `pyproject.toml` with pip:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Reproduce all statistical results from existing data (no API calls)

```bash
uv run python e_smartvote_web_pca.py                 # Fit Smartvote PCA on parliament members
uv run python f_smartvote_web_model_pca.py --batch 2025
uv run python f_smartvote_web_model_pca.py --batch 2026
uv run python g_smartvote_web_table.py --batch 2025
uv run python g_smartvote_web_table.py --batch 2026
uv run python h_smartvote_web_agreement.py --batch 2025
uv run python h_smartvote_web_agreement.py --batch 2026
uv run python i_smartvote_statistical_analysis.py    # All Smartvote paper tests
uv run python k_web_blog.py                          # Web blog data (timeline, refusal, heatmap)
uv run python m_volksabstimmungen_pca.py             # PCA on Volksabstimmungen Parolen
uv run python n_volksabstimmungen_analysis.py        # All Volksabstimmungen paper tests
uv run python p_extended_analysis.py                 # Cross-instrument tests + unified BH correction
```

This regenerates everything under `results/paper/`, `results/website/`, and `results/volksabstimmungen_pca/`. Compare against the committed files; numbers should match bit-for-bit (all RNG seeds are fixed).

### Re-collect raw data from scratch (requires API keys + several hours)

Only needed if you want to query a *different* set of models, repeat the experiment with different parameters, or audit the data collection itself.

```bash
# 1. Smartvote questionnaire + parliament-member answers (no API key required)
uv run python a_scrape_smartvote_questions.py
uv run python b_scrape_smartvote_answers.py

# 2. Volksabstimmungen scrape (no API key required)
uv run python k_scrape_volksabstimmungen.py

# 3. LLM queries (requires OPENAI_API_KEY in .env, billed via OpenRouter)
uv run python c_generate_smartvote_dataset.py --batch 2026
uv run python l_generate_volksabstimmungen_dataset.py
```

Both LLM scripts are **incremental and resumable**: they append to the consolidated answers file and skip questions that already have a recorded response, so you can interrupt and restart.

---

## Pipeline scripts

Scripts are alphabetised so the execution order is unambiguous. Lower letters produce inputs for higher letters.

### Smartvote (`a` – `j`, plus `k_web_blog`)

| Script | Purpose | Reads | Writes |
|--------|---------|-------|--------|
| `a_scrape_smartvote_questions.py` | Scrape the 75-question Smartvote 2023 questionnaire (14 categories, 13 Standard + 1 Budget) from the public Smartvote GraphQL API. | Smartvote API | `data/questionnaire/questionnaire.json` |
| `b_scrape_smartvote_answers.py` | Scrape all 184 elected National Council members with full demographics, party, district, and Smartvote answers. | Smartvote API | `data/answers/nationalrat_members.json` |
| `c_generate_smartvote_dataset.py --batch 2025\|2026` | Query each LLM on every Smartvote question via OpenRouter. System prompt forces one of *Ja / Eher Ja / Eher Nein / Nein* (mapped to 100/75/25/0; refusals = -1). Incremental and resumable. | questionnaire + OpenRouter API | `data/answers/all_model_answers.json` (appended) |
| `d_create_smartvote_plots.py` | Exploratory plots: 2D PCA scatter, agreement boxplots, spider/radar, 1D spectrum, correlation heatmaps. Opens matplotlib windows; not used for paper figures. | all data files | (interactive) |
| `e_smartvote_web_pca.py` | Fit a PCA on the 184-MP × 60-question matrix (BudgetCategory excluded). Saves the fitted model and the projected MP coordinates. **Run only once**; downstream scripts project models into this fixed space. | questionnaire + nationalrat_members | `results/website/smartvote_pca_{1d,2d}.json` |
| `f_smartvote_web_model_pca.py --batch 2025\|2026` | Project the LLMs of a given batch into the fixed PCA space and merge with the parliament-member projections. | all_model_answers + smartvote_pca | `results/website/smartvote_combined_{1d,2d}[_2026].json` |
| `g_smartvote_web_table.py --batch 2025\|2026` | For every question, build a row comparing the party-mean answer with each LLM's answer. Drives the interactive comparison table on the website. | all data files | `results/website/smartvote_comparison[_2026].json` |
| `h_smartvote_web_agreement.py --batch 2025\|2026` | Compute squared-difference-based agreement scores between each LLM and each party. | all data files | `results/website/smartvote_agreement[_2026].json` |
| `i_smartvote_statistical_analysis.py` | All paper-quality Smartvote tests: PCA validation, permutation tests for displacement, Kruskal-Wallis for geographic effects, open-vs-closed permutation, sign tests on temporal drift pairs, refusal rates, category profiles, effect sizes, BH correction. | all data files + PCA | 13 files in `results/paper/smartvote_*.json` |
| `j_smartvote_paper_figures.py` | Publication-quality PDFs: 2D + 1D PCA, agreement heatmap, drift arrows, timeline, refusal, category profiles. Projects models on-the-fly into the PCA space (independent of `f_*` outputs). | PCA + all_model_answers + model_families | `paper/figures/*.pdf` (figures consumed by paper, not in this repo) |
| `k_web_blog.py` | Blog-specific website data: release-date timeline, per-model refusal bar chart, flagship × flagship agreement heatmap. | all data files + PCA | `results/website/{timeline_data,refusal_data,agreement_heatmap}.json` |

### Volksabstimmungen (`k` – `o`)

| Script | Purpose | Reads | Writes |
|--------|---------|-------|--------|
| `k_scrape_volksabstimmungen.py` | Scrape the last 50 federal referenda from the public Zurich-statistics archive: multilingual texts (de/fr/it/rm), party Parolen, national + cantonal results. | Zurich statistics API | `data/volksabstimmungen/volksabstimmungen.json` |
| `_volksabstimmungen_constants.py` | Shared constants: party-name normalisation (`glp`→`GLP`, `CVP`→`Die Mitte`, ...), canton language classification, response keywords per language, system/user prompts for each detail condition. | — | (module) |
| `l_generate_volksabstimmungen_dataset.py` | Query each of 9 flagship models on each of 48 usable votes × 4 languages × 3 detail conditions (~5,184 calls). Binary Ja/Nein parsed via language-specific keyword sets. Incremental and resumable. `--dry-run` prints prompts without calling the API. | volksabstimmungen.json + OpenRouter | `data/answers/volksabstimmungen_model_answers.json` |
| `m_volksabstimmungen_pca.py` | Fit a PCA on the 6-party × 48-vote Parolen matrix and project LLM vote vectors. Used primarily for the website visualisation; statistical claims do not rely on this projection. | volksabstimmungen.json + answers | `results/volksabstimmungen_pca/pca_{1d,2d}.json` |
| `n_volksabstimmungen_analysis.py` | All paper-quality Volksabstimmungen tests: Parolen agreement, popular-vote alignment by margin, cross-linguistic consistency + McNemar pairwise tests, Röstigraben correlation, prompt-sensitivity, Stimmfreigabe behaviour, Bundesrat agreement, model convergence permutation, refusal-by-language, party-by-language shifts, temporal split. | model_answers + volksabstimmungen + Smartvote results | 14 files in `results/paper/volksabstimmungen_*.json` |
| `o_paper_volksabstimmungen_figures.py` | Publication-quality PDFs for the Volksabstimmungen section. | `results/paper/volksabstimmungen_*.json` | `paper/figures/fig-volksabstimmungen-*.pdf` |

### Cross-instrument (`p`)

| Script | Purpose | Reads | Writes |
|--------|---------|-------|--------|
| `p_extended_analysis.py` | Five blocks: (1) systematic Nein-tendency binomials, (2) instrument divergence / gradient-flip Wilcoxon test, (3) Smartvote PC1 vs Volksabstimmungen Ja-rate Spearman, (4) model-vs-party convergence permutation, (5) **unified BH correction across all 28 tests** from `i_*`, `n_*`, and this script. | all `results/paper/*.json` + raw data | 4 × `results/paper/cross_instrument_*.json` + `results/paper/bh_correction.json` |

---

## Data sources

All raw inputs are scraped from public sources. No proprietary data is included.

| Source | Scraper | Notes |
|--------|---------|-------|
| **Smartvote 2023 National Council questionnaire** (smartvote.ch GraphQL) | `a_*.py` | 75 questions in 14 categories. Election ID `1057`. |
| **Smartvote 2023 candidate answers** | `b_*.py` | All 184 elected National Council members with full profile data. |
| **Federal Volksabstimmungen archive** (`app.statistik.zh.ch`) | `k_*.py` | Last 50 federal popular votes. Multilingual texts, official party Parolen, cantonal-level results. Two votes excluded from analysis: a Stichfrage (run-off question) and a Direkter Gegenentwurf (counterproposal). |
| **LLM responses** (OpenRouter) | `c_*.py`, `l_*.py` | Queried with `temperature=0.0`, `seed=42`, identical system prompts within each experiment. See [Models studied](#models-studied). |

### Reproducibility caveats

- **LLM responses are not bit-stable across model checkpoints.** Even with `temperature=0.0` and `seed=42`, providers occasionally update model snapshots without renaming them. The committed `data/answers/*.json` files reflect the snapshots available between March and April 2026. Re-running `c_*` or `l_*` later may produce slightly different numbers.
- **Smartvote and Volksabstimmungen scrapers** depend on third-party APIs; expect small schema drift over time. Adjust the scraper if needed.
- All statistical scripts read from the committed JSON files and are deterministic.

---

## Data schemas

All data files are UTF-8 JSON. Field-level documentation follows.

### `data/questionnaire/questionnaire.json`

The full Smartvote 2023 questionnaire as returned by the GraphQL API. Top-level structure:

```json
{
  "election": {
    "id": "1057",
    "questionnaire": {
      "id": "...",
      "nofQuestions": 75,
      "categories": [
        {
          "id": "...", "categoryId": "...", "name": "Wirtschaft", "type": "Standard",
          "description": "...", "sortorder": 1,
          "questions": [
            {
              "id": "32218", "type": "STANDARD",
              "label": "Sind Sie für die Einführung einer Mindestlohnpflicht ...?",
              "options": [{"value": 0, "label": "Nein"}, {"value": 25, "label": "Eher Nein"}, ...]
            }
          ]
        }
      ]
    }
  }
}
```

13 Standard categories + 1 BudgetCategory (the latter is excluded from PCA).

### `data/answers/nationalrat_members.json`

Array of 184 elected National Council members.

```json
[{
  "id": "55674",
  "firstname": "Jean-Luc", "lastname": "Addor",
  "yearOfBirth": 1964, "gender": "MALE", "city": "Sion", "country": "CH",
  "isElected": true, "isIncumbent": true,
  "partyAbbreviation": "SVP", "partyColor": "#4B8A3E",
  "party": {"name": "Schweizerische Volkspartei", "abbreviation": "SVP", "color": "#4B8A3E"},
  "district": {"name": "Wallis"},
  "answers": [
    {"questionId": "32218", "value": 0, "weight": 100}
  ]
}]
```

`value` is on the 0/25/75/100 scale; missing answers are simply absent from the array.

### `data/answers/all_model_answers.json`

Array of LLM response sets (currently 69 entries: 9 flagships × 2 batches + drift / deep-series models).

```json
[{
  "name": "openai/gpt-5.4",                 // OpenRouter model ID — primary key
  "display": "GPT-5.4",                     // Human-readable name
  "batch": "2026",                          // "2025" | "2026" | "deep-series" | "grok-series"
  "family": "openai-gpt",                   // Stable family ID for drift / timeline
  "predecessor": "openai/gpt-4o-2024-11-20",// Drift partner (or null)
  "country": "USA", "continent": "North America", "provider": "OpenAI",
  "released": "2026-03-05",                 // Model release date (NOT experiment timestamp)
  "size": null,                             // Active params (nullable)
  "reasoning": false, "open_source": false,
  "temperature": 0.0, "seed": 42,           // Sampling parameters used for *every* call
  "answers": [
    {
      "questionId": "32218",
      "raw_value": "Eher Ja",               // Verbatim parsed token from model output
      "value": 75,                          // 100=Ja, 75=Eher Ja, 25=Eher Nein, 0=Nein, -1=refused
      "usage": {"prompt_tokens": 888, "completion_tokens": 2}
    }
  ]
}]
```

### `data/answers/volksabstimmungen_model_answers.json`

```json
[{
  "name": "openai/gpt-5.4",
  "display": "GPT-5.4",
  "country": "USA", "continent": "North America",
  "open_source": false, "reasoning": false,
  "temperature": 0.0, "seed": 42,
  "conditions": {
    "in_kuerze": {                           // Brief summary only
      "de": {"answers": [
        {"vorlagenId": 6380, "raw_value": "Ja", "value": 100,
         "usage": {"prompt_tokens": 500, "completion_tokens": 2}}
      ]},
      "fr": {"answers": [...]}, "it": {"answers": [...]}, "rm": {"answers": [...]}
    },
    "in_kuerze_im_detail": {                 // Brief summary + factual background
      "de": {"answers": [...]}, "fr": {...}, "it": {...}, "rm": {...}
    },
    "full_text": {                           // All chapters incl. "Warum Ja"/"Warum Nein"
      "de": {"answers": [...]}, "fr": {...}, "it": {...}, "rm": {...}
    }
  }
}]
```

`value` semantics: `100 = Ja`, `0 = Nein`, `-1 = refused/unparseable`.

### `data/volksabstimmungen/volksabstimmungen.json`

Array of 50 federal referenda (the analysis filters out 2: a Stichfrage and a Direkter Gegenentwurf — see `EXCLUDED_TITLES` in `_volksabstimmungen_constants.py`).

```json
[{
  "vorlagenId": 6380,
  "abstimmtag": "2021-03-07",
  "titel": "Volksinitiative «Ja zum Verhüllungsverbot»",
  "titles": {"de": "...", "fr": "...", "it": "...", "rm": "..."},
  "resultat": {
    "jaStimmenInProzent": 51.19,
    "jaStimmenAbsolut": 1427344,
    "neinStimmenAbsolut": 1360750,
    "stimmbeteiligungInProzent": 51.40,
    "anzahlStimmberechtigte": 5498611
  },
  "kantone": [
    {"geoLevelnummer": 1, "name": "Kanton Zürich", "jaStimmenInProzent": 45.14, ...}
  ],
  "parolen": {
    "parties":  [{"name": "SP",        "parole": "Ja",  "color": "#E8462A"}],
    "councils": [{"name": "Bundesrat", "parole": "Nein"}]
  },
  "texts": {
    "de": [
      {"title": "In Kürze",  "content": ["paragraph1", "paragraph2"]},
      {"title": "Im Detail", "content": [...]},
      {"title": "Warum Ja",  "content": [...]},
      {"title": "Warum Nein","content": [...]}
    ],
    "fr": [{"title": "En bref", ...}, {"title": "En détail", ...}, {"title": "Pour", ...}, {"title": "Contre", ...}],
    "it": [{"title": "In breve", ...}, ...],
    "rm": [{"title": "Curtamain", ...}, {"title": "Detagls", ...}, {"title": "Per", ...}, {"title": "Cunter", ...}]
  }
}]
```

Parole values seen: `Ja`, `Nein`, `Stimmfreigabe`, `keine Angabe`, `keine Empfehlung`, plus `A`/`B` (Stichfragen only — excluded). Party names are normalised before analysis (`glp`→`GLP`, `GRÜNE`→`Grüne`, `CVP`→`Die Mitte`).

### `data/model_families.json`

```json
{
  "openai-gpt": {
    "display": "OpenAI GPT",
    "flagship": "openai/gpt-5.4",
    "drift": ["openai/gpt-3.5-turbo", "openai/gpt-5.4"],
    "note": "Flagship chat line: GPT-3.5 Turbo (May 2023) → GPT-5.4 (Mar 2026)"
  }
}
```

`flagship` identifies the newest member of a family used in cross-sectional figures. `drift` is the (oldest, newest) pair used for temporal-drift analysis; `null` when comparison would be unfair (e.g. dense vs. MoE).

### `results/paper/*.json` schema convention

Every file under `results/paper/` is **self-documenting** with the following metadata fields, plus a `results` payload that varies by test:

```json
{
  "description": "...",
  "generated_by": "i_smartvote_statistical_analysis.py",
  "generated_at": "2026-04-04T19:23:11+00:00",
  "models_analyzed": ["openai/gpt-5.4", ...],
  "models_excluded": ["google/gemini-3.1-pro-preview"],
  "n_models": 8,
  "methodology": "Squared-difference-based party-model agreement, ...",
  "results": { ... }
}
```

The full per-file documentation is in [`results/README.md`](results/README.md).

### `results/website/*.json`

Drives the public web visualisations on the project page. Schemas are bespoke per visualisation (PCA scatter, 1D spectrum, agreement bar chart, comparison table). The `results/README.md` lists which file feeds which Vue component, in case you want to embed similar visualisations.

---

## Result files

### `results/paper/`

| File | Test |
|------|------|
| `smartvote_pca_validation.json` | Explained variance, silhouette score, Spearman with party ordering |
| `smartvote_displacement.json` | Permutation test for centre-left displacement of the LLM cluster |
| `smartvote_geographic_effect.json` | Kruskal-Wallis: country / continent vs. PC1 |
| `smartvote_open_vs_closed.json` | Permutation test: open-source vs. closed-source positioning |
| `smartvote_temporal_drift.json` | Sign test on 16 predecessor-successor pairs |
| `smartvote_refusal_rates.json` | Per-model refusal rates over 75 Smartvote questions |
| `smartvote_category_profiles.json` | Per-category mean answer values for flagship models |
| `smartvote_deep_timeseries.json` | PC1 across model-family versions (deep series) |
| `smartvote_imputation_sensitivity.json` | Sensitivity of displacement test to refusal-imputation strategy |
| `smartvote_effect_sizes.json` | Cohen's *d* and η² for all Smartvote tests |
| `smartvote_agreement_scores.json` | Party-model agreement matrix (squared-difference metric) |
| `smartvote_agreement_robustness.json` | Robustness checks for the agreement metric |
| `smartvote_bh_correction.json` | BH correction within the Smartvote test family |
| `volksabstimmungen_parolen_agreement.json` | Model-party agreement on 48 referenda (incl. per-model Ja/Nein counts) |
| `volksabstimmungen_popular_alignment.json` | Alignment with popular vote by margin bucket; temporal split |
| `volksabstimmungen_cross_linguistic.json` | 4-language consistency rates + McNemar pairwise tests |
| `volksabstimmungen_roestigraben.json` | Actual DE-FR cantonal voting gap vs. model DE-FR prompt-answer gap |
| `volksabstimmungen_prompt_sensitivity.json` | Consistency across the 3 detail conditions |
| `volksabstimmungen_stimmfreigabe.json` | Model behaviour on ambiguous vs. clear votes |
| `volksabstimmungen_bundesrat.json` | Agreement with the Federal Council recommendation |
| `volksabstimmungen_model_convergence.json` | Inter-model vs. inter-party similarity |
| `volksabstimmungen_refusal_by_language.json` | Refusal rates by language and condition (Gemini focus) |
| `volksabstimmungen_consensus.json` | Per-vote model consensus and split-vote inventory |
| `volksabstimmungen_convergent_validity.json` | Smartvote vs. Volksabstimmungen agreement correlation |
| `volksabstimmungen_party_agreement_by_language.json` | Party-agreement shift by query language |
| `volksabstimmungen_temporal.json` | Agreement patterns by referendum year |
| `volksabstimmungen_bh_correction.json` | BH correction within the Volksabstimmungen test family |
| `cross_instrument_gradient_flip.json` | Left-right agreement-gradient reversal between instruments (Wilcoxon) |
| `cross_instrument_nein_tendency.json` | Systematic Nein/status-quo bias (binomial tests, Grok + Mistral) |
| `cross_instrument_convergent_validity.json` | Smartvote PC1 vs. Volksabstimmungen Ja-rate (Spearman) |
| `cross_instrument_convergence_permutation.json` | Models more similar to each other than to parties |
| `bh_correction.json` | **Unified BH correction across all 28 tests** (the global α-control file) |

### `results/website/`

JSON files that power the interactive visualisations on the public-facing project page:

| File | Visualisation |
|------|---------------|
| `smartvote_pca_{1d,2d}.json` | Fitted PCA model + parliament-member projections |
| `smartvote_combined_{1d,2d}[_2026].json` | Politicians + LLMs in PCA space (per batch) |
| `smartvote_agreement[_2026].json` | LLM-party agreement bar charts |
| `smartvote_comparison[_2026].json` | Per-question table |
| `timeline_data.json` | Release date vs. PC1 |
| `refusal_data.json` | Per-model refusal rates |
| `agreement_heatmap.json` | Flagship × flagship agreement heatmap |

### `results/volksabstimmungen_pca/`

| File | Description |
|------|-------------|
| `pca_1d.json` | 1D PCA of party Parolen + LLM vote-vector projections |
| `pca_2d.json` | 2D variant of the above |

---

## Models studied

### Nine flagships (primary analysis set)

The same nine models are used in both Smartvote and Volksabstimmungen experiments. Gemini is **excluded from Volksabstimmungen analyses** (98% refusal rate in German), leaving 8 usable models there.

| Model | Provider | Country | Open / Closed | Released |
|-------|----------|---------|---------------|----------|
| `openai/gpt-5.4` | OpenAI | USA | Closed | 2026-03 |
| `anthropic/claude-opus-4.6` | Anthropic | USA | Closed | 2026 |
| `google/gemini-3.1-pro-preview` | Google | USA | Closed | 2026 |
| `deepseek/deepseek-v3.2` | DeepSeek | China | Open | 2026 |
| `meta-llama/llama-4-maverick` | Meta | USA | Open | 2026 |
| `x-ai/grok-4.20` | xAI | USA | Closed | 2026 |
| `mistralai/mistral-large-2512` | Mistral | France | Closed | 2025-12 |
| `qwen/qwen3.5-plus-02-15` | Alibaba | China | Closed | 2026-02 |
| `cohere/command-a` | Cohere | Canada | Closed | 2026 |

5 countries (USA, China, France, Canada, varied), 2 open / 7 closed. All queried via OpenRouter with `temperature=0.0`, `seed=42`.

### Drift models

Older models from OpenAI, Anthropic, Mistral, and xAI families have **Smartvote data only** and are used solely for temporal-drift analysis (how a single provider's political position evolves across model generations). See `data/model_families.json` for pairing rules; pairs are `null` when comparison would be unfair (size class, dense vs. MoE active params, vision vs. text).

### Prompt designs

- **Smartvote**: German, 4-point scale (Ja / Eher Ja / Eher Nein / Nein → 100/75/25/0). System prompt forces one of four exact tokens.
- **Volksabstimmungen**: Binary Ja/Nein in 4 languages (de/fr/it/rm) × 3 detail conditions:
  1. *In Kürze* — brief summary only (primary condition)
  2. *In Kürze + Im Detail* — summary + factual background
  3. *Full text* — everything, including the partisan "Warum Ja" / "Warum Nein" arguments

48 votes (1 Stichfrage + 1 Direkter Gegenentwurf excluded). ~5,184 API calls for one full sweep.

The exact prompt strings are in `_volksabstimmungen_constants.py` (`SYSTEM_PROMPTS`, `USER_PROMPT_SUFFIX`) and in the prompt-construction functions of `c_generate_smartvote_dataset.py`.

---

## Statistical methodology

A high-level summary; see the individual `results/paper/*.json` files for exact test statistics.

- **PCA validation** (Smartvote): explained variance, silhouette score on parties as ground-truth clusters, Spearman correlation between PC1 ordering and the canonical SP→SVP party ordering. Sign of PC1 is fixed for display so that left-wing parties appear on the left; this is purely a display convention and does not affect any statistic.
- **Displacement** (Smartvote): permutation test (10 000 reshuffles) checking whether the LLM cloud is significantly displaced toward the centre-left half of MP-space.
- **Geographic / open-vs-closed effects** (Smartvote): Kruskal-Wallis and permutation tests on PC1, conditional on country / continent / open-source flag.
- **Temporal drift** (Smartvote): sign test on 16 predecessor → successor model pairs from `model_families.json`.
- **Parolen agreement** (Volksabstimmungen): exact-match agreement between LLM Ja/Nein and each party's official Parole, Stimmfreigabe excluded from the denominator.
- **Popular alignment** (Volksabstimmungen): per-margin-bucket alignment with the popular outcome.
- **Cross-linguistic consistency**: per-vote modal answer across {de, fr, it, rm}, McNemar tests on every language pair, Röstigraben correlation between the actual DE-FR cantonal voting gap and the model's DE-FR answer gap.
- **Cross-instrument convergence**: permutation test that LLMs are pairwise more similar to each other than parties are pairwise to each other.
- **Multiple-testing correction**: a **unified Benjamini-Hochberg correction across all 28 paper tests** is computed in `p_extended_analysis.py` and stored in `results/paper/bh_correction.json`. Per-family corrections (`smartvote_bh_correction.json`, `volksabstimmungen_bh_correction.json`) are also provided.

All RNG seeds are fixed (`numpy` and `python` `random`) so every test is bit-stable on the committed data.

---

## Practical notes for reproduction

- **Python 3.13 only.** The project uses match statements, `Annotated`-style typing, and recent stdlib APIs that may not back-port cleanly.
- **Dependencies are pinned in `uv.lock`.** Use `uv sync` to install exactly the versions used to generate the committed results.
- **API keys.** Only `c_*` and `l_*` need an API key (for re-collection). Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your OpenRouter key (`sk-or-v1-...`). The OpenAI Python SDK is configured against the OpenRouter base URL inside the scripts, so it accepts any OpenRouter key transparently.
- **Cost.** A full re-run of `l_generate_volksabstimmungen_dataset.py` queries 9 models × 48 votes × 4 languages × 3 conditions ≈ 5 184 calls. Token use is dominated by the *full_text* condition. As a rough order of magnitude expect tens of dollars on OpenRouter pricing as of 2026.
- **Re-running scrapers.** `a_*`, `b_*`, `k_*` (the Volksabstimmungen scrape) hit public APIs with low rate limits. Be polite; they are not parallelised on purpose.
- **PCA stability.** `e_smartvote_web_pca.py` is run once to fix the PCA basis. Do not regenerate it casually — every downstream figure that references absolute PC1 / PC2 coordinates assumes that exact basis.
- **Sign of PC1.** PCA components are sign-invariant. Figure scripts negate PC1 so that SP appears on the left. Numerical statistics derive only from rank or distance properties and are unaffected.
- **No mocked LLMs.** Every "model answer" in the dataset comes from a real provider call. The committed JSON is the ground truth; nothing is synthetic.

---

## Citing this work

If you use this code, data, or any committed result, please cite:

```bibtex
@misc{barmettler2026invisiblecoalition,
  title        = {The Invisible Coalition Partner: Auditing the Political
                  Behaviour of Large Language Models Through Direct Democracy},
  author       = {Barmettler, Joel P.},
  year         = {2026},
  eprint       = {arXiv:TBD},
  primaryClass = {cs.CL},
  url          = {https://github.com/joelbarmettlerUZH/invisible-coalition-partner}
}
```

Questions, corrections, and replication issues are welcome via GitHub issues.

---

*This release contains every script, raw dataset, and intermediate result needed to recompute every number and figure in the paper. The paper text, LaTeX sources, and compiled PDF are not part of this repository; they are distributed via arXiv.*
