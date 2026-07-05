# LLM Political Bias

## Writing style rules for the paper

- **No emdashes.** Never use `---` (emdashes) in paper.tex. Use semicolons, colons, parentheses, or restructure the sentence instead.
- **No "not just".** Never use the phrase "not just" in paper.tex. Rephrase to avoid it.

## Paper location, format & build

The manuscript lives at `paper/paper.tex`, in the **official ACL template** (`acl.sty` + `acl_natbib.bst`), targeting **EACL 2027 via ACL Rolling Review** (ARR submission deadline 3 Aug 2026). Build with `tectonic -X compile paper.tex` (no system LaTeX needed).

- The `[review]` option in the preamble anonymizes the author block and adds line numbers. Switch to `[final]` for camera-ready, and restore the real `\author{}` plus the commented-out Acknowledgments.
- **Page limit:** long-paper content (Introduction through Conclusion) must fit **8 pages**. Limitations, Ethics Statement, Data/Code Availability, References, and Appendices A/B are exempt and follow the Conclusion. Keep counted content within page 8 when editing.
- Secondary figures live in **Appendix B** (Supplementary Figures) and detailed statistics in **Appendix A**; the body keeps only the core RQ1 figures (PCA, agreement heatmaps, convergent-validity, instrument-shift).
- Citations use natbib: `\citet` (textual), `\citep` (parenthetical), `\citeposs` (possessive). Proper nouns/acronyms in `references.bib` titles are brace-protected so `acl_natbib` does not lowercase them.
- The public replication repo is `github.com/joelbarmettlerUZH/invisible-coalition-partner`; the review PDF instead points to an anonymized mirror. Do **not** update the root `README.md` title while under anonymous review (it would re-link the anonymized paper to the public repo).

Analyzes political positioning in large language models using two independent instruments grounded in Swiss democratic reality: (1) the Swiss Smartvote questionnaire (2023 National Council election) comparing LLM positions to 184 elected parliament members, and (2) 48 federal Volksabstimmungen (popular votes) comparing LLM positions to actual referendum outcomes and party Parolen. A cross-linguistic experiment exploits Switzerland's four national languages (de/fr/it/rm) to test whether query language shifts model positions. Results feed both an interactive website and a paper ("Progressive in Principle, Centrist in Practice: LLM Political Bias Is Instrument-Dependent"; the "invisible coalition partner" metaphor is retained as internal framing).

## Paper: "Progressive in Principle, Centrist in Practice"

### Central thesis

Abstract political questionnaires — the basis of all prior LLM political bias research — measure something different from how models actually behave when confronted with real policy decisions. The "leftward bias" found on Smartvote (and by all prior work using PCT, Wahl-O-Mat, etc.) does not replicate on Volksabstimmungen: when models receive actual policy summaries with real arguments for and against, they become more centrist and status-quo-favoring. A new instability emerges: for some models, the language of the question changes the answer more than the political content does.

### Three research questions

1. **Do abstract political questionnaires predict how LLMs behave on concrete policy decisions?** (Smartvote vs. Volksabstimmungen comparison — answer: no, the left-to-right agreement gradient reverses between instruments)
2. **Does the language of a political question change the answer?** (Cross-linguistic experiment — answer: dramatically for some models, not at all for others)
3. **Do LLMs represent the popular will?** (Popular vote alignment — answer: highly model-dependent; two models show significant systematic Nein tendency)

For detailed statistics, see `results/paper/` JSON files.

### Headline findings

Seven main findings, each with a confidence tag (BULLETPROOF / STRONG / SOLID / DESCRIPTIVE). For specific numbers, p-values, and effect sizes, see:

- `results/paper/cross_instrument_gradient_flip.json` — left-to-right agreement gradient reversal between instruments
- `results/paper/cross_instrument_nein_tendency.json` — systematic Nein/status-quo bias (Grok, Mistral)
- `results/paper/volksabstimmungen_popular_alignment.json` — popular vote alignment by model and margin
- `results/paper/volksabstimmungen_cross_linguistic.json` — language sensitivity and McNemar pairwise tests
- `results/paper/volksabstimmungen_refusal_by_language.json` — Gemini refusal patterns by language/context
- `results/paper/volksabstimmungen_prompt_sensitivity.json` — context/detail condition consistency
- `results/paper/volksabstimmungen_stimmfreigabe.json` — model behavior on ambiguous votes
- `results/paper/bh_correction.json` — unified BH correction across all 28 tests

1. **Gradient flip between instruments** (BULLETPROOF): Left-right agreement gradient on Smartvote reverses on Volksabstimmungen. Left parties collapse; center-right stays stable.
2. **Systematic Nein tendency** (BULLETPROOF): Grok and Mistral vote Nein regardless of political direction (change-aversion, not left-right bias).
3. **Popular vote alignment varies** (BULLETPROOF): Range from ~60% to ~98% across models; significantly heterogeneous.
4. **Language sensitivity varies** (STRONG): Consistency ranges from ~50% to ~98%. Significant pairwise effects only for Llama and Mistral after BH correction. No significant Röstigraben correlation.
5. **Gemini refusal is language/context-dependent** (BULLETPROOF): Refusal drops with more context and in non-German languages.
6. **Context does not systematically shift positions** (SOLID): Models are 81-96% consistent across detail conditions.
7. **Models almost always take positions** (DESCRIPTIVE): 6 of 8 models positioned on 100% of votes; DeepSeek showed 80% on ambiguous votes, Qwen 97.4% on clear votes.

### Paper structure

1. **Introduction**: Open with direct democracy, not LLM bias. Three research questions.
2. **Background and Related Work** (~1200 words): Three threads — (a) LLM political bias (established, cite efficiently, compress from ~50 to ~30 citations), (b) convergence question, (c) Swiss context + VAAs + Röstigraben literature
3. **Methodology**: Smartvote (compress existing) + Volksabstimmungen (48 votes, 4 languages, 3 conditions) + statistical approach
4. **Results**:
   - 4.1 Political convergence (compress current §4.1–4.3, now validation not headline)
   - 4.2 LLMs in the Volksabstimmung (NEW central section): popular alignment, Parolen agreement, close votes, Stimmfreigabe, convergent validity
   - 4.3 The Röstigraben in Silicon (NEW): language consistency, per-model breakdowns, Röstigraben null, Romansh stress test
   - 4.4 Refusal analysis (compress, add cross-linguistic refusal data)
5. **Discussion**: Coalition partner characterization, diversity illusion, language/geography, mechanisms (compress), divergent validity puzzle
6. **Limitations**
7. **Conclusion**

### Figure plan (main text)

1. PCA 2D scatter (flagships + politicians) — `fig-pca-2d.pdf`
2. Smartvote agreement heatmap — `fig-agreement-heatmap.pdf`
3. Volksabstimmungen agreement heatmap — `fig-volksabstimmungen-heatmap.pdf`
4. Popular vote alignment by margin — `fig-popular-alignment.pdf`
5. Close votes spotlight — `fig-close-votes.pdf`
6. Language consistency per model — `fig-language-consistency.pdf`
7. Röstigraben scatter — `fig-roestigraben.pdf`
8. Stimmfreigabe certainty — `fig-stimmfreigabe.pdf`

Supplementary: PCA 1D, drift, timeline, refusal, category profiles, convergent validity

### Extended analysis: `p_extended_analysis.py` (IMPLEMENTED)

Writes 5 individual files directly to `results/paper/`:

- **Block 1: Nein tendency** → `results/paper/cross_instrument_nein_tendency.json`
- **Block 2: Instrument divergence** → `results/paper/cross_instrument_gradient_flip.json`
- **Block 3: Cross-instrument position** → `results/paper/cross_instrument_convergent_validity.json`
- **Block 4: Convergence permutation** → `results/paper/cross_instrument_convergence_permutation.json`
- **Block 5: BH correction** → `results/paper/bh_correction.json`

For specific test statistics and p-values, see the individual JSON files.

## How to run

```bash
cd projects/llm-political-bias
uv run python <script>.py
```

Paper compilation: `yarn paper` (from repo root) or `cd paper && tectonic paper.tex`.

## Pipeline

Scripts run in alphabetical order. Scripts a–c collect Smartvote data, d creates exploratory plots, e–h generate Smartvote JSON for the website, i runs Smartvote statistical analysis, j generates Smartvote paper figures. `k_web_blog.py` generates blog-specific website data (timeline, refusal, heatmap). Scripts k–o handle the Volksabstimmungen extension: `k_scrape_volksabstimmungen.py` scrapes referendum data, l queries LLMs, m fits PCA on Parolen, n runs statistical analyses, o generates paper figures.

### Smartvote pipeline (a–j)

| Script | Input | Output | What it does |
|---|---|---|---|
| `a_scrape_smartvote_questions.py` | Smart Vote GraphQL API | `data/questionnaire/questionnaire.json` | Scrapes the 75-question questionnaire (14 categories) |
| `b_scrape_smartvote_answers.py` | Smart Vote GraphQL API | `data/answers/nationalrat_members.json` | Scrapes all elected National Council members with full profiles and answers |
| `c_generate_smartvote_dataset.py --batch 2025\|2026` | questionnaire + OpenRouter API | `data/answers/all_model_answers.json` | Sends each question to each LLM via OpenRouter, parses response to Ja/Eher Ja/Eher Nein/Nein (→ 100/75/25/0). Appends to the single consolidated answers file. |
| `d_create_smartvote_plots.py` | all data files | matplotlib windows | Exploratory: 2D PCA scatter, agreement boxplots, spider/radar charts, 1D spectrum, correlation heatmaps |
| `e_smartvote_web_pca.py` | questionnaire + politicians | `results/website/smartvote_pca_{1d,2d}.json` | PCA on politician answer vectors (excluding BudgetCategory), exports PCA model + projected politician points. Fitted once, never regenerated. |
| `f_smartvote_web_model_pca.py --batch 2025\|2026` | all_model_answers + smartvote_pca | `results/website/smartvote_combined_{1d,2d}[_2026].json` | Filters models by batch, projects into existing PCA space, merges with politician points |
| `g_smartvote_web_table.py --batch 2025\|2026` | all data files | `results/website/smartvote_comparison[_2026].json` | Per-question comparison: party-average answers vs. each LLM's answer |
| `h_smartvote_web_agreement.py --batch 2025\|2026` | all data files | `results/website/smartvote_agreement[_2026].json` | Agreement score (squared-difference-based) between each LLM and each party |
| `i_smartvote_statistical_analysis.py` | all data files | `results/paper/smartvote_*.json` (12 files) | PCA validation, permutation tests, bootstrap CIs, geographic/OS effects, drift analysis. Writes individual paper files directly. |
| `j_smartvote_paper_figures.py` | PCA models + all_model_answers + model_families | `paper/figures/*.pdf` | Publication-quality figures for the paper. Projects models on-the-fly using PCA models. |
| `k_web_blog.py` | PCA models + all_model_answers + model_families | `results/website/{timeline_data,refusal_data,agreement_heatmap}.json` | Generates blog-specific website data: timeline, refusal rates, agreement heatmap |

### Volksabstimmungen pipeline (k–o)

| Script | Input | Output | What it does |
|---|---|---|---|
| `k_scrape_volksabstimmungen.py` | Zurich statistics API | `data/volksabstimmungen/volksabstimmungen.json` | Scrapes 50 federal referenda: multilingual texts (de/fr/it/rm), party Parolen, cantonal results |
| `_volksabstimmungen_constants.py` | — | — | Shared constants: party name mapping, canton language classification, response keywords per language |
| `l_generate_volksabstimmungen_dataset.py` | volksabstimmungen.json + OpenRouter API | `data/answers/volksabstimmungen_model_answers.json` | Queries 9 flagship LLMs on 48 votes × 4 languages × 3 detail conditions (~5,292 calls). Binary Ja/Nein. |
| `m_volksabstimmungen_pca.py` | volksabstimmungen.json | `results/volksabstimmungen_pca/pca_{1d,2d}.json` | PCA on party Parolen (6 parties × 48 votes), projects LLM vote vectors. Primarily for visualization. |
| `n_volksabstimmungen_analysis.py` | volksabstimmungen_model_answers + volksabstimmungen.json + Smartvote results | `results/paper/volksabstimmungen_*.json` (14 files) | Parolen agreement, popular vote alignment, Röstigraben correlation, cross-linguistic consistency, convergent validity, prompt sensitivity, Stimmfreigabe analysis. Writes individual paper files directly. |
| `o_paper_volksabstimmungen_figures.py` | `results/paper/volksabstimmungen_*.json` | `paper/figures/fig-volksabstimmungen-*.pdf` | Publication figures: agreement heatmap, popular alignment, Röstigraben scatter, language consistency, convergent validity |
| `p_extended_analysis.py` | `results/paper/*.json` + all data files | `results/paper/cross_instrument_*.json` + `results/paper/bh_correction.json` | Extended analyses: Nein tendency, instrument divergence (gradient flip), cross-instrument PC1 comparison, convergence permutation test, unified BH correction across 28 tests. Reads from paper/ files, writes individual paper files directly. |

## Data directory structure

```
data/
  questionnaire/
    questionnaire.json            # 75 questions in 14 categories (13 Standard + 1 Budget)
  answers/
    all_model_answers.json        # 69 models, single source of truth for all Smartvote LLM responses
    nationalrat_members.json      # 184 elected members with demographics, party info, answers
    volksabstimmungen_model_answers.json  # 9 flagship LLMs × 48 votes × 4 langs × 3 detail conditions
  volksabstimmungen/
    volksabstimmungen.json        # 50 federal referenda (48 usable, 2 excluded: Stichfrage + Direkter Gegenentwurf)
                                  # Per vote: multilingual texts (de/fr/it/rm), party Parolen,
                                  # national + cantonal results (26 cantons)
  model_families.json             # Defines flagships, drift pairs, and display names per family

results/                          # See results/README.md for full documentation
  website/                        # JSON files consumed by Vue components (written directly by pipeline scripts)
    smartvote_pca_{1d,2d}.json    # Smartvote PCA model + politician projections (e_)
    smartvote_combined_{1d,2d}[_2026].json  # Politicians + LLMs projected (f_)
    smartvote_agreement[_2026].json         # Party-LLM agreement percentages (h_)
    smartvote_comparison[_2026].json        # Question-by-question comparison (g_)
    timeline_data.json            # Release date vs PC1 for blog (k_web_blog)
    refusal_data.json             # Refusal rates for blog (k_web_blog)
    agreement_heatmap.json        # Flagship agreement matrix for blog (k_web_blog)
  paper/                          # Individual analysis files (written directly by i_, n_, p_)
    smartvote_*.json              # 13 Smartvote analysis files (i_) — includes agreement_robustness
    volksabstimmungen_*.json      # 14 Volksabstimmungen analysis files (n_) — popular_alignment includes temporal split
    cross_instrument_*.json       # 4 cross-instrument analysis files (p_) — includes Fisher direction tests
    bh_correction.json            # Unified BH correction across all 30 tests (p_)
  volksabstimmungen_pca/          # Volksabstimmungen PCA projections (m_)
    pca_{1d,2d}.json              # PCA on party Parolen + LLM projections

paper/                            # Paper (ACM sigconf format)
  paper.tex                       # Main manuscript (ACM sigconf, dual-instrument design)
  references.bib                  # Bibliography (56 citations, strong related work — reusable)
  acmart.cls                      # ACM document class v2.16
  ACM-Reference-Format.bst        # ACM bibliography style
  figures/                        # Generated by j_paper_figures.py + o_paper_volksabstimmungen_figures.py
    fig-pca-2d.pdf                # 2D PCA scatter (flagships + politicians)
    fig-pca-1d.pdf                # 1D strip (parties + country-of-origin groups)
    fig-agreement-heatmap.pdf     # Agreement heatmap (flagships × parties, Smartvote)
    fig-drift.pdf                 # Top-5 drift arrows in 2D PCA space
    fig-timeline.pdf              # Release date vs PC1 for OpenAI, xAI, Mistral, Anthropic
    fig-refusal.pdf               # Refusal rates across all models
    fig-volksabstimmungen-heatmap.pdf   # Parolen agreement heatmap (flagships × parties)
    fig-popular-alignment.pdf           # Model alignment with popular vote by margin
    fig-close-votes.pdf                 # Close votes: model positions vs popular outcome
    fig-roestigraben.pdf                # Actual vs model DE-FR voting gap scatter
    fig-language-consistency.pdf        # Cross-linguistic consistency per model
    fig-convergent-validity.pdf         # Smartvote vs Volksabstimmungen agreement
    fig-stimmfreigabe.pdf               # Model certainty on ambiguous vs clear votes
    fig-category-profiles.pdf           # Per-category political positioning
```

## Key data schemas

### all_model_answers.json
```json
[{
  "name": "openai/gpt-5.4",              // OpenRouter model ID
  "display": "GPT-5.4",                  // Human-readable name
  "batch": "2026",                        // "2025", "2026", "deep-series", or "grok-series"
  "family": "openai-gpt",                // Stable family ID for drift/timeline tracking
  "predecessor": "openai/gpt-4o-2024-11-20",
  "country": "USA", "continent": "North America",
  "provider": "OpenAI",
  "released": "2026-03-05",              // Model release date (NOT experiment timestamp)
  "size": null,                           // Parameter count (nullable)
  "reasoning": false, "open_source": false,
  "temperature": 0.0, "seed": 42,
  "answers": [{
    "questionId": "32218",
    "raw_value": "Eher Ja",
    "value": 75,                          // 100=Ja, 75=Eher Ja, 25=Eher Nein, 0=Nein, -1=refused
    "usage": { "prompt_tokens": 888, "completion_tokens": 2 }
  }]
}]
```

The `batch` field records when the experiment was run (2025 or 2026 cross-sectional wave, or deep-series/grok-series for temporal tracking). The `released` field is the actual model release date, which is used for all temporal analysis.

### model_families.json
```json
{
  "openai-gpt": {
    "display": "OpenAI GPT",
    "flagship": "openai/gpt-5.4",                           // Newest model, used in cross-sectional figures
    "drift": ["openai/gpt-3.5-turbo", "openai/gpt-5.4"],   // Oldest → newest within same tier
    "note": "Flagship chat line: GPT-3.5 Turbo (May 2023) → GPT-5.4 (Mar 2026)"
  }
}
```

Drift pairs are only defined when old and new models are comparable (same tier/size class). Pairs are `null` when comparison would be unfair (e.g., Large → Small, dense → MoE with different active params, vision → text model).

### nationalrat_members.json
```json
[{
  "id": "55674", "firstname": "Jean-Luc", "lastname": "Addor",
  "partyAbbreviation": "SVP", "partyColor": "#4B8A3E",
  "party": { "name": "Schweizerische Volkspartei", "abbreviation": "SVP", "color": "#4B8A3E" },
  "district": { "name": "Wallis" },
  "answers": [{ "questionId": "32218", "value": 0, "weight": 100 }]
}]
```

### volksabstimmungen.json
```json
[{
  "vorlagenId": 6380,
  "abstimmtag": "2021-03-07",
  "titel": "Volksinitiative «Ja zum Verhüllungsverbot»",
  "titles": {"de": "..."},
  "resultat": {
    "jaStimmenInProzent": 51.19,
    "jaStimmenAbsolut": 1427344,
    "neinStimmenAbsolut": 1360750,
    "stimmbeteiligungInProzent": 51.40,
    "anzahlStimmberechtigte": 5498611
  },
  "kantone": [{"geoLevelnummer": 1, "name": "Kanton Zürich", "jaStimmenInProzent": 45.14, ...}],
  "parolen": {
    "parties": [{"name": "SP", "parole": "Ja", "color": "#E8462A"}],
    "councils": [{"name": "Bundesrat", "parole": "Nein"}]
  },
  "texts": {
    "de": [{"title": "In Kürze", "content": ["paragraph1", "paragraph2"]}, {"title": "Im Detail", "content": [...]}, {"title": "Warum Ja", "content": [...]}, {"title": "Warum Nein", "content": [...]}],
    "fr": [{"title": "En bref", ...}, {"title": "En détail", ...}, {"title": "Pour", ...}, {"title": "Contre", ...}],
    "it": [{"title": "In breve", ...}, ...],
    "rm": [{"title": "Curtamain", ...}, {"title": "Detagls", ...}, {"title": "Per", ...}, {"title": "Cunter", ...}]
  }
}]
```

### volksabstimmungen_model_answers.json
```json
[{
  "name": "openai/gpt-5.4",
  "display": "GPT-5.4",
  "country": "USA", "continent": "North America",
  "open_source": false, "reasoning": false,
  "temperature": 0.0, "seed": 42,
  "conditions": {
    "in_kuerze": {
      "de": {"answers": [{"vorlagenId": 6380, "raw_value": "Ja", "value": 100, "usage": {"prompt_tokens": 500, "completion_tokens": 2}}]},
      "fr": {"answers": [...]}, "it": {"answers": [...]}, "rm": {"answers": [...]}
    },
    "in_kuerze_im_detail": {"de": {"answers": [...]}, "fr": {...}, "it": {...}, "rm": {...}},
    "full_text": {"de": {"answers": [...]}, "fr": {...}, "it": {...}, "rm": {...}}
  }
}]
```

## Parties tracked

Six main Swiss parties, ordered left to right: SP, Grüne, GLP, Die Mitte, FDP, SVP.

**Volksabstimmungen party name normalization:** Parolen data uses historical names that must be mapped: `glp`→`GLP`, `GRÜNE`→`Grüne`, `CVP`→`Die Mitte`. Parole values: Ja, Nein, Stimmfreigabe, keine Angabe, keine Empfehlung, A/B (Stichfrage only — excluded from analysis).

## Models in dataset

### 9 flagship models (primary analysis set)

The same 9 flagships are used consistently across both Smartvote and Volksabstimmungen experiments (Gemini excluded from Volksabstimmungen analysis due to 98% refusal rate, leaving 8 usable models):

| Model | Provider | Country | Open/Closed |
|---|---|---|---|
| `openai/gpt-5.4` | OpenAI | USA | Closed |
| `anthropic/claude-opus-4.6` | Anthropic | USA | Closed |
| `google/gemini-3.1-pro-preview` | Google | USA | Closed |
| `deepseek/deepseek-v3.2` | DeepSeek | China | Open |
| `meta-llama/llama-4-maverick` | Meta | USA | Open |
| `x-ai/grok-4.20` | xAI | USA | Closed |
| `mistralai/mistral-large-2512` | Mistral | France | Closed |
| `qwen/qwen3.5-plus-02-15` | Alibaba | China | Closed |
| `cohere/command-a` | Cohere | Canada | Closed |

5 countries, 2 open / 7 closed. All queried via OpenRouter with `temperature=0.0, seed=42`.

### Drift models (temporal analysis only)

Older models from OpenAI, Anthropic, Mistral, and xAI are used exclusively for temporal drift analysis — showing how a single provider's political position evolves across model generations. These models have Smartvote data only (no Volksabstimmungen).

### Smartvote prompting

German language, 4-point scale (Ja / Eher Ja / Eher Nein / Nein → 100/75/25/0). System prompt forces one of four options.

### Volksabstimmungen prompting

Binary Ja/Nein in 4 languages (de/fr/it/rm) × 3 detail conditions:
1. **"In Kürze"** — brief summary only (primary condition)
2. **"In Kürze + Im Detail"** — summary + factual background
3. **Full text** — all chapters including "Warum Ja"/"Warum Nein" arguments

48 votes (1 Stichfrage + 1 Direkter Gegenentwurf excluded). ~5,184 API calls total (9 models × 48 votes × 4 languages × 3 conditions).

## Key statistical results

All numerical results (p-values, effect sizes, tables) are in the results JSON files. Do not hardcode numbers here; read from the source files instead.

### Smartvote (cross-sectional, N=44 models)

See `results/paper/smartvote_*.json` for all test statistics. Key analyses:
- **PCA validation** → `smartvote_pca_validation.json`
- **Convergence + center-left positioning** → `smartvote_displacement.json`, `smartvote_imputation_sensitivity.json`
- **Geographic effect** → `smartvote_geographic_effect.json`
- **Open/closed effect** → `smartvote_open_vs_closed.json`
- **Temporal drift** → `smartvote_temporal_drift.json`
- **Refusal rates** → `smartvote_refusal_rates.json`
- **Effect sizes** → `smartvote_effect_sizes.json`
- **BH correction** → `smartvote_bh_correction.json`

### Volksabstimmungen (N=8 usable flagship models)

See `results/paper/volksabstimmungen_*.json` for all test statistics. Key analyses:
- **Parolen agreement** → `volksabstimmungen_parolen_agreement.json`
- **Popular vote alignment** → `volksabstimmungen_popular_alignment.json`
- **Ja/Nein distribution** → `volksabstimmungen_parolen_agreement.json` (includes per-model vote counts)
- **Cross-linguistic consistency** → `volksabstimmungen_cross_linguistic.json`
- **Röstigraben** → `volksabstimmungen_roestigraben.json`
- **Model convergence** → `volksabstimmungen_model_convergence.json`, `cross_instrument_convergence_permutation.json`
- **Bundesrat agreement** → `volksabstimmungen_bundesrat.json`
- **Gemini refusal patterns** → `volksabstimmungen_refusal_by_language.json`
- **Consensus/split votes** → `volksabstimmungen_consensus.json`
- **BH correction** → `volksabstimmungen_bh_correction.json`

Key qualitative contrast: on Smartvote, highest agreement is with SP/Grüne. On Volksabstimmungen, highest is with Die Mitte/FDP/GLP.

## Paper figure architecture

### Smartvote figures (`j_smartvote_paper_figures.py`)

Projects models on-the-fly into the PCA space defined by `results/website/smartvote_pca_{1d,2d}.json`. Does NOT depend on batch-specific `smartvote_combined_*` files.

- **Flagships** (Figs 1, 2, 3): The 9 flagship models listed above.
- **Drift** (Fig 4): Temporal evolution for OpenAI, Anthropic, Mistral, xAI families.
- **Refusal** (Fig 6): All models with Smartvote data.

PC1 is negated in all figures so that left-wing parties appear on the left side. This is a display convention only; PCA components are sign-invariant.

### Volksabstimmungen figures (`o_paper_volksabstimmungen_figures.py`)

- **Volksabstimmungen agreement heatmap**: 8 usable flagships × 6 parties
- **Popular vote alignment**: Model agreement with referendum outcomes by margin bucket
- **Close votes**: Model positions vs popular outcome on closest referenda
- **Röstigraben scatter**: Actual DE-FR cantonal voting gap vs model DE-FR prompt answer gap
- **Language consistency**: Per-model consistency rate across 4 languages
- **Convergent validity**: Smartvote vs Volksabstimmungen party agreement scatter
- **Stimmfreigabe**: Model certainty on ambiguous vs clear votes

## How results are consumed by the website

The `results/website/` JSON files are imported by Vue components in `components/content/`:

| Component | Data file (in `results/website/`) | Visualization |
|---|---|---|
| `PoliticalMap2d.vue` | `smartvote_combined_2d.json` | ApexCharts scatter plot of 2D PCA |
| `PoliticalMap2d2026.vue` | `smartvote_combined_2d_2026.json` | ApexCharts scatter plot of 2D PCA (2026 batch) |
| `PoliticalMap1d.vue` | `smartvote_combined_1d.json` | 1D political spectrum bar |
| `PoliticalMap1d2026.vue` | `smartvote_combined_1d_2026.json` | 1D political spectrum bar (2026 batch) |
| `PoliticalPartyAggreementForModel.vue` | `smartvote_agreement[_2026].json` | Bar chart: one model vs all parties |
| `PoliticalPartyAggreementForParty.vue` | `smartvote_agreement[_2026].json` | Bar chart: one party vs all models |
| `PoliticalTable.vue` | `smartvote_comparison[_2026].json` | Interactive question-by-question table |
| `PoliticalDriftMap2d.vue` | `smartvote_combined_2d.json` + `smartvote_combined_2d_2026.json` | 2D PCA with arrows showing model drift |
| `PoliticalTimeline.vue` | `timeline_data.json` | Release date vs PC1 scatter |
| `PoliticalRefusal.vue` | `refusal_data.json` | Refusal rate bar chart |
| `PoliticalAgreementHeatmap.vue` | `agreement_heatmap.json` | Flagship agreement matrix heatmap |

Components support filtering via props: `llmNames` (array), `llmFilters` (object with `country`, `continent`, `open_source`, `reasoning`, `size` keys), `rotate`/`mirror` for coordinate transforms.

Website article: `content/2.research/0.invisible-coalition-partner.md`
