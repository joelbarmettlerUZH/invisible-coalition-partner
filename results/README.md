# Results Directory

This directory contains all analysis outputs for the LLM Political Bias project. Files are organized into three categories: paper results, website visualization data, and Volksabstimmungen PCA projections.

## Directory Structure

```
results/
├── website/                          # JSON files consumed by Vue components on the website
├── paper/                            # Individual analysis files for the paper (split from monoliths)
├── volksabstimmungen_pca/            # Volksabstimmungen PCA projections (website visualization)
│
├── statistical_analysis.json         # [INTERMEDIATE] Monolithic Smartvote analysis (written by i_*)
├── volksabstimmungen_analysis.json   # [INTERMEDIATE] Monolithic Volksabstimmungen analysis (written by n_*)
├── extended_analysis.json            # [INTERMEDIATE] Monolithic cross-instrument analysis (written by p_*)
│
├── pca_results_1d.json               # [SHARED] Smartvote PCA model (read by paper + website scripts)
├── pca_results_2d.json               # [SHARED] Smartvote PCA model (read by paper + website scripts)
├── combined_results_{1d,2d}[_2026].json   # [SHARED] PCA projections (read by website + k_web_blog)
├── agreement_scores[_2026].json      # [SHARED] Agreement scores (read by website)
├── comparison_table[_2026].json      # [SHARED] Question-by-question comparison (read by website)
├── volksabstimmungen_pca_{1d,2d}.json # [SHARED] Volksabstimmungen PCA (read by website)
├── timeline_data.json                # [SHARED] Timeline data for blog (written by k_web_blog)
├── refusal_data.json                 # [SHARED] Refusal rates for blog (written by k_web_blog)
└── agreement_heatmap.json            # [SHARED] Agreement heatmap for blog (written by k_web_blog)
```

Files marked [INTERMEDIATE] are monolithic outputs from analysis scripts that are split into individual files in `paper/` by `_restructure_results.py`. They are kept for backward compatibility with figure scripts (`j_smartvote_paper_figures.py`, `o_paper_volksabstimmungen_figures.py`) that read from them directly.

Files marked [SHARED] are written by pipeline scripts (e/f/g/h/k/m) and copied to `website/` by `_restructure_results.py` with clearer names.

## How to Regenerate

After running analysis scripts, run the restructuring script to update the split files:

```bash
uv run python _restructure_results.py
```

The full regeneration order is:

1. `e_smartvote_web_pca.py` -- PCA model (fitted once, rarely rerun)
2. `f_smartvote_web_model_pca.py --batch 2025|2026` -- project models into PCA
3. `g_smartvote_web_table.py --batch 2025|2026` -- comparison table
4. `h_smartvote_web_agreement.py --batch 2025|2026` -- agreement scores
5. `i_smartvote_statistical_analysis.py` -- Smartvote statistical analysis
6. `j_smartvote_paper_figures.py` -- Smartvote paper figures
7. `k_web_blog.py` -- blog-specific website data
8. `m_volksabstimmungen_pca.py` -- Volksabstimmungen PCA
9. `n_volksabstimmungen_analysis.py` -- Volksabstimmungen analysis
10. `o_paper_volksabstimmungen_figures.py` -- Volksabstimmungen paper figures
11. `p_extended_analysis.py` -- cross-instrument analysis
12. `_restructure_results.py` -- **split monoliths into paper/, copy to website/**

## website/

Files consumed by Vue components in `components/content/Political*.vue`. These are copies of the root-level files, renamed for clarity.

| File | Source | Vue Component |
|---|---|---|
| `smartvote_pca_1d.json` | `pca_results_1d.json` | (used by PCA model internals) |
| `smartvote_pca_2d.json` | `pca_results_2d.json` | (used by PCA model internals) |
| `smartvote_combined_1d.json` | `combined_results_1d.json` | `PoliticalMap1d.vue` |
| `smartvote_combined_1d_2026.json` | `combined_results_1d_2026.json` | `PoliticalMap1d2026.vue` |
| `smartvote_combined_2d.json` | `combined_results_2d.json` | `PoliticalMap2d.vue`, `PoliticalDriftMap2d.vue` |
| `smartvote_combined_2d_2026.json` | `combined_results_2d_2026.json` | `PoliticalMap2d2026.vue`, `PoliticalDriftMap2d.vue` |
| `smartvote_agreement.json` | `agreement_scores.json` | `PoliticalPartyAggreementForModel.vue`, `PoliticalPartyAggreementForParty.vue` |
| `smartvote_agreement_2026.json` | `agreement_scores_2026.json` | `PoliticalPartyAggreementForModel2026.vue`, `PoliticalPartyAggreementForParty2026.vue` |
| `smartvote_comparison.json` | `comparison_table.json` | `PoliticalTable.vue` |
| `smartvote_comparison_2026.json` | `comparison_table_2026.json` | `PoliticalTable2026.vue` |
| `timeline_data.json` | `timeline_data.json` | `PoliticalTimeline.vue` |
| `refusal_data.json` | `refusal_data.json` | `PoliticalRefusal.vue` |
| `agreement_heatmap.json` | `agreement_heatmap.json` | `PoliticalAgreementHeatmap.vue` |

## paper/

Individual analysis result files for the paper. Each file is self-documenting with metadata fields: `description`, `generated_by`, `generated_at`, `models_analyzed`, `models_excluded`, `n_models`, `methodology`, and `results`.

### Smartvote analyses (from `i_smartvote_statistical_analysis.py`)

| File | Description |
|---|---|
| `smartvote_pca_validation.json` | PCA explained variance, silhouette score, Spearman with party ordering |
| `smartvote_displacement.json` | Permutation test for center-left displacement of LLM cluster |
| `smartvote_geographic_effect.json` | Kruskal-Wallis test: country/continent vs. PC1 |
| `smartvote_open_vs_closed.json` | Permutation test: open-source vs. closed-source positioning |
| `smartvote_temporal_drift.json` | Sign test on 16 predecessor-successor pairs |
| `smartvote_refusal_rates.json` | Per-model refusal rates on 75 Smartvote questions |
| `smartvote_category_profiles.json` | Per-category mean answer values for flagship models |
| `smartvote_deep_timeseries.json` | PC1 tracking across model family versions |
| `smartvote_imputation_sensitivity.json` | Sensitivity of displacement test to refusal imputation |
| `smartvote_effect_sizes.json` | Cohen's d and eta-squared for all tests |
| `smartvote_agreement_scores.json` | Party-model agreement (squared-difference metric) |
| `smartvote_bh_correction.json` | BH correction for the Smartvote test family |

### Volksabstimmungen analyses (from `n_volksabstimmungen_analysis.py`)

| File | Description |
|---|---|
| `volksabstimmungen_parolen_agreement.json` | Model-party agreement on 48 referendum Parolen |
| `volksabstimmungen_popular_alignment.json` | Model alignment with popular vote by margin |
| `volksabstimmungen_cross_linguistic.json` | 4-language consistency + McNemar pairwise tests |
| `volksabstimmungen_roestigraben.json` | DE-FR voting gap correlation (actual vs. model) |
| `volksabstimmungen_prompt_sensitivity.json` | Consistency across detail conditions |
| `volksabstimmungen_stimmfreigabe.json` | Model behavior on ambiguous vs. clear votes |
| `volksabstimmungen_bundesrat.json` | Agreement with Federal Council recommendation |
| `volksabstimmungen_model_convergence.json` | Inter-model vs. inter-party similarity |
| `volksabstimmungen_refusal_by_language.json` | Refusal rates by language (Gemini focus) |
| `volksabstimmungen_consensus.json` | Per-vote model consensus and split votes |
| `volksabstimmungen_convergent_validity.json` | Smartvote vs. Volksabstimmungen agreement correlation |
| `volksabstimmungen_party_agreement_by_language.json` | Party agreement shift by query language |
| `volksabstimmungen_temporal.json` | Agreement patterns by referendum year |
| `volksabstimmungen_bh_correction.json` | BH correction for the Volksabstimmungen test family |

### Cross-instrument analyses (from `p_extended_analysis.py`)

| File | Description |
|---|---|
| `cross_instrument_gradient_flip.json` | Left-right agreement gradient reversal between instruments (Wilcoxon p=0.008) |
| `cross_instrument_nein_tendency.json` | Systematic Nein/status-quo bias in Grok and Mistral (binomial tests) |
| `cross_instrument_convergent_validity.json` | Smartvote PC1 vs. Volksabstimmungen Ja rate (Spearman) |
| `cross_instrument_convergence_permutation.json` | Models more similar to each other than parties (permutation test) |
| `bh_correction.json` | Unified BH correction across all 28 tests from all three analysis scripts |

## volksabstimmungen_pca/

PCA projections for the Volksabstimmungen data, used for website visualization.

| File | Description |
|---|---|
| `pca_1d.json` | 1D PCA projection of party Parolen + LLM vote vectors |
| `pca_2d.json` | 2D PCA projection of party Parolen + LLM vote vectors |

## Model Exclusion Criteria

- **Smartvote**: Models with >50% refusal rate are excluded from PCA analysis (5 models excluded, including Gemini 3.1 Pro at 84%). All 66 models included in refusal rate reporting.
- **Volksabstimmungen**: Gemini 3.1 Pro excluded (98% refusal rate in German), leaving 8 flagship models for all analyses.
- **Cross-instrument**: Same 8 flagship models used for both instruments.
