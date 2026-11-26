# Problem D manuscript and reproducible experiments

[D_review.pdf](D_review.pdf) is the English manuscript. The team number is intentionally blank. The summary, contents, main text, references and full program listings are in the PDF. The two full candidate pools were solved with the local Kaiwu CPU exact optimizer. There was no SPQC device experiment.

| File | Contents |
|---|---|
| [main.tex](main.tex) | Editable manuscript entry point |
| [artifacts/q1_holdout.csv](artifacts/q1_holdout.csv) | 3,703 composition-only Q1 predictions, trained on 02 |
| [artifacts/q23_full.csv](artifacts/q23_full.csv) | 8,112 frozen predictions from the 02+03 refit, including every Q2 row |
| [artifacts/score_runs.csv](artifacts/score_runs.csv) | SHA-256 digests and feature, fitting and prediction times |
| [artifacts/selections.csv](artifacts/selections.csv) | Exact Kaiwu selections for pools 07 and 08, pool runners-up and enumeration gaps |
| [artifacts/blind_minima.csv](artifacts/blind_minima.csv) | Direct scans of blind subsets 11 and 12 |
| [artifacts/block_runs.csv](artifacts/block_runs.csv) | All 338 nine-spin calls, block indices (alphabetical element order), ground-state tie counts, original scores and timings |
| [artifacts/group_folds.csv](artifacts/group_folds.csv) | Five-fold grouped development results |
| [artifacts/costs.csv](artifacts/costs.csv) | Timings, calls, integer ranges and tie counts |

The nine manuscript figures include four perovskite-specific illustrations, available as vector PDFs and high-resolution PNGs. [Cubic cell and coordination shells](figures/crystal_sites.pdf) and [oxide site-exchange sections](figures/site_exchange.pdf) use ideal coordinates. [Band-edge levels](figures/band_edges.pdf) compare public oxide records and the seed-26 property model. [Anion substitutions](figures/anion_substitution.pdf) use composition fractions and the frozen La–Ti scores. The full-grid family heatmaps remain a supplementary figure. The paired resampling and seed overlap results are in [shortlist_sensitivity.csv](artifacts/shortlist_sensitivity.csv).

## Reproduce the numbers

From the repository root, use a Python 3.12 environment with the packages listed in `problem_d/requirements-lock.txt`. Keep all original D CSV files and the organizer's SHA-256 manifest and data notes in `problem_d/date/`. If starting from the support archive, supply this folder from the competition download. The official manifest also includes the original Chinese-named data notes.

The commands below use PowerShell and assume that `python` invokes the selected environment. Timing columns depend on machine load. For the plots and manuscript alone, use the stored outputs in the repository or archive.

```powershell
python problem_d/exploration/audit_data.py
foreach ($seed in 26,42) {
    python problem_d/exploration/compare_surrogates.py --seed $seed --predictions "problem_d/results/validation_seed$seed.csv"
    python problem_d/exploration/test_innovation_ideas.py stress --seed $seed --predictions "problem_d/results/stress_seed$seed.csv"
    python problem_d/exploration/compare_physical_properties.py --seed $seed --predictions "problem_d/results/physical_seed$seed.csv"
}
python problem_d/exploration/test_innovation_ideas.py encoding --seed 26 --predictions problem_d/results/encoding_seed26.csv
python problem_d/exploration/audit_sdk_precision.py
python problem_d/paper/screen.py scores
python problem_d/paper/screen.py development
python problem_d/paper/solve.py
python problem_d/paper/figures.py
python problem_d/paper/material_figures.py
python problem_d/paper/perovskite_plates.py
python problem_d/paper/tables.py
```

The Q1 predictions use seed 26 and the 02-only model. The Q2/Q3 refit uses the same seed and features. The SDK's public exact CPU solver works on these small blocks without interactive activation in the tested installation. The separate simulated-annealing backend prompted for official authorization. The paper's cost comparison uses the CPU backend.

## Compile the manuscript

With TeX Live and the APMCM template dependencies installed, run from this directory in PowerShell. The official template class and configuration are included under `sources/latex/`. A local TeX Live installation may need the `cprotect`, `bigfoot` (suffix.sty) and `appendix` packages when compiling with XeLaTeX.

```powershell
$env:TEXINPUTS = '.;../../sources/latex;' + $env:TEXINPUTS + ';'
xelatex -halt-on-error -interaction=nonstopmode -jobname=D_review main.tex
xelatex -halt-on-error -interaction=nonstopmode -jobname=D_review main.tex
xelatex -halt-on-error -interaction=nonstopmode -jobname=D_review main.tex
```

To insert the real registration number, set `\baominghao{}` in the entry file, rebuild and use the organizer's team-specific file name. The signed control sheet is a separate team document.

