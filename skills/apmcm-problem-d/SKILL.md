---
name: apmcm-problem-d
description: Work on the 2026 APMCM Problem D perovskite screening task, including surrogate prediction, Kaiwu SDK QUBO optimization, reproducible experiments, and the required English paper. Use for implementation, result review, writing, or final checks for this specific contest problem.
---

# 2026 APMCM Problem D workflow

This skill adapts the modeling-to-paper workflow of `cumcm-live-workflow` to the 2026 Asia and Pacific Mathematical Contest in Modeling, Problem D. It applies to this problem only. Communicate with the team in their preferred language. Write the submitted paper, abstract, captions, legends, tables, references, and appendix explanations in English.

## Authoritative materials

Resolve paths from the repository root containing this skill and `problem_d/date/`.

- Problem statement: `sources/problems/2026_APMCM_Problem_D.pdf`
- Data and data notes: `problem_d/date/`
- Essay rules: `sources/rules/essay_format_and_submission_specifications.pdf`
- Contest discipline and deadline: `sources/rules/contest_discipline_and_submission.pdf`
- Provided LaTeX template: `sources/latex/`

Read `docs/02_modeling_plan.md` for the current mathematical plan, `docs/03_innovations_and_evidence.md` for measured results and limitations, and `docs/04_team_workflow.md` for the next work. Reproduce experiments using `docs/06_environment.md`. Treat the documented methods as candidates until validation and resource checks support a final choice.

Re-read relevant source pages before treating a numerical threshold, submission detail, or wording requirement as authoritative. Follow the user's current instructions and the APMCM materials above. The CUMCM skill's national-contest templates, Chinese typography, AI disclosure rules, fixed figure/formula/reference counts, and result spreadsheets are not APMCM requirements. Use the reusable parts of that workflow only when they help this problem.

## Start and coordination

When asked to start work, inspect the existing project and running environment. Check the Python interpreter actually used, necessary numerical and plotting packages, `kaiwu` availability and version, LaTeX compiler, and PDF inspection tools. Install task dependencies as needed for the authorized work. Record the Kaiwu backend actually used. A local simulator and special-purpose quantum hardware must be identified accurately in results.

For a two-person team, give one owner the surrogate model and validation, and the other the QUBO and Kaiwu solution. Both own the final checks. Share a versioned candidate-score table with columns `id,A,B,X,V_hat`, the fitted-model version, training files, validation metrics, and code revision. Keep older tables so reported optimization results remain attributable to one model. The optimization owner can begin with synthetic score tables before the first real model is ready. Freeze one score version before final experiments and paper assembly.

## Data and objective checks

1. Inventory the supplied CSV files and check row counts, unique `id` and `(A,B,X)` tuples, file schemas, and permitted candidate pools. The whole composition grid has 52 A choices, 52 B choices, and 7 X choices. Equal A and B are legal.
2. Recompute the five squared hinge penalties from `e_form`, `gap`, `vbm`, and `cbm`, each using a divisor of 0.5, and compare their sum with `V` in labeled data. Keep six-decimal source rounding in mind. `gap=0` means a metallic material, not a missing observation.
3. State the boundary convention explicitly. The problem quotes 15 candidates satisfying all strict inequalities and 19 candidates with `V=0` under its penalty definition. Use the specified penalty sum for the surrogate objective. Avoid treating the two counts as interchangeable.
4. File `02` is Problem 1 training data, `03` is validation/model selection data, and `04` has composition-only holdout records. The split is grouped by the `(A,B)` pair. Verify that no pair leaks between train and validation. The data notes say all 19 zero-violation candidates are in the hidden holdout; confirm any public-data counts from the files before reporting them.
5. For Problems 2 and 3, refit a selected model on `02+03` as specified. File `07` is the complete 2,704-state NO2 pool; `08` is the complete 8,112-state pool for O3, NO2, FO2. Files `11` and `12` are blind subsets. State which pool each result uses. A QUBO for a blind subset must enforce membership in that subset; a full-grid one-hot formulation alone allows excluded combinations.

## Problem 1: surrogate model

Predict `V_hat(A,B,X)` using only information available for unlabeled candidates, including deterministic elemental descriptors if their source is documented. Do not use measured `e_form`, `gap`, `vbm`, `cbm`, lattice geometry, or any target-derived column as an inference input unless the same value is genuinely provided for every candidate being scored. Preserve the distinction between A and B sites.

Build a simple reproducible baseline, then try improvements justified by validation results. The training and validation sets contain no `V=0` examples, so evaluate continuous error and rare-target ranking rather than claiming measured positive-class recall. Useful measures include MAE/RMSE, the best true `V` among predicted top-k candidates, and recall of validation samples below explicitly stated near-target thresholds at budgets such as k=20, 50, and 100. Report uncertainty or sensitivity when top-ranked candidates change under reasonable model choices. Use the official validation split for model selection. Generate predictions for composition-only holdout and candidate pools without inventing hidden labels.

## Problems 2 and 3: QUBO and solver

Problem 2 fixes X at NO2. With binary A variables `a_i` and B variables `b_j`, represent the full-pool score exactly as `sum_ij V_hat(i,j,NO2) a_i b_j`, plus penalties enforcing `sum_i a_i=1` and `sum_j b_j=1`. Explain the score normalization and penalty choice. Check decoded feasibility, original unscaled surrogate score, and QUBO energy for every reported solution.

For Problem 3, avoid an unexplained cubic `a_i b_j x_k` term. An exact quadratic representation for the full pool uses 156 one-hot variables for the `(A,X)` choice and 52 one-hot variables for B, with score-table couplings between them. Alternatively, exploit the complete X grid: formulate and solve the three fixed-X QUBOs, then take the lowest of their optima. Explain why this decomposition covers every candidate and document how it was solved with Kaiwu. Choose the encoding that fits verified SDK and hardware coefficient limits; quantify coefficient scaling or precision changes that can alter the score ordering.

The current preferred comparison also uses exact partial minimization. Define `C[a,b] = min_x V_hat(a,b,x)`, solve the resulting 104-variable A/B QUBO, then recover all minimizing X choices for the selected pair. Check equality against all 8,112 original scores. Record the preprocessing cost and preserve the joint formulation as a baseline.

Enumerate all 2,704 and 8,112 full-pool surrogate scores as exact classical benchmarks. Compare solver output against the relevant benchmark using feasible-solution rate, original-score optimality gap, repetitions, wall time, backend and compute/resource use. Break cost down by surrogate training, score generation, QUBO construction, solver calls, and checking. Exact minimization of `V_hat` is evidence about the optimization stage, not proof that an unseen material satisfies the physical target.

## English paper and verification

Use the provided APMCM template when practical and replace its example content. Set the problem number to D and use the team's real registration number only in allowed locations. The first PDF page is the summary page with problem letter and team number; the second page is the contents page; the main text begins after it. The main text is in English, including all chart labels and captions, and is limited to 25 pages. Use readable 11- or 12-point text. The header carries only team number and page number. Keep personal names and school information out of the summary and main paper.

Cover all three problems and report assumptions, model selection, ranking validation, QUBO construction, actual Kaiwu solve, costs, and limitations. Cite external sources in the paper with numbered references such as `[1]`, including sources for elemental descriptors and methods. State software used and provide the computer source code in the appendix if code was used, as requested by the APMCM essay rules. Make every headline result traceable to a saved run and score-table version. Review plots and compiled PDF visually, then cross-check the summary, main text, tables, figures, and code output for matching numbers. Verify the paper does not claim hidden target labels or hardware access that were not observed.

Prepare the paper PDF as `D<team-number>.pdf`. Prepare the signed control sheet as `D<team-number>cns.pdf` or `.jpg`; the team supplies its own English signatures. Keep reproducible code and derived results available. Prepare supporting materials only when they are to be submitted under the current APMCM instructions, and exclude the original competition data. The contest deadline is 2026-11-30 09:00 China Standard Time; leave upload time before that deadline. Do not import the CUMCM skill's separate AI-use declaration or its national-contest attachment requirements into the APMCM paper without an applicable APMCM rule.

## Upstream attribution

Adapted from the workflow ideas in haoxilin/cumcm-live-workflow-skill at commit `f024d8b92b432936c4b792359cf44bf4885c07e2`. Its MIT notice is included in `UPSTREAM-LICENSE`. The APMCM-specific paths, rules, and experiments were prepared for this project.
