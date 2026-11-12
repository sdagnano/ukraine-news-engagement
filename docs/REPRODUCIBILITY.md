# Reproducibility

## Integrity

`python scripts/verify_repository.py` from the repository root checks every supplied file against MANIFEST_SHA256.json using Python's standard library. This is an integrity check, not a model rerun.

## Numerical checks

From reproduction/, install requirements.txt, obtain the Facebook input using `python scripts/fetch_inputs.py --only facebook`, then run:

```console
python scripts/verify_submission.py
python analysis/r7_reproduce_composition.py --check
```

The first command checks saved numerical identities, input fingerprints and scientific verification records. It does not refit models. The second reconstructs central raw summaries, OLS accounting and standardized Poisson predictions from the processed Facebook data and cached model parameters. Its reference count is 277 comparisons at absolute tolerance 1e-10. It writes audit/r7_portable_composition_check.json, which can change the snapshot hash without changing estimates.

## Full execution

In a working copy, obtain all analysis inputs and run:

```console
python scripts/fetch_inputs.py
python scripts/run_all.py --verify-inputs-only
python scripts/run_submission_analysis.py --list-stages
python scripts/run_submission_analysis.py --full
```

The full sequence repeats the declared model fits, simulations, support analyses and hypothetical label corrections, and regenerates derived assets. It is computationally expensive. It does not compile documents or rewrite the repository manifest. Numerical comparisons across platforms use tolerances rather than bitwise equality. Python 3.12 is the reference interpreter; dependencies are pinned in requirements.txt. The main runner sets numerical-library thread counts to one for its subprocesses.

Historical helpers can use optional sibling work/pydeps or work/readr_deps directories if present. These are package fallbacks, not required data dependencies. A standalone checkout uses the installed requirements.

## Interpretation

The predictor is a released automatic label. Linked raw texts, matched independent human annotations and impressions are not available for these analyses. Hypothetical corrections are outcome-informed sensitivity calculations. Numerical verification does not establish semantic label validity, causal identification or calibrated interval coverage. The delivered checks do not constitute an independent external replication or a fresh full execution of every model and simulation.
