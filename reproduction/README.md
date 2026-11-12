# Reproduction

Run commands in this directory. Use Python 3.12 and install requirements.txt.

```console
python scripts/fetch_inputs.py --only facebook
python scripts/verify_submission.py
python analysis/r7_reproduce_composition.py --check
```

For the full sequence, first obtain every analysis input:

```console
python scripts/fetch_inputs.py
python scripts/run_all.py --verify-inputs-only
python scripts/run_submission_analysis.py --list-stages
python scripts/run_submission_analysis.py --full
```

Full fits and simulations are computationally expensive. The saved outputs and numerical reconstruction checks are distinct from a new execution of the entire pipeline. Consult ../docs/REPRODUCIBILITY.md for the scope of each command.
