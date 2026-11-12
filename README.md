# Ukraine news engagement

Computational materials for **Interpreting engagement with hostility-labelled news: Occurrence, volume and composition in Ukraine**.

**Author:** Simone D'Agnano, Universita del Piemonte Orientale.  
**ORCID:** [0009-0003-6394-9408](https://orcid.org/0009-0003-6394-9408).

The exploratory secondary analysis compares action occurrence, unconditional action counts and relative action composition in 535,936 Ukrainian Facebook news posts from 84 sources over 200 dates. Identical source-day linear projections distinguish occurrence and count outcomes; separate Poisson standardizations quantify fitted means and their ratios. Additional outputs report support, concentration, hypothetical label corrections and inferential diagnostics.

The estimates describe recorded automatic labels and actions in the released data. They do not identify causal responses, reach or the semantic validity of the labels. Hypothetical corrections are sensitivity analyses rather than observed classification errors.

## Contents

| Path | Contents |
| --- | --- |
| `reproduction/analysis/` | Analysis and numerical reconstruction programs |
| `reproduction/results/` | Saved estimates, diagnostic outputs and model parameters |
| `reproduction/tables/`, `reproduction/figures/` | Derived tables and figures |
| `reproduction/audit/` | Scientific specifications and numerical verification records |
| `reproduction/input/` | Versioned input-download manifest and derived reference fixtures |
| `docs/REPRODUCIBILITY.md` | Execution and verification instructions |
| `docs/DATA_PROVENANCE.md` | Data sources and reference populations |

## Verify the supplied files

From the repository root, using Python:

```console
python scripts/verify_repository.py
```

This checks file integrity without installing analysis dependencies or running models.

## Reconstruct the central results

Use Python 3.12 and the reference dependencies:

```console
cd reproduction
python -m pip install -r requirements.txt
python scripts/fetch_inputs.py --only facebook
python scripts/verify_submission.py
python analysis/r7_reproduce_composition.py --check
```

The final command reconstructs 277 central quantities from the processed Facebook input and cached parameters, using an absolute tolerance of 1e-10. It writes a numerical check record. Run it in a working copy if you want to retain the original snapshot unchanged.

The data downloader checks exact versioned SHA256 hashes. Original datasets are acquired from their source projects and are not redistributed here. The full model and simulation sequence is described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Attribution

The Ukrainian processed release is associated with Kyrychenko, Brik, van der Linden and Roozenbeek (2024), [Social identity correlates of social media engagement before and after the 2022 Russian invasion of Ukraine](https://doi.org/10.1038/s41467-024-52179-8), [OSF rmc3e](https://osf.io/rmc3e/). The retained external-setting reference input is from Pipal and colleagues, [OSF 68cje](https://osf.io/68cje/).

See [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) and [REUSE_AND_ATTRIBUTION.md](REUSE_AND_ATTRIBUTION.md) for input provenance and reuse scope. Citation metadata for this repository are in CITATION.cff.
