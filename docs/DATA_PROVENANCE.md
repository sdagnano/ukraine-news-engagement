# Data provenance

The Ukrainian processed records and original analysis code are associated with Kyrychenko et al. (2024), https://doi.org/10.1038/s41467-024-52179-8, and https://osf.io/rmc3e/.

The main Facebook reference population has 535,936 posts from 84 sources over 200 dates and 14,689 source-day cells, from 25 February to 12 September 2022. The supplied processed data contain a broader population before analysis filtering. The separate Twitter sample is not pooled with Facebook. The 139-record difference from the published private analysis sample is not treated as resolved.

The legacy external-setting input is from Pipal et al., If You Have Choices, Why Not Choose (and Share) All of Them? A Multiverse Approach to Understanding News Engagement on Social Media, https://osf.io/68cje/. Its original final_dataset.rds maps to input/pipal_final_dataset.rds.

reproduction/input/upstream_manifest.json records versioned source URLs, byte counts, roles, original names and SHA256 hashes. The downloader retrieves the files directly from those sources and checks the hashes. Optional original R scripts are downloaded only with the provenance selection. No original upstream data or third-party R scripts are included in this repository.

The two prior_OLS CSV files are derived reference fixtures. Scientific input fingerprints are in reproduction/audit/v4_input_hashes.json. After obtaining all analysis inputs, `python scripts/run_all.py --verify-inputs-only` checks their fingerprints without fitting models. Original-source terms continue to govern the upstream materials.
