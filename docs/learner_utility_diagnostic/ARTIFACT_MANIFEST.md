# Learner-Utility Diagnostic Artifact Manifest

Artifact root:

```text
/disk1/zmh_projects/evorl_artifacts/learner_utility_diagnostic/
```

All hashes are SHA256. Paths below are relative to the artifact root. The completed three-seed runs are the formal replication. Interrupted and failed attempts are preserved for provenance but are not included in formal statistics.

## Formal three-seed replication

| Path | Size | SHA256 | Description |
|---|---:|---|---|
| `replication_3seed/swimmer/seed_1/records/utility_records.csv` | 2,715,658 B | `349eb78e95cc6f3baa1f57a2bd220f5302d706d4670e90a1e80f835fef213233` | Seed 1 source-level records; 9,664 rows. |
| `replication_3seed/swimmer/seed_1/records/checkpoint_timing.csv` | 8,777 B | `686edfa31121be1f039e8a65637ec2ff03deafe18956544836065b3e463acb38` | Seed 1 instrumented checkpoint timing. |
| `replication_3seed/swimmer/seed_1/manifest/baseline_config.yaml` | 1,296 B | `f398ac83482fc891564fe1a83133d3e121339766abfbac2a0433321e65a71b20` | Seed 1 Original ERL baseline configuration. |
| `replication_3seed/swimmer/seed_1/manifest/diagnostic_config.yaml` | 1,531 B | `8a0a19e19548a63a5767e413d58aed7453481b0a06f0149ce1063c616579350a` | Seed 1 diagnostic configuration. |
| `replication_3seed/swimmer/seed_1/manifest/run_manifest.json` | 560 B | `17257c137a594db3bba822154e2f9fa58b9ca525ddfbf1977c22a6a347ad05cc` | Seed 1 commit, environment, and budget metadata. |
| `replication_3seed/swimmer/seed_1/logs/run.log` | 38,501 B | `a406bea5ac80a8a12c6ac12412b56da72dc77137816326d47e627af839070ae7` | Seed 1 execution log. |
| `replication_3seed/swimmer/seed_2/records/utility_records.csv` | 2,717,709 B | `8472b2cf6d34877af416572ae22b82afd9e791d16b20c85b9a118de9afd169e4` | Seed 2 source-level records; 9,664 rows. |
| `replication_3seed/swimmer/seed_2/records/checkpoint_timing.csv` | 8,774 B | `a55d1c8bfb93a890fb4168cb735e264b83570dcc17df446cdfd8ddd70cccb3c8` | Seed 2 instrumented checkpoint timing. |
| `replication_3seed/swimmer/seed_2/manifest/baseline_config.yaml` | 1,296 B | `575213b2a164980ba8c78cfd8ac6b17928f7fef776bc7db2258950f7ffdf7bd8` | Seed 2 Original ERL baseline configuration. |
| `replication_3seed/swimmer/seed_2/manifest/diagnostic_config.yaml` | 1,531 B | `bd9d85a78c7c6cd6d8b2d7c4b03e696c8d1354bd807824739f742aa4431b114e` | Seed 2 diagnostic configuration. |
| `replication_3seed/swimmer/seed_2/manifest/run_manifest.json` | 560 B | `980c559d027470c0ab70e9a1b9500f9f87487f270e3612028374788e69e65f5e` | Seed 2 commit, environment, and budget metadata. |
| `replication_3seed/swimmer/seed_2/logs/run.log` | 37,905 B | `954ec7622324e90d1ab42cc18cf1873e19e70b06c1a8d6d9442b510d6e745c5e` | Seed 2 execution log. |
| `replication_3seed/swimmer/seed_3/records/utility_records.csv` | 2,716,820 B | `1f0db094c55e15c239f00bc0233d13bd38a7a587d7711f726f8301fb99bf88b0` | Seed 3 source-level records; 9,664 rows. |
| `replication_3seed/swimmer/seed_3/records/checkpoint_timing.csv` | 8,760 B | `75e5b849fc84f7d32881c7bff523c5ecb7275d3d82bb65b8e0c2f5177834ff40` | Seed 3 instrumented checkpoint timing. |
| `replication_3seed/swimmer/seed_3/manifest/baseline_config.yaml` | 1,296 B | `59a060d915146cb03f64b4c37fed620efd270238b799002686032e1de86b7622` | Seed 3 Original ERL baseline configuration. |
| `replication_3seed/swimmer/seed_3/manifest/diagnostic_config.yaml` | 1,531 B | `3579f55b7a89e029a7459c5d3cc7ccfefee146b2d2fd2e941e4d2c9679431cf8` | Seed 3 diagnostic configuration. |
| `replication_3seed/swimmer/seed_3/manifest/run_manifest.json` | 560 B | `15fa5f0d0c2457b25fe553b435445abcfc735e0333df9705ddf4160002def983` | Seed 3 commit, environment, and budget metadata. |
| `replication_3seed/swimmer/seed_3/logs/run.log` | 37,905 B | `43952008f85f7c7f5f0891f9bee8a58413dee41554628957199a8d0d739b849c` | Seed 3 execution log. |

Each completed seed also contains `manifest/SHA256SUMS`, `progress.json`, and `allowed_config_diff.json`. `sha256sum -c manifest/SHA256SUMS` passes for all three seeds.

## Aggregate analysis

| Path | Size | SHA256 | Description |
|---|---:|---|---|
| `aggregate/aggregate_summary.json` | 2,104 B | `a696bb1d6c148658ae5abb94f74026a05ba9ecaa418e5d29f36642a35ea21db6` | Three-seed aggregate statistics. |
| `aggregate/all_seed_raw_records.csv` | 7,751,625 B | `e3675d6f8b93d116f289e0ec17963633b7562526697eecfaa7bd6386e422703e` | Combined 28,992-record source dataset; excluded from Git. |
| `aggregate/per_generation_metrics.csv` | 205,290 B | `1e0c2094f03cb2349bdcdad232809c2abde62cb8bcf3346c6a1b90b11c026b03` | 906 checkpoint-level metric rows. |
| `aggregate/per_seed_summary.csv` | 2,741 B | `dc1081fbc74395b939f626cdaf63089a32c775943b3d3d68dafb514e078d0bed` | Compact per-seed summary; copied into Git. |
| `aggregate/cross_seed_recurrence.csv` | 402 B | `4d101baa4625f5f13c25365d1bd4fd367fd76ce5caa210be5e5543082ab48eb1` | Cross-seed recurrence summary; copied into Git. |
| `aggregate/timing_summary.csv` | 276 B | `e7122466fc3f9bfe54d0bf1a307166f2494bcd77aacaa429c40b09332f54bae6` | Instrumented checkpoint wall-time summary. |
| `aggregate/REPORT.md` | 1,170 B | `cb54bfa41842465a2ebd665b1dd9995802559a1d9b93f2560105a181b9591765` | Automatically generated compact aggregate report. |

## Visualization

| Path | Size | SHA256 | Description |
|---|---:|---|---|
| `visualization/learner_utility_dashboard.png` | 817,355 B | `b4ae9a93d5db1d7e510d757ba98089ae50fe3a1e22faf42b63ffaf4b8c705a1f` | Final dashboard; copied into Git. |
| `visualization/fitness_utility_scatter_by_seed.png` | 421,034 B | `ef02102991f83a1900bec23de263e8f036a69668731e544c68c1c9ff8f4e6135` | Raw Fitness–Utility scatter by seed. |
| `visualization/learner_utility_3seed_analysis.xlsx` | 4,487,601 B | `c7140fd039b3f48a40e8e10dfed25d1ebd6263e46668592f30f4f5d9c17f35f4` | Local-review workbook with summary, native charts, metrics, and raw records; excluded from Git. |
| `visualization/verification.json` | 646 B | not separately frozen | Workbook structural verification metadata. |

## Exploratory smoke

| Path | Size | SHA256 | Description |
|---|---:|---|---|
| `smoke/20260927_swimmer_seed42_76309c58/records/utility_records.csv` | 3,166 B | `ec6dffb874b70a5d6898cb5324d1985ec083af5141597b6fb4d97e6f1df4ebd9` | Twelve-record exploratory smoke dataset. |
| `smoke/20260927_swimmer_seed42_76309c58/analysis/pooled_exploratory_analysis.json` | 2,135 B | `a9858d6c6c20725d870411008382f463cced5c93b6bde4d7d3c3e1c1282c51ed` | Exploratory pooled statistics; not formal evidence. |
| `smoke/20260927_swimmer_seed42_76309c58/figures/fitness_utility_scatter.svg` | 4,710 B | `3383494436274e493e0371dd2d3cbe1b9929ad325ca45aa23a8006f86337b939` | Smoke scatter plot. |
| `smoke/20260927_swimmer_seed42_76309c58/manifest/smoke_manifest.json` | 969 B | `bfd04862320908864d762483b75537d36e97d4f407ad607ffa486c2bec83970c` | Smoke configuration and provenance. |

## Preserved non-formal attempts

- `replication_3seed/swimmer/seed_1_interrupted_gen410/`: interrupted when the artifact directory was moved during execution; 1,280 records preserved with checksums.
- `replication_3seed/swimmer/seed_1_failed_ptxas_gen60/`: failed on `ptxas fatal: Internal error: writing file`; 160 records and failure evidence preserved.

Neither attempt contributes to the formal three-seed statistics.

## Git inclusion policy

Included in Git:

```text
docs/learner_utility_diagnostic/FINAL_REPORT.md
docs/learner_utility_diagnostic/ARTIFACT_MANIFEST.md
docs/learner_utility_diagnostic/per_seed_summary.csv
docs/learner_utility_diagnostic/cross_seed_recurrence.csv
docs/learner_utility_diagnostic/learner_utility_dashboard.png
```

Excluded from Git: raw 28,992-record data, per-generation bulk data, checkpoints, complete logs, failed/interrupted run data, and the large XLSX workbook.
