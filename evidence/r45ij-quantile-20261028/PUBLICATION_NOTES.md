# R45-I/J evidence publication notes

- Source PR branch base: 8ed6f0bfa9eec4adbd32f8c875ec9121c1db6d63.
- Raw R45-I/J JSON reports, stdout/stderr, commands, exit codes, and original run-local SHA256SUMS are preserved byte-for-byte.
- The historical R44 environment.txt is published as .redacted: only the internal hostname was masked. See redaction_manifest.json; the original file hash remains recorded in evidence_index.json and historical_r44/SHA256SUMS.
- The historical normalizer is an 84-scalar task-level q01/q99 summary, with schema/fit metadata and no per-sample data. It is included byte-for-byte for reproducibility.
- No HDF5/JPEG payloads, model weights, checkpoints, credentials, virtual environments, caches, or private configuration files are included.
- Historical R44 training_config.json is archival context and reports alpha=32; it is not the current approved R44 training configuration.
- The historical source commit is a candidate from the clean archived worktree, not cryptographically bound by the retained preflight report; see evidence_index.json.
- The historical SHA256SUMS entry for environment.txt names the unredacted original. All other entries were verified against the byte-preserved published files; the redacted derivative is separately hashed in redaction_manifest.json and the root SHA256SUMS.
