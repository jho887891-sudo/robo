# R45-L isolated NumPy 1.26.4 reproduction

- Source code SHA: 1cbf46b74c5e08561c68a33e79a89d891b010fd3.
- Python 3.10.12, NumPy 1.26.4, h5py 3.16.0 were installed in an isolated venv under /tmp; no system package was replaced.
- The repository's original float32 np.quantile path scanned 300 TRAIN/DEV episodes, 108780 TRAIN rows and 28268 DEV rows. Frozen and recomputed payload SHA256 match exactly.
- This is a delta-normalizer numerical reproduction only. The report explicitly says no full HDF5 content SHA was checked and no formal manifest was created. It does not pass the training CPU Gate or authorize GPU/training.
- No source, normalizer, split, training configuration, HDF5 data, image payload, model weights, checkpoints, CUDA or GPU were modified or used.
