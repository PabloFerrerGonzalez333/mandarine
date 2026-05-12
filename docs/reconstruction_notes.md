# Reconstruction Notes

## Historical baseline

- The strongest historical reference for the delivered TFG is `anomalib 0.3.7`.
- The historical environment aligns with `Python 3.8`.
- The original project mixed notebooks, local configs and a partial local copy of `anomalib`.

## Modernization notes

- The partial local package has been archived in `legacy/anomalib_snapshot/`.
- The modern pipeline now depends on the installed PyPI package instead of any repo-local `anomalib` import.
- The modern experiment uses `data/mandarins_pynq_cropped` as canonical input and treats `data/mandarins_pynq_augmented` as legacy-only material.

## Why the split changed

- The modern pipeline generates deterministic train/val/test folders for each seed.
- This avoids mixing synthetic augmentations derived from the whole dataset into validation or test.
- It also makes benchmarking and final training reproducible from scripts instead of notebooks.
