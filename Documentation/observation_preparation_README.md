# observation_preparation

Run from the tomography project root after replacing the entire existing folder with this folder:

```bash
python -m observation_preparation.test_ro_preparation
python -m observation_preparation.test_igs_preparation
python -m observation_preparation.test_roi_consistency
```

The package contains an actual `__init__.py` and both standalone test modules. RO and IGS test settings use the same Tromsø-centered 2000 km ROI. The shared ROI helper uses the common 5° Fibonacci spacing.
