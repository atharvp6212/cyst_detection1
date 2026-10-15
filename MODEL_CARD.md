# Model card

## Intended use

Educational research on image-classification reliability. This is not a medical
device and must not be used for diagnosis or treatment decisions.

## Required evidence before reporting performance

- Documented dataset provenance and licence
- Patient-independent split
- Frozen preprocessing and model version
- Held-out metrics with uncertainty
- Subgroup and failure-case analysis
- External validation before any claim of generalization

## Known risks

Dataset leakage, class imbalance, acquisition-device differences, poor
calibration and visually plausible but clinically meaningless explanations.
