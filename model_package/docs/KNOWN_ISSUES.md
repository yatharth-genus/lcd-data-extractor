# Known Limitations

- Detector precision: 55.56%
- Detector recall: 39.47%
- Detector Hmean: 46.15%
- False positives may include icons, inactive LCD segments, and borders.
- Some predictions may merge adjacent text regions or split one region into fragments.
- Recognition metrics were calculated on manually prepared crops, not detector-generated crops.
- Ground truths were later found to contain a small number of annotation mistakes, so the v1 metrics are retained as historical baseline results.
- The branch is intended for reproducibility and comparison, not production deployment.