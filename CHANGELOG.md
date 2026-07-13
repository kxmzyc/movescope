# Changelog

## 0.2.0 - 2026-07-13

- Reject non-finite features and degenerate bone geometry instead of returning misleading scores.
- Add a 5-degree template tolerance floor for single or low-variance expert sets.
- Preserve complete DTW coverage by falling back to full weighted alignment when segment counts differ.
- Validate feature weights, action names, upload extensions, empty uploads, file size, and pose coverage.
- Add local-development CORS and a deterministic synthetic `/demo` assessment.
- Add template discovery, synthetic verification, richer findings, and JSON export to the React workspace.
- Expand the Python regression suite to 40 tests.
- Correct MotionBERT, benchmark, citation, and data-provenance documentation.

## 0.1.0 - 2026-06-28

- Initial prototype with MediaPipe extraction, angle features, template scoring, DTW alignment, FastAPI, Gradio, and React entry points.
