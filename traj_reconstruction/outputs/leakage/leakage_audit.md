# Leakage audit report

**ok:** `True`

- **info** `predict_kwargs`: predict_orbit kwargs are audio-only
- **info** `forbidden_list`: critical GT paths are inference-forbidden
- **info** `assert_ok`: allowed paths pass assert_inference_safe
- **info** `assert_block`: forbidden path correctly blocked
- **info** `split_overlap`: no train/val/test clip id overlap
- **info** `bundle_gt`: InferenceBundle has no GT attributes
- **info** `normalization`: Clips are peak-normalized in sim export; no site gain tables are used at inference.
- **info** `feature_tokens`: forbidden feature token list recorded for reviews
