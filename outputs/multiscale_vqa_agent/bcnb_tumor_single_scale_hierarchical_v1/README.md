# BCNB Tumor single-scale hierarchical VQA

Complete Tumor-only run on the adapted SlideBench-VQA-BCNB dataset using the
single-scale hierarchical structured-evidence runtime and the conservative
Round 0 direct-evidence guard.

## Result

- Questions: 1058 / 1058
- Correct: 764
- Accuracy: 72.2117%
- Errors: 0
- JSON parse failures: 0
- Answers outside choices: 0
- Round 0 strong-direct exits: 781
- Cases entering visual inspection: 277

Gold labels contain 957 invasive ductal carcinoma, 25 invasive lobular
carcinoma, and 76 Other cases. This class imbalance should be considered when
interpreting overall accuracy.

`mc_answers.jsonl.gz` is the losslessly compressed full per-question output.
The original uncompressed result remains outside Git at
`/data_nas3/wl/BCNB_VQA_outputs/bcnb_tumor_single_scale_hierarchical_v1/`.
