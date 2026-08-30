# BCNB VQA Slim Results

This directory contains the Git-friendly export of the completed BCNB VQA run.

- Runtime: best no-gate hierarchical RAG logic from `46042d9`, with BCNB data adapters
- Questions: 7,274
- Correct: 3,976
- Accuracy: 54.6604%
- Abstained: 0
- Errors: 0

`predictions_slim.jsonl` retains question identifiers, task, choices, gold and predicted answers, correctness, routing, structured-candidate summary, override status, and parsing status. Large patch evidence, model traces, working memory, relation arrays, and raw model responses were omitted.

The complete source result remains outside Git at:

`/data_nas3/wl/BCNB_VQA_outputs/bcnb_live_metrics_v1/mc_answers.jsonl`

SHA-256:

- Complete source: `2e2dc87744e39f99e6e83dd35dd9cffd738b674d13200c87a1ab76ea74270bd5`
- Slim export: `15ad0ac09b49789d0ca88d927effb289ea361e4f5634e502a938e7361c8aa8af`
