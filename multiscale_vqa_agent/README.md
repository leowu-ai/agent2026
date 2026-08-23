# Multi-scale G2P WSI VQA Agent

This package implements the seven-stage TCGA-BRCA pipeline:

1. Answerability Gate using only the question and choices
2. Numbered phenotype prototype Router (Qwen selects stable P001-style IDs)
3. Patient-level selected-phenotype prediction with G2P-1024/2048/4096
4. Prior-versus-learned relation reasoning for selected phenotypes
5. Coarse-to-fine multi-scale patch retrieval and deduplication
6. Multi-image pathology description with Patho-R1
7. Qwen semantic evidence fusion with canonical A/B/C output IDs

The Answerability Gate emits a strict JSON boolean `can_answer`. A false result
is saved as an abstention before the Router. If every pending question for a
patient is unanswerable, G2P is not run for that patient. Gold answerability
labels are optional and are read only after inference by the offline evaluator.

After the Answerability Gate accepts a question, the Router emits one of two
evidence routes. `phenotype_direct` uses the selected numbered prototype
prediction, relation evidence, selected phenotype patches when needed,
Patho-R1, and Fusion; its existing `direct` and `partial` matches are retained.
`morphology_only` combines all compact fused phenotype prototype predictions,
the existing representative all-phenotype patch pyramids, up to two WSI
overview thumbnails, a Patho-R1 visual summary, and final Fusion. No
question-conditioned or choice-conditioned patch retrieval is used in this
version.

The three G2P models run once per patient. Same-scale candidates are deduplicated; cross-scale matches are retained as evidence pyramids. The 2048/1024 global bypass prevents a weak 4096 candidate from suppressing focal evidence. Final option semantics are decided by Qwen from explicit labels and definitions; code validates only prototype IDs and answer IDs.

## Configurations

- `config.json`: mock Qwen/Patho-R1 adapters. It runs the numerical and retrieval pipeline without LLM servers.
- `config.servers.json`: OpenAI-compatible Qwen and Patho-R1 endpoints on ports 8000 and 8001.

Qwen and Patho-R1 should run in their own serving environment. G2P keeps using the validated `mil` environment. This also accommodates Qwen3.5's newer Transformers requirements.

## Quick checks

Planner only, without loading G2P:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/run_vqa.py \
  --planner_only --limit 20 --no_resume
```

One end-to-end case with mock LLM adapters:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/run_vqa.py \
  --limit 1 --no_resume
```

## Full run

With mock adapters:

```bash
bash /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/run_full_vqa.sh
```

With Qwen and Patho-R1 servers already running:

```bash
bash /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/run_full_vqa.sh \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/config.servers.json
```

Run only the inexpensive binary Answerability Gate for all 390 multiple-choice
questions:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/run_vqa.py \
  --config /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/config.servers.json \
  --vqa_json /home/wl/agent_2026/dataset/WsiVQA_test.json \
  --mc_only --answerability_only \
  --output /home/wl/agent_2026/g2p_toolbank_brca/outputs/multiscale_vqa_agent/answerability_binary_v1/gate_predictions.jsonl \
  --answerability_labels /home/wl/agent_2026/dataset/WsiVQA_answerability_binary_flat_v1.json \
  --no_resume
```

Run all 390 multiple-choice questions with binary answerability evaluation
after full VQA inference:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/run_mc_vqa.py \
  --config /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/config.servers.json \
  --vqa_json /home/wl/agent_2026/dataset/WsiVQA_test.json \
  --output /home/wl/agent_2026/g2p_toolbank_brca/outputs/multiscale_vqa_agent/answerability_binary_v1/mc_answers.jsonl \
  --answerability_labels /home/wl/agent_2026/dataset/WsiVQA_answerability_binary_flat_v1.json \
  --no_resume
```

Evaluate an existing answers file without running any model:

```bash
cd /home/wl/agent_2026/g2p_toolbank_brca
/home/wl/anaconda3/envs/mil/bin/python -m \
  multiscale_vqa_agent.answerability_evaluation \
  outputs/multiscale_vqa_agent/answerability_binary_v1/mc_answers.jsonl \
  --answerability_labels /home/wl/agent_2026/dataset/WsiVQA_answerability_binary_flat_v1.json
```

The JSONL output is resumable. Existing `(case_id, question)` pairs are skipped. Each row stores the execution plan, fused and per-scale prediction, prior/initial/learned relation evidence, multi-scale patch groups, Patho-R1 description, final answer, and reference answer.

Evaluate exact answer matching:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/evaluate_answers.py \
  /home/wl/agent_2026/g2p_toolbank_brca/outputs/multiscale_vqa_agent/answers.jsonl
```

Run the selective direct Qwen-VLM control using only whole-slide overview
thumbnails and the multiple-choice question (no G2P, retrieval, or Patho-R1).
Qwen first judges the information type from question and choices only, using
the same H&E answerability definition as the main method. It loads thumbnails
and answers only accepted questions. Existing `qwen_wsi_direct` and
`qwen_wsi_selective` outputs remain the forced-answer and thumbnail-gated v1
baselines:

```bash
cd /home/wl/agent_2026/g2p_toolbank_brca
mkdir -p outputs/multiscale_vqa_agent/qwen_wsi_selective_v2
nohup /home/wl/anaconda3/envs/mil/bin/python \
  multiscale_vqa_agent/run_qwen_wsi_baseline.py \
  --config multiscale_vqa_agent/config.servers.json \
  --vqa_json /home/wl/agent_2026/dataset/WsiVQA_test.json \
  --output outputs/multiscale_vqa_agent/qwen_wsi_selective_v2/mc_answers.jsonl \
  --answerability_labels /home/wl/agent_2026/dataset/WsiVQA_answerability_binary_flat_v1.json \
  --no_resume \
  > outputs/multiscale_vqa_agent/qwen_wsi_selective_v2/run.log 2>&1 &
```

The runner prioritizes diagnostic `DX` slides, caches patient thumbnails,
resumes by default, saves every answer immediately, and updates live accuracy.

The full 735-question run is a pipeline demonstration because most cases overlap the G2P training or validation split. It is not an unbiased test estimate.

## Hierarchical RAG Agent

The optional `hierarchical_rag` mode sends every selected MCQ through the
existing planner, patient-level G2P cache, and multiscale structured prediction.
There is no Answerability Gate in the inference path. Evidence search may
`finalize` with unresolved evidence, but Final Fusion still returns one option.
It first verifies compact structured G2P evidence at Round 0. When more
evidence is needed, it adaptively selects a spatial scale and keeps finer
visual rounds linked to the same WSI region:

```text
structured Round 0 -> answer, 4096/2048/1024, or finalize
4096 -> optional spatial child at 2048 or 1024
2048 -> optional spatial child at 1024
1024 -> optional Program@1024 -> optional Gene@1024
```

Program and gene observations are supportive WSI-derived evidence, not measured
RNA or clinical assays. The default `legacy` mode retains the previous pipeline.

Knowledge RAG v2 supplies evidence limitations and reasoning constraints, not
patient answers. Optional `--answerability_labels` are read only after inference
for evaluation:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  multiscale_vqa_agent/run_mc_vqa.py \
  --config multiscale_vqa_agent/config.servers.json \
  --vqa_json /home/wl/agent_2026/dataset/WsiVQA_test.json \
  --output outputs/multiscale_vqa_agent/hierarchical_rag_kb_v2_full390/mc_answers.jsonl \
  --metrics outputs/multiscale_vqa_agent/hierarchical_rag_kb_v2_full390/mc_answers_metrics.json \
  --agent_mode hierarchical_rag \
  --knowledge_base /home/wl/agent_2026/g2p_toolbank_brca/hybrid_pathology_knowledge_base_v2.zip \
  --no_resume
```
