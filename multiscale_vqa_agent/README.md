# Multi-scale G2P WSI VQA Agent

This package implements the six-stage TCGA-BRCA pipeline:

1. Numbered phenotype prototype Router (Qwen selects stable P001-style IDs)
2. Patient-level selected-phenotype prediction with G2P-1024/2048/4096
3. Prior-versus-learned relation reasoning for selected phenotypes
4. Coarse-to-fine multi-scale patch retrieval and deduplication
5. Multi-image pathology description with Patho-R1
6. Qwen semantic evidence fusion with canonical A/B/C output IDs

The Router emits one of three evidence routes. phenotype_direct invokes only selected numbered prototypes. morphology_only pools a small top-patch set from every phenotype attention map, deduplicates consensus regions, and sends the resulting multi-scale pyramids to Patho-R1. nonvisual skips visual inference for report, treatment, age, exact size/time, and similar questions that WSI cannot establish.

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

The JSONL output is resumable. Existing `(case_id, question)` pairs are skipped. Each row stores the execution plan, fused and per-scale prediction, prior/initial/learned relation evidence, multi-scale patch groups, Patho-R1 description, final answer, and reference answer.

Evaluate exact answer matching:

```bash
/home/wl/anaconda3/envs/mil/bin/python \
  /home/wl/agent_2026/g2p_toolbank_brca/multiscale_vqa_agent/evaluate_answers.py \
  /home/wl/agent_2026/g2p_toolbank_brca/outputs/multiscale_vqa_agent/answers.jsonl
```

The full 735-question run is a pipeline demonstration because most cases overlap the G2P training or validation split. It is not an unbiased test estimate.
