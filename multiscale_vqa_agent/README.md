# Multi-scale G2P WSI VQA Agent

This package implements the six-stage TCGA-BRCA pipeline:

1. Prototype-aware planner (Qwen or deterministic fallback)
2. Patient-level phenotype prediction with G2P-1024/2048/4096
3. Prior-versus-learned relation reasoning
4. Coarse-to-fine phenotype/program/gene patch retrieval
5. Multi-image pathology description with Patho-R1
6. Qwen answer fusion and verification

The three G2P models run once per patient. Questions targeting the same phenotype reuse relation and pathology evidence. Same-scale candidates are deduplicated; cross-scale matches are retained as evidence pyramids. The 2048/1024 global bypass prevents a weak 4096 candidate from suppressing focal evidence.

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
