# G2P ToolBank BRCA

最小可行版 Gene-to-Program-to-Phenotype Hypergraph Prototype MIL。输入 TCGA-BRCA WSI patch features，输出表型预测、gene program score、prototype attention 以及 H/R 关系矩阵。

## 环境

服务器上推荐使用：

```bash
/home/wl/anaconda3/envs/mil/bin/python
```

base 环境中的 `torch` 不完整，可能出现 `No module named torch._C`。

## 1. 生成 manifest

```bash
cd ~/agent_2026/g2p_toolbank_brca
/home/wl/anaconda3/envs/mil/bin/python scripts/prepare_manifest.py \
  --label_dir ~/agent_2026/dataset \
  --feature_dir /data_nas2/ljs/Share/TCGA_Embed/TCGA-BRCA/clam_gen_1024/conch_v1_5_new \
  --out_dir outputs
```

输出：

- `outputs/aligned_manifest.csv`: 每行一个 slide 特征文件，包含 `case_id, slide_id, feature_path, has_phenotype, has_gene, split`。
- `outputs/case_splits.csv`: 固定 seed=42 的 case 级 split。
- `outputs/missing_feature_cases.txt`: 有标签但无特征的 case。

## 2. Smoke test

```bash
/home/wl/anaconda3/envs/mil/bin/python scripts/train_g2p_toolbank.py \
  --label_dir ~/agent_2026/dataset \
  --feature_dir /data_nas2/ljs/Share/TCGA_Embed/TCGA-BRCA/clam_gen_1024/conch_v1_5_new \
  --out_dir outputs/debug_run \
  --epochs 1 \
  --max_samples 20
```

## 3. 正式训练

```bash
/home/wl/anaconda3/envs/mil/bin/python scripts/train_g2p_toolbank.py \
  --label_dir ~/agent_2026/dataset \
  --feature_dir /data_nas2/ljs/Share/TCGA_Embed/TCGA-BRCA/clam_gen_1024/conch_v1_5_new \
  --out_dir outputs/full_run \
  --epochs 50
```

训练完成后会导出：

```text
outputs/full_run/G2P_ToolBank_Minimal/
├── model.pt
├── vocab.json
├── relations.npz
├── normalization.json
├── tool_registry.json
├── tool_metrics.csv
└── train_config.json
```

## 4. 评估和 prototype evidence

```bash
/home/wl/anaconda3/envs/mil/bin/python scripts/evaluate_g2p_toolbank.py \
  --label_dir ~/agent_2026/dataset \
  --manifest_csv outputs/full_run/aligned_manifest.csv \
  --tool_dir outputs/full_run/G2P_ToolBank_Minimal \
  --out_dir outputs/full_run
```

输出：

- `evidence_topk.csv`: 每个 program/phenotype prototype 的 top-k patch index 和 attention score。
- `H_gene_to_program.csv`: 训练后的 gene 到 program 学习矩阵。
- `R_program_to_phenotype.csv`: program 到 phenotype 的学习矩阵。
- `learned_gene_programs.csv/json`: 每个 gene program 训练后权重最高的基因及其先验成员标记。

## Agent 调用

后续 agent 可先读取 `tool_registry.json` 查找某个表型工具，获得相关 phenotype 字段、program、gene 和输出类型；再加载 `model.pt`、`vocab.json`、`normalization.json`、`relations.npz` 对新的 slide 特征运行推理，并从 attention 中提取 top-k patch evidence。
