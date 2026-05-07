# Sequential recommenders on ML-20M — production sweep

Учебный проект по курсу «Ускоряем трансформеры в глубоком обучении рекомендательных систем» (научник Данил, [TOPAPEC](https://github.com/TOPAPEC)). Один человек.

Один прогон `bash scripts/train_all.sh` тренирует семь архитектур на одинаковом протоколе (full-catalog leave-one-out по [arXiv:2309.07602 (gSASRec)](https://arxiv.org/abs/2309.07602)) и собирает сравнительную таблицу `runs/results.md`.

## Архитектуры в семействе

| Модель | Конфиг | Откуда |
|---|---|---|
| **SASRec — modern** (Pre-LN + RoPE + SwiGLU + LiGR + tied) | `configs/sasrec.yaml` | [Kang & McAuley 2018](https://arxiv.org/abs/1808.09781) + [eSASRec 2025](https://arxiv.org/abs/2508.06450) |
| **SASRec — vanilla baseline** (post-LN, ReLU, learned PE) | `configs/sasrec_baseline.yaml` | [V1adls1aV/esasrec](https://github.com/V1adls1aV/esasrec) baseline |
| **NextItNet** (causal dilated CNN + GLU + tied) | `configs/nextitnet.yaml` | [Yuan et al. WSDM'19](https://arxiv.org/abs/1808.05163) |
| **FMLP-Rec** (learnable filter в частотной области) | `configs/fmlp.yaml` | [Zhou et al. WWW'22](https://arxiv.org/abs/2202.13556) |
| **FNet hybrid** (3 FFT + 1 attention) | `configs/fnet_hybrid.yaml` | [Lee-Thorp 2021](https://arxiv.org/abs/2105.03824) |
| **Linear-attn SASRec** (elu+1 prefix-sum) | `configs/linear_attn.yaml` | [Katharopoulos 2020](https://arxiv.org/abs/2006.16236) |
| **Mamba4Rec** (selective SSM, опционально) | `configs/mamba.yaml` | [Liu et al. RelKD'24](https://arxiv.org/abs/2403.03900) |
| **StackRec SASRec** (4 → 8 → 16 layer-stacking) | `configs/stackrec_sasrec.yaml` | [Wang et al. SIGIR'21](https://arxiv.org/abs/2012.07598) |

Все семь делят:

- gBCE loss с `t=0.75`, `n_neg=256` ([arXiv:2309.07602](https://arxiv.org/abs/2309.07602)) — это **главный source of NDCG** в каталоге
- side info из ML-20M: 19 жанров (multi-hot) + decade (year из title) + Tag Genome (1128 тегов), сумма с item ID embedding
- AdamW с no-decay группой (emb/norm/bias) + warmup-cosine + bf16 autocast + EMA(0.999) + grad clip 1.0
- SSE-PT input augmentation (`p=0.1`) — [Wu et al. RecSys 2020](https://arxiv.org/abs/2008.13775)
- full-catalog leave-one-out evaluation с filter-seen, векторизованный scatter (~10× быстрее loop'а из чужих реп)
- tied input/output embeddings

## Workflow

```bash
# на GPU-сервере (x32-gpu-01, A100-80GB)
ssh SkeletonDadGaming@176.109.79.43 -p 2222
cd ~/recsys_project           # после git clone

# одноразово
bash scripts/setup_server.sh        # venv + системный torch + numpy/pandas/yaml
bash scripts/download_data.sh       # ml-20m.zip (~190 MB)
bash scripts/preprocess.sh          # 5-core + dense ids + LOO → data/processed.pkl
bash scripts/install_mamba.sh       # опционально, для Mamba4Rec

# полный sweep — ~24-72 ч на A100, results.md появится автоматически
bash scripts/train_all.sh

# или одна модель
bash scripts/train.sh configs/sasrec.yaml runs/sasrec_modern

# латентность / throughput / VRAM по моделям и L ∈ {200, 500, 1000, 2000}
bash scripts/benchmark.sh
```

После каждого тренинга в `runs/<exp>/` появляется `result.json` (final metrics + config), `log.jsonl` (per-step log), `best.pt`. `python -m src.results` собирает их в одну таблицу `runs/results.csv` + `runs/results.md`.

## Целевые числа на ML-20M

Базис команды по сводной таблице семинара: **NDCG@10 ≥ 0.14** (good), **≥ 0.15** (paper-replication eSASRec). Их собственный baseline в репе [V1adls1aV/esasrec/METRICS.md](https://github.com/V1adls1aV/esasrec/blob/main/METRICS.md): SASRec+ с 3000 negatives sampled CE → **HR@10=0.309, NDCG@10=0.187**. Наш стек должен взять эту планку и идти дальше — реальная цель **NDCG@10 ≥ 0.20** на flagship `configs/sasrec.yaml`.

## Структура

```
src/
├── data.py           # ML-20M 5-core + LOO + Datasets
├── sideinfo.py       # genres + decade + Tag Genome → SideInfoEmbedding
├── losses.py         # gBCE + sampled softmax + neg sampling (uniform / pop^0.75)
├── augment.py        # SSE-PT (active) + CL4SRec mask/crop/reorder (available)
├── utils.py          # set_seed, EMA, WarmupCosineLR, SnapshotEnsemble, JSONLogger
├── eval.py           # full-catalog HR/NDCG/MRR with vectorised filter-seen
├── stackrec.py       # layer-stacking utility (adjacent / sequential)
├── train.py          # main loop with optional StackRec multi-stage
├── benchmark.py      # forward-only latency / throughput / VRAM
├── results.py        # aggregate runs/<*>/result.json into table
└── models/
    ├── sasrec.py        Pre-LN + RoPE + SwiGLU + LiGR + tied
    ├── nextitnet.py     causal dilated CNN + GLU + tied
    ├── fmlp.py          FMLP-Rec — learnable filter (FFT × W × IFFT)
    ├── linear_attn.py   SASRec scaffolding + elu+1 prefix-sum attention
    ├── fnet.py          FNet mixer + hybrid (FFT bottom + attn top)
    └── mamba4rec.py     Mamba block via mamba-ssm wheel

configs/                 — YAML per model variant
scripts/                 — bash wrappers (setup, data, train, benchmark)
runs/                    — auto-generated, gitignored
data/                    — ML-20M raw + processed.pkl, gitignored
PROGRESS.md              — чеклист состояния и метрик
NOTES.md                 — научный лог: какая идея откуда взята
RESEARCH.md              — глубокий ресёрч (архитектуры, протокол, ссылки)
SERVER.md                — инвентарь GPU-сервера + workflow
PLAN.md                  — старый поэтапный план (с пометками что выполнено)
IDEAS.md                 — каталог идей для прироста NDCG (ROI-сортировка)
```

## Ссылки от Данила

- gSASRec / gBCE protocol (canonical для команды): https://arxiv.org/abs/2309.07602
- Reference SASRec репа: https://github.com/TOPAPEC/esasrec
- Параллельная команда (V1adls1aV): https://github.com/V1adls1aV/esasrec — наш baseline-anchor 0.187
- Сводная таблица команд: https://docs.google.com/spreadsheets/d/11faVJOewsAIuRLmHnzrRzV8T9cZ93PViSZZSGNkkd5Q
- NextItNet: https://arxiv.org/abs/1808.05163
- StackRec: https://arxiv.org/abs/2012.07598
- Linear Attention: https://arxiv.org/abs/2006.16236
- Mamba: https://arxiv.org/abs/2312.00752 + https://github.com/state-spaces/mamba
- SSM vs Transformer обзор: https://goombalab.github.io/blog/2025/tradeoffs/
- FNet: https://arxiv.org/abs/2105.03824
- FMLP-Rec: https://arxiv.org/abs/2202.13556 + https://github.com/RUCAIBox/FMLP-Rec
- DL база: https://dls.samcs.ru/
