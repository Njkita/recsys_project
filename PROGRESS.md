# PROGRESS — sequential recommenders на ML-20M

Состояние: **код полностью готов к запуску**. Тестируется на GPU-сервере (`x32-gpu-01`, A100-80GB), здесь GPU нет.

## Что сделано

### Инфраструктура
- 12 модулей в `src/` (модели, losses, eval, train, benchmark, results aggregator, StackRec, utils, data, sideinfo, augment)
- 8 YAML-конфигов в `configs/`
- 7 bash-скриптов в `scripts/` (setup_server, download_data, preprocess, install_mamba, train, train_all, benchmark)
- model registry в `src/models/__init__.py` — единая точка `build_model(cfg)` для всех 6 моделей

### Архитектуры
- [x] **SASRec — modern** (`models/sasrec.py`): Pre-LN, RoPE, SwiGLU, LiGR-gates(init=0), tied weights, SDPA с правильной combined causal+pad маской и защитой от all-masked rows (NaN).
- [x] **NextItNet** (`models/nextitnet.py`): causal dilated 1D conv с padding слева, GLU вместо ReLU, tied weights.
- [x] **FMLP-Rec** (`models/fmlp.py`): learnable complex filter в частотной области (rFFT → W × → irFFT) + SwiGLU FFN.
- [x] **Linear-attn SASRec** (`models/linear_attn.py`): elu+1 feature map, causal через cumsum, RoPE сохранён.
- [x] **FNet hybrid** (`models/fnet.py`): 3 parameter-free FFT-блока + 1 attention сверху.
- [x] **Mamba4Rec** (`models/mamba4rec.py`): drop-in через `mamba-ssm`, lazy import + helpful error если зависимость не установлена.

### Loss + sampling
- [x] gBCE с numerically-stable log-sigmoid + nan_to_num защитой
- [x] Sampled softmax с logQ correction (uniform + pop-weighted)
- [x] Faster `sample_negatives` — chunked multinomial для больших B·L·K (≥1M)
- [x] Optional in-batch negatives через append уникальных positives батча
- [x] popularity^0.75 sampling weights + соответствующий logQ_pop

### Training
- [x] AdamW + split_decay_params (no decay на emb / norm / bias)
- [x] WarmupCosineLR (5% linear warmup + cosine decay до 1% base lr)
- [x] bf16 autocast (на A100 — torch.bfloat16, иначе fp32)
- [x] EMA(0.999) с apply/restore хуками вокруг eval
- [x] Gradient clipping по норме
- [x] Snapshot ensemble (опционально)
- [x] Early stopping по val NDCG@10
- [x] Полная checkpoint: state_dict + ema state_dict + epoch + val_metrics + cfg
- [x] StackRec multi-stage runner: 4 → 8 → 16, lr scaling per stage, warm-start через `stack_blocks`

### Eval
- [x] Векторизованный filter-seen через padded [n_users, max_seen=3000] tensor + scatter_ — на порядок быстрее Python-loop'а из чужих реп
- [x] HR@K, NDCG@K, MRR@max_k для K ∈ {5, 10, 20}
- [x] PAD-column всегда замаскирована
- [x] Защита target-item от случайной фильтрации seen (clamp)

### Reporting
- [x] Каждый запуск пишет `result.json` + `log.jsonl`
- [x] `python -m src.results` собирает CSV + markdown leaderboard
- [x] `python -m src.benchmark` сравнивает forward latency / VRAM / throughput по моделям × L ∈ {200, 500, 1000, 2000}

### Документация
- [x] README.md — обзор, workflow, ссылки
- [x] PROGRESS.md (этот файл)
- [x] NOTES.md — научный лог концепций и источников
- [x] RESEARCH.md — глубокий ресёрч (без изменений с initial setup)
- [x] SERVER.md — инвентарь GPU-сервера + sync workflow

## Что не делаем (явно отказались)

- **ReaRec ERL/PRL** (V1adls1aV/sasrec_reasoning) — METRICS.md их же команды показывает: ERL → −9.3% NDCG, PRL → −19.1% на ML-20M. Авторы объясняют: помогает на разреженных датасетах с короткими историями, ML-20M плотный (avg 144) — base уже видит rich context. Не стоит времени.
- **HSTU** (Meta generative recommenders) — не drop-in: требует переписывать pipeline под `PreprocessorModule` / `OutputPostprocessorModule` + custom Triton kernels. Расцветает на industrial-scale. На ML-20M overkill.
- **Contrastive (CL4SRec, DuoRec)** — augmentations лежат в `src/augment.py` как building blocks, но contrastive head вне scope v1. Можно подключить в follow-up.

## Ожидаемые цифры на ML-20M (full-catalog LOO, filter-seen)

Цели по приоритету:

1. `sasrec_baseline` ≈ 0.18-0.19 NDCG@10 (соответствие с V1adls1aV anchor 0.187)
2. `sasrec` (modern) ≥ **0.20** — должен превзойти их baseline на ~7-15%
3. `nextitnet` в пределах ±2% от modern SASRec
4. `fmlp` потенциально лучший на L=200 (по бенчмаркам Beauty/Yelp)
5. `linear_attn`, `fnet_hybrid` ниже modern SASRec, но дешевле
6. `stackrec_sasrec` финал на 16 блоках ≥ modern на 3 блоках, wall-clock −30..45%

## Перед сдачей

- [ ] Запустить `bash scripts/train_all.sh` на A100 (~24-72 ч)
- [ ] Установить mamba (`bash scripts/install_mamba.sh`) и догнать Mamba отдельно
- [ ] `bash scripts/benchmark.sh` — таблица latency/throughput/VRAM
- [ ] Проверить что `runs/results.md` сгенерировался и числа разумные
- [ ] Сверить с командой через [таблицу](https://docs.google.com/spreadsheets/d/11faVJOewsAIuRLmHnzrRzV8T9cZ93PViSZZSGNkkd5Q)
- [ ] Защита: показать сводную таблицу + benchmark.csv + RESEARCH.md / NOTES.md как нарратив

## Коммитим в git

Conventional commits, как просит V1adls1aV. Структура веток:

```
main         — стабильное состояние, готовое к запуску
dev          — в процессе работы
feature/...  — новая модель / фича
experiment/... — гипотезы, можно не мерджить
```

Артефакты `runs/`, `data/`, `.venv/` — в .gitignore.
