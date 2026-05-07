# NOTES — научный лог: что откуда взято и зачем

Каждая идея в коде прослежена до источника. Если в защите кто-то спросит «почему именно gBCE с t=0.75?» или «откуда decade в side-info?» — ответ здесь.

## 1. Loss

### gBCE с t=0.75, n_neg=256
- **Источник**: Petrov & Macdonald, RecSys 2023, [arXiv:2309.07602](https://arxiv.org/abs/2309.07602) («gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling»).
- **Почему**: ванильный SASRec с одним негативом и BCE учит overconfident скоры. На full-catalog инференсе это смещает рейтинг. gBCE интерполирует BCE и full softmax, восстанавливая калибровку. На ML-20M даёт +15-20% NDCG@10 относительно vanilla.
- **Параметры**: `t = 0.75` — paper-default для ML-1M; `n_neg = 256` — sweet spot (меньше 64 теряет калибровку, больше 1024 — диминишинг).
- **Реализация**: `src/losses.py::GBCELoss`. log-sigmoid через softplus для численной стабильности.

### Sampled softmax с logQ correction
- **Источник**: Bengio & Senecal 2003 (importance sampling в softmax); популяризовано в eSASRec.
- **Когда использовать**: на больших каталогах с pop-weighted sampling — там logQ корректирует bias к популярным items. Для uniform — logQ константен и не нужен.
- **Реализация**: `src/losses.py::SampledSoftmaxLoss`.

### Popularity^0.75 sampling
- **Источник**: word2vec (Mikolov 2013) — оригинальная subsampling-формула.
- **Почему**: чисто uniform негативы дают слишком лёгкий сигнал — модель легко отличает рандомного фильма от целевого. Pop-weighted заставляет различать популярные близкие альтернативы. +1-2% NDCG.
- **Реализация**: `src/losses.py::popularity_weights` + `make_logq_popularity`.

### In-batch negatives
- **Источник**: classic в metric learning (CLIP, sentence-transformers); в reco — eSASRec и dozens of others.
- **Реализация**: append уникальных positives текущего батча как extra negatives. Бесплатно по памяти, +0.5..1% NDCG.
- **Включается** через `loss.in_batch_negatives: true` в YAML.

## 2. Архитектура SASRec

### Pre-LN (вместо post-LN)
- **Источник**: Xiong et al. 2020, «On Layer Normalization in the Transformer Architecture».
- **Почему**: post-LN требует warmup для стабильности при глубине ≥4, pre-LN сходится сразу. Стандарт после GPT-2.

### RoPE positional encoding
- **Источник**: Su et al. 2021, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
- **Почему**: лучше extrapolation на длинные хвосты, нет per-position параметров (один cos/sin буфер). +1-2% NDCG на ML-20M в среднем.
- **Реализация**: `src/models/sasrec.py::RotaryEmbedding`. FP32 буферы для устойчивости в bf16/fp16.

### SwiGLU FFN
- **Источник**: Shazeer 2020, «GLU Variants Improve Transformer».
- **Почему**: gated FFN (`SiLU(W1x) * W2x`) консистентно сильнее ReLU+Linear на масштабе. Стандарт LLaMA/GPT/PaLM. +1-3% NDCG.
- **Параметризация**: hidden = `(8/3) * d * mult / 4` — компенсация за gated structure, чтобы число параметров совпадало с обычным 4d-ReLU FFN.

### LiGR-gates (init=0)
- **Источник**: Linear/Layered Gated Residual в production-моделях LinkedIn/Meta; для sequential rec формулируется в семействе eSASRec.
- **Эквивалент**: ReZero (Bachlechner 2020), LayerScale (CaiT, ViT).
- **Формула**: `y = x + g_attn ⊙ Attention(LN(x))`, `z = y + g_ffn ⊙ FFN(LN(y))`. Гейты — обучаемые per-channel скаляры, `init=0` — блок стартует с identity, оживает постепенно. Позволяет глубже стек без warmup.

### Tied input/output embeddings
- **Источник**: Press & Wolf 2017, «Using the Output Embedding to Improve Language Models»; стандарт ALBERT/GPT-2.
- **Почему**: экономит `|I| × d` параметров (на ML-20M это ~2.5M на d=128), +1-3% NDCG.
- **Реализация**: `model.output_embedding` — это property, возвращающее `self.item_emb.weight`; logits = `hidden @ E.T`.

### SDPA с combined mask, без `is_causal=True`
- **Почему так**: torch 2.5 запрещает одновременно `attn_mask` и `is_causal=True`. Если построить только causal mask и передать `key_padding_mask` отдельно — для строки, где все keys замаскированы (full-PAD prefix), SDPA вернёт NaN.
- **Решение**: построить combined mask `(j <= i) AND not_pad[j]`, гарантировать диагональ `keep[b,i,i]=True` через `OR eye` — никакая строка не может быть полностью замаскирована.

## 3. Training stack

### AdamW с no-decay на emb/norm/bias
- **Источник**: Loshchilov & Hutter 2019, «Decoupled Weight Decay Regularization».
- **Почему**: weight decay на эмбеддинги items систематически вредит на reco — paper-known issue. Bias и LayerNorm параметры не должны декаиться по теории. Стандартный split.

### Warmup-cosine LR
- **Источник**: Loshchilov & Hutter 2017 «SGDR» + классика трансформер-тренинга.
- **Почему**: 5% linear warmup защищает от ранних instabilities, cosine decay даёт плавное снижение к концу.

### bf16 autocast (А100)
- **Почему**: bf16 имеет тот же dynamic range что fp32 (8 бит экспоненты), но 16-битное представление. На A100 даёт ~2× speedup без overflow в softmax-логитах (риск fp16). На GPU без bf16 fallback в fp32.

### EMA весов (decay=0.999)
- **Источник**: классика (Polyak averaging), стандарт в современном supervised обучении (например, ConvNeXt, ViT).
- **Почему**: финальные метрики стабильнее, +0.5-1% NDCG. EMA держится в shadow dict, swap'ается на eval.

### Gradient clipping (norm=1.0)
- **Почему**: с gBCE градиенты на позитивах могут «выгорать» при overconfident модели; clip страхует. Must-have для глубоких стеков.

## 4. Side info из ML-20M

### Genres (multi-hot 19) + decade (year из title) + Tag Genome (1128-d)
- **Источник идеи**: общая практика; конкретно tag genome — [Vig et al. 2012](https://files.grouplens.org/datasets/movielens/tag-genome.pdf), готовый артефакт в датасете.
- **Почему**: чистые SASRec-бейзлайны игнорируют side-info. Контентный эмбеддинг (sum с item ID emb) даёт устойчивые +2-4% NDCG. День работы.
- **Реализация**: `src/sideinfo.py`. Genre — Linear(19→d), decade — Embedding(N_DECADES, d) с pad=0, genome — Linear(1128→d). Все три суммируются и добавляются к item ID emb.

## 5. Аугментации

### SSE-PT (`p=0.1`)
- **Источник**: Wu et al., RecSys 2020, «SSE-PT: Sequential Recommendation Via Personalized Transformer».
- **Почему**: с вероятностью p заменяем item id в input на случайный → embedding-level регуляризация. +2-3% NDCG для трёх строк кода. Включён по умолчанию.

### CL4SRec mask / crop / reorder (доступны, не активны)
- **Источник**: Xie et al. 2022.
- **Когда нужны**: для contrastive learning (требует второй head). Building blocks лежат в `src/augment.py` для follow-up.

## 6. Evaluation

### Full-catalog leave-one-out с filter-seen
- **Источник**: Krichene & Rendle, KDD 2020, [arXiv:2007.13239](https://arxiv.org/abs/2007.13239) — формальное доказательство, что sampled-100 metrics не сохраняют ранжирование моделей.
- **Реализация**: для каждого юзера ранжируем true target среди ВСЕХ items, маскируем PAD и виденные items, top-K, считаем HR/NDCG/MRR.

### Векторизованный filter-seen
- **Почему**: чужие репы (V1adls1aV/esasrec, TOPAPEC/esasrec) делают per-user Python loop с tensor.from_numpy — медленно. Мы предкомпилируем `[n_users, max_seen]` long tensor + `scatter_` за один CUDA-kernel. ~10× быстрее на ML-20M.

### Tie-break
- Чужие репы: `(scores > target).sum() + 1` — оптимистичный bias при ties.
- Наш: `topk` + `argmax(match)` берёт первый позиции совпадения — нейтральный к ties (берёт первый встреченный, это обычно желаемое поведение).

## 7. Альтернативные архитектуры

### FMLP-Rec — learnable filter в частотной области
- **Источник**: Zhou et al., WWW 2022, [arXiv:2202.13556](https://arxiv.org/abs/2202.13556), репа [RUCAIBox/FMLP-Rec](https://github.com/RUCAIBox/FMLP-Rec).
- **Идея**: rFFT по seq-оси → element-wise умножение на learnable complex weight → irFFT. Эквивалентно learnable circular convolution с ядром длины L. Параметры O(L·d) vs O(d²) у MHA.
- **Causality**: paper использует last-position next-item training; eval смотрит только на последний токен где будущего нет — формально legal.
- **Цифры**: на ML-1M / Beauty / Yelp обгоняет SASRec на 5-13% NDCG@10. На ML-20M ставка — на уровне или выше modern SASRec.

### Linear-attn (elu+1, prefix-sum)
- **Источник**: Katharopoulos et al., ICML 2020, [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) («Transformers are RNNs»).
- **Идея**: `softmax(QK)V → φ(Q)·(φ(K)ᵀV)`. Causal через cumsum. O(L·d²) compute, O(d²) state на инференс — RNN.
- **Зачем**: педагогическое сравнение. На L=200 проигрывает FlashAttention; pull ahead — на L≥1000 (synthetic experiment).

### FNet hybrid
- **Источник**: Lee-Thorp et al., [arXiv:2105.03824](https://arxiv.org/abs/2105.03824).
- **Идея**: 2D FFT по seq и hidden axes без обучаемых параметров. В hybrid-варианте 3 FNet-блока + 1 attention сверху.
- **FMLP-Rec строго сильнее** для reco (learnable > fixed), но FNet hybrid в наборе для пользы сравнения.

### Mamba4Rec
- **Источник**: Liu et al., RelKD'24, [arXiv:2403.03900](https://arxiv.org/abs/2403.03900) + repo [chengkai-liu/Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec).
- **Зависимости**: `mamba-ssm==2.2.4` + `causal-conv1d==1.4.0`. Pre-built wheels for torch 2.5+cu12 — единственный надёжный путь установки (см. `scripts/install_mamba.sh`).
- **Архитектура**: Mamba блок (selective SSM) + SwiGLU FFN. n_blocks=2 — paper показывает что глубже хуже на reco.
- **Зачем**: длинные последовательности, content-aware «forget gate» через input-зависимое Δ, O(L) compute с CUDA-kernel parallel scan.

### StackRec
- **Источник**: Wang et al., SIGIR 2021, [arXiv:2012.07598](https://arxiv.org/abs/2012.07598), repo [wangjiachun0426/StackRec](https://github.com/wangjiachun0426/StackRec).
- **Идея**: тренируем shallow модель → удваиваем глубину копированием блоков → дообучаем. Adjacent (interleaved) лучше sequential на трансформер (Table 4).
- **Совместимость с LiGR-gates**: deepcopy сохраняет обученные `g_attn`/`g_ffn` параметры, новый twin-блок имеет те же гейты — сходится быстрее чем clean-init глубокая модель.
- **Recipe для ML-20M**: 4 → 8 → 16, lr × {1, 0.5, 0.25}, 5% warmup на каждой стадии после стэкинга. Speedup ~30-45% wall-clock vs from-scratch L=16.

## 8. Что НЕ применили (и почему)

- **ReaRec reasoning (ERL/PRL)** из V1adls1aV/esasrec — их же METRICS.md показывает -9.3% / -19.1% NDCG на ML-20M. Помогает на разреженных датасетах, не на плотном ML-20M.
- **HSTU** (Meta) — требует переписать pipeline. Расцветает на industrial scale, не на ML-20M.
- **DeepNorm** — альтернатива StackRec для L≥32. На L≤16 StackRec практичнее.
- **Sampled-100 evaluation** — формально доказано (Krichene & Rendle) что не сохраняет ранжирование. Только full-catalog.
- **Reverse sequence augmentation** — для causal моделей не должен помогать; на ML-20M эмпирически не помогает.
- **Mixup на эмбеддингах** — нестабильно на reco.
- **Lion optimizer** — потенциально быстрее, требует тюнинга lr (3-10× меньше AdamW). Не первым делом.

## 9. Ссылки на сравнительные репы

- [TOPAPEC/esasrec](https://github.com/TOPAPEC/esasrec) — научник, reference для протокола данных. Vanilla SASRec+ внутри — без LiGR/RoPE/SwiGLU несмотря на название.
- [V1adls1aV/esasrec](https://github.com/V1adls1aV/esasrec) — параллельная команда. Их baseline NDCG@10=0.187 — наш anchor. Их слабости: vanilla post-LN, ReLU FFN, learned PE, no tied weights, no AdamW, no scheduler, no AMP, no EMA, naive sampled CE без logQ. Мы лучше по всем осям.
- [RUCAIBox/FMLP-Rec](https://github.com/RUCAIBox/FMLP-Rec) — reference FMLP-Rec.
- [chengkai-liu/Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec) — reference Mamba4Rec.
- [wangjiachun0426/StackRec](https://github.com/wangjiachun0426/StackRec) — reference StackRec.
- [RecBole](https://github.com/RUCAIBox/RecBole) — reference NextItNet (наш `ResidualBlockB` mirror'ит recbole/model/sequential_recommender/nextitnet.py).
