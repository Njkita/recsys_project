# Глубокий ресёрч: семейство sequential recommenders на ML-20M

Документ собран из четырёх параллельных ресёрч-сессий + прямых WebFetch'ей на arxiv/github. Где источник был недоступен — отмечено явно как «по памяти, сверить с кодом/статьёй». Цель: знать и понимать всё, что нужно для воспроизведения и улучшения SASRec / NextItNet / eSASRec на MovieLens-20M, не выходя за пределы этого файла.

## Навигация

1. [SASRec → gSASRec → eSASRec](#1-sasrec--gsasrec--esasrec)
2. [NextItNet](#2-nextitnet)
3. [StackRec и масштабирование вглубь](#3-stackrec-и-масштабирование-вглубь)
4. [Efficient attention: Linear, Mamba, FNet](#4-efficient-attention-linear-mamba-fnet)
5. [Протокол MovieLens-20M](#5-протокол-movielens-20m)
6. [Что проверить руками в репе TOPAPEC/esasrec](#6-что-проверить-руками-в-репе-topapecesasrec)

---

## 1. SASRec → gSASRec → eSASRec

### 1.1 Оригинальный SASRec (Kang & McAuley, ICDM 2018, arXiv:1808.09781)

Идея простая: causal Transformer-decoder на задачу next-item prediction в духе language modelling. До 2018 в sequential rec доминировали GRU4Rec (RNN) и Caser (CNN); SASRec выиграл за счёт того же преимущества, что и Transformer в NLP — параллельная обработка последовательности и явное моделирование длинных зависимостей через attention.

Архитектура: эмбеддинг items (vocabulary = N_items + pad), обучаемые позиционные эмбеддинги фиксированной длины `max_len`, далее стек из `n_blocks` блоков self-attention. Каждый блок — multi-head self-attention с **causal mask** (нижнетреугольная маска, позиция t видит только ≤ t) + point-wise FFN из двух линейных слоёв с ReLU. Residual + LayerNorm по схеме **post-norm** (как у Vaswani; pre-norm станет стандартом позже — это один из аспектов, который меняет eSASRec). Dropout прикладывается к эмбеддингам, к выходу attention и к выходу FFN.

Цифры из оригинальной статьи: `d = 50` (мало, но работало на маленьких ML-1M / Amazon), `n_blocks = 2`, `n_heads = 1`, dropout `0.2` на ML-1M и `0.5` на разреженных, `max_len = 200` для ML-1M и `50` для Amazon. Adam, lr `1e-3`, batch `128`, без LR scheduler, 200 эпох с early stopping по NDCG@10.

**Loss — тот самый «dross»**, от которого избавляется gSASRec. На каждый позитивный таргет `y_t` (просто следующий item) сэмплируется ровно **один** негатив `y_t⁻` равномерно из items, не встречавшихся у пользователя. BCE:

```
L = − Σ_t [ log σ(r_{y_t, t}) + log(1 − σ(r_{y_t⁻, t})) ]
```

Таргет — shifted-sequence: вход `[i_1..i_{n-1}]`, целевые позитивы `[i_2..i_n]`. Эквивалент next-token prediction.

Эвалюация в оригинале — **leave-one-out** (последний → test, предпоследний → val) ранжирование среди **100 случайных негативов** + 1 позитив. Метрики на ML-1M: HR@10 ≈ 0.823, NDCG@10 ≈ 0.592 (sampled — сильно завышены). Walid Krichene & Steffen Rendle (KDD 2020, «On Sampled Metrics for Item Recommendation», arXiv:2007.13239) формально доказали, что sampled metrics **не сохраняют ранжирование между моделями** — нелинейность top-K фильтра подавляет различия в хвосте. Сообщество с 2020-21 переехало на full-catalog evaluation.

### 1.2 gSASRec и gBCE: «Turning Dross into Gold Loss» (Petrov & Macdonald, RecSys 2023)

К 2020-22 был консенсус: BERT4Rec лучше SASRec, потому что cloze + двунаправленная маска. Petrov & Macdonald в replication study (RecSys 2022) показали: оригинальный BERT4Rec тренировался на порядки дольше, при равном бюджете разрыв во многом исчезает. Оставшийся разрыв они объясняют **не архитектурой, а функцией потерь**.

**Диагноз.** SASRec с одним негативом и BCE производит overconfident скоры — сигмоид «выгорает», вероятности на позитивах прижимаются к 1, на негативах к 0. На full-catalog инференсе логиты для непросмотренных правильных кандидатов оказываются «зажаты»: модель училась на одном vs одном, а не на калиброванной плотности по словарю. BERT4Rec использует softmax по всему словарю (или большой негативной выборке) → калиброванное распределение.

**gBCE (generalized BCE)** интерполирует между ванильной BCE и sampled softmax, явно компенсируя bias от подмножества негативов:

```
L_gBCE = − [ α · log σ(s⁺) + (1/n) · Σ_j log(1 − σ(s_j⁻)) ]
α = (t · (|I| − 1)) / n
```

`t ∈ (0, 1]` — калибровочный параметр. При `t = 1` градиенты gBCE ведут к **тому же оптимуму, что и полный softmax по каталогу** (калиброванные вероятности). При `t → 0` поведение → стандартная BCE с переуверенностью. На практике хорошие значения `t = 0.75` или `0.9` (для ML-1M в статье — 0.75).

Математика. Производная `log σ(s⁺)` по `s⁺` равна `1 − σ(s⁺) = 1 − p⁺`. В стандартной BCE с одним негативом этот градиент тянет p⁺ строго к 1, что неверно для full-catalog ranking (настоящий ожидаемый позитив имеет вероятность `1/|I|`). Множитель α ослабляет давление позитивного градиента ровно настолько, чтобы при оптимуме позитивный скор соответствовал доле позитивов в полном каталоге. Получается importance-sampling-подобная коррекция, но на уровне формы потерь.

`n_neg = 256` — sweet spot для ML-20M. Меньше 64 — теряется калибровка, больше 1024 — диминишинг и память. Архитектура gSASRec **идентична** SASRec: отличие чисто в loss + количестве негативов.

Протокол на ML-20M в gSASRec-статье: **5-core фильтрация** (минимум 5 взаимодействий и у пользователей, и у items), сортировка по timestamp, **leave-one-out** split, `max_len = 200`, **full-catalog evaluation**, метрики NDCG@10 и Recall@10. Числа: SASRec с BCE-1 ≈ NDCG@10 0.10–0.12, BERT4Rec ≈ 0.135, gSASRec ≈ 0.137–0.141. Тренировка: Adam lr 1e-3, batch 128, `d=64`, `n_blocks=2`, `n_heads=2`, dropout 0.2.

> ⚠️ Был зафиксирован конфликт arxiv-ID: задание ссылается на 2309.07602, но эта статья по содержанию — про сравнение SASRec/BERT4Rec и предложение gBCE; «Turning Dross Into Gold Loss» (правильное название) живёт в семействе работ Petrov-Macdonald 2022-2023. На arxiv 2309.07602 — это и есть **gSASRec / gBCE**, проверено WebFetch. Так что одна статья, разные взгляды на её содержание.

### 1.3 eSASRec (arXiv:2508.06450, репа TOPAPEC/esasrec)

«eSASRec» = «enhanced SASRec». Три ортогональных апгрейда: **LiGR-блоки**, **sampled softmax loss**, **shifted-sequence target** (последний — формально как у SASRec, но эксплицитно выделен против MLM/cloze).

**LiGR (Linear/Layered Gated Residual).** Термин LiGR в reco-литературе всплывает в production-моделях Meta и LinkedIn (LinkedIn использовал LiGR как название ranking-фреймворка), в применении к sequential rec как блок-замена self-attention возник в 2024-25. По факту LiGR-блок:

```
y = x + g₁ ⊙ Attention(LayerNorm(x))
z = y + g₂ ⊙ FFN(LayerNorm(y))
```

`g₁, g₂` — обучаемые гейты (часто скалярные, init так, что в начале gate ≈ 0, блок начинает с identity). Та же идея, что ReZero, LayerScale в ViT/CaiT. Это **pre-norm** (LayerNorm перед attention/FFN, а не после residual) + **gated residual** (residual через сигмоид-гейт). На ML-20M даёт стабильное обучение при `n_blocks=3-4` и `d=128-256`.

**Sampled softmax loss** (вариант eSASRec):

```
L_sampled = − log( exp(s⁺) / (exp(s⁺) + Σ_j exp(s_j⁻)) )
```

Часто с **logQ correction** (поправка по вероятности сэмплирования негатива, чтобы убрать bias к популярным items, если негативы сэмплируются by frequency). На больших каталогах sampled softmax работает чуть лучше gBCE при том же `n_neg` — нормализация по всем кандидатам в знаменателе даёт более резкий градиент.

**Shifted-sequence objective** — эксплицитное название «вход — последовательность, таргет — сдвинутая на 1, loss на каждой позиции», против BERT4Rec'овского cloze (mask 15%, предсказать только их). Shifted-sequence даёт `max_len` градиентов с одной последовательности вместо `~0.15·max_len`, в 6-7× больше supervision на ту же выборку — отсюда быстрее тренировка.

**Цифры eSASRec на ML-20M** (по памяти, требует сверки с paper): NDCG@10 ≈ 0.145–0.155, выше gSASRec на 5-10%, сравнимо с PinnerSAGE/HSTU-подобными свежими моделями.

### 1.4 Известные числа на ML-20M в full-catalog протоколе

| Модель | NDCG@10 | Заметки |
|---|---|---|
| Popularity baseline | ~0.05 | Тривиальный |
| BPR-MF, EASE | ~0.09–0.11 | EASE на удивление силён за счёт low-rank структуры ML-20M |
| GRU4Rec (BPR-max / TOP1-max) | ~0.10–0.12 | При правильной тренировке |
| **SASRec ванильный (BCE-1neg)** | **~0.10–0.12** | Страдает от переуверенности |
| BERT4Rec при достаточной тренировке | ~0.135 | |
| **gSASRec** | **~0.137–0.141** | gBCE + 256 негативов |
| **eSASRec** | **~0.145–0.155** | LiGR + sampled softmax |
| HSTU (Meta 2024), SASRec-RoPE-SwiGLU | ~0.16+ | Generative recommenders, дороже на порядок |

**Целевая планка для проекта:**
- NDCG@10 ≥ 0.140 — отличный результат, gBCE/sampled softmax + LiGR работает.
- NDCG@10 ≥ 0.150 — paper-replication уровень.
- NDCG@10 = 0.10–0.12 — где-то ошибка: либо sampled metrics вместо full-catalog, либо loss = ванильная BCE, либо нет 5-core фильтрации.
- NDCG@10 ≥ 0.16 — проверить, не утекают ли тестовые items в train.

### 1.5 Ручки, которые реально влияют на качество

| Гиперпараметр | Дефолт | Замечания |
|---|---|---|
| `hidden_dim` | 128 на ML-20M | 64 → 128 заметный буст; 128 → 256 диминишинг |
| `n_blocks` | 2 (SASRec post-norm), 3-4 (eSASRec pre-norm + LiGR) | >6 переобучение |
| `n_heads` | 2-4 | 1 мало, 8 overkill для d=128 |
| `dropout` | 0.2 (плотный ML-20M), 0.5 (разреженный Amazon) | В 3 местах: emb / attn / FFN |
| `max_len` | 200 | 500 ловит +5% сигнала, но 6× память attention; с FlashAttention ок |
| `batch_size` | 128 paper-default | 256-512 со scaling lr; >1024 ухудшает generalization |
| `n_neg` | 256 | <64 теряет калибровку, >1024 диминишинг |
| `lr` | 1e-3 Adam | + warmup 1-5% и cosine decay стабильнее |
| `weight_decay` | 0 или 1e-2 | На эмбеддинги items decay часто **не** прикладывают |
| `warmup_steps` | 1-5% от total | Без warmup глубокий трансформер может развалиться |
| `gbce_t` | 0.75 | Калибровка под full-catalog |
| AMP | bf16 на A100 | fp16 рискует overflow в softmax |

---

## 2. NextItNet

**Yuan et al., WSDM'19, "A Simple Convolutional Generative Network for Next Item Recommendation"** (arXiv:1808.05163). Кода в репе TOPAPEC нет — пишем самим. Опорный материал: arxiv абстракт + RecBole-имплементация (`recbole/model/sequential_recommender/nextitnet.py`).

### 2.1 Идея

Causal sequential recommender, где attention заменён на стек **dilated 1D causal convolutions** + residual. Сильный конкурент GRU4Rec и Caser в 2018-19. Главная фишка — экспоненциально растущее **receptive field** через dilated свёртки: с дилатациями `[1, 2, 4, 8]` и kernel size 3 после 4 блоков покрытие ≈ 30 позиций, после 8 блоков (повтор паттерна) — около 60.

**Self-check вопрос Данила**: «за счёт чего модель сможет / не сможет уловить связь между айтемами расположенными в истории очень далеко друг от друга?»

Ответ: receptive field dilated conv-стэка растёт как `Σᵢ dilation_i × (k - 1)`. С достаточным числом блоков покрывается вся последовательность. **НО**: информация физически проходит через каждый слой, это не direct attention с O(1) hops. Длинные связи требуют (а) достаточной глубины стека, чтобы receptive field покрывал расстояние, (б) хорошего gradient flow через residual connections. Сравнение: attention имеет O(1) информационную дистанцию между любыми двумя позициями, dilated conv — O(log_dilation). Поэтому dilated CNN всегда **дешевле** трансформера по compute (O(N·k) vs O(N²)), но менее «прямолинеен» для дальних зависимостей.

### 2.2 Архитектура residual block (по RecBole-имплементации)

RecBole использует **ResidualBlock_b** (вариант b лучше варианта a из статьи):

```python
class ResidualBlock_b(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation):
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size),
                               padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size),
                               padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

    def forward(self, x):
        # causal padding: левый padding (kernel_size-1)*dilation, правый 0
        x_pad = F.pad(x, ((kernel_size-1)*dilation, 0, 0, 0))
        out1 = F.relu(self.ln1(self.conv1(x_pad)))
        x_pad2 = F.pad(out1, ((kernel_size-1)*dilation*2, 0, 0, 0))
        out2 = F.relu(self.ln2(self.conv2(x_pad2)))
        return out2 + x   # residual
```

Ключевые детали:
- **Conv2d с kernel `(1, kernel_size)`** — формально 2D, но эффективно 1D по time-axis. Можно заменить на Conv1d (RecBole так делает по историческим причинам).
- **Causal padding**: `ZeroPad2d(((kernel_size - 1) * dilation, 0, 0, 0))` — паддинг ТОЛЬКО слева. Это и обеспечивает причинность: позиция t не видит будущее.
- **Внутри блока 2 свёртки**, у второй dilation удвоен (`dilation * 2`) — это даёт ещё больший receptive field на блок.
- **LayerNorm** + ReLU **после** свёртки (post-activation норм).
- **Residual** в конце.

### 2.3 Forward pass и loss

```python
def forward(self, item_seq):
    item_seq_emb = self.item_embedding(item_seq)   # [B, L, d]
    dilate_outputs = self.residual_blocks(item_seq_emb)
    hidden = dilate_outputs[:, -1, :]              # берём только last position!
    seq_output = self.final_layer(hidden)
    return seq_output
```

**Внимание**: RecBole-имплементация берёт **только последнюю позицию** при инференсе. Но **в оригинальной статье** loss считается на **каждой позиции** — это generative training, как у LM. Стиль NextItNet оригинала: для каждой позиции t предсказываем `i_{t+1}` через full softmax. Это даёт `max_len` градиентов на последовательность вместо одного, что в 200× увеличивает supervision-density. Тренируем generatively, инферим — последняя позиция. **При самостоятельной реализации делаем full-position training**.

**Loss варианты в RecBole**:

```python
# CE (full softmax over catalog)
test_item_emb = self.item_embedding.weight             # [|I|, d]
logits = seq_output @ test_item_emb.T                   # [B, |I|]
loss = nn.CrossEntropyLoss()(logits, pos_items)

# BPR (pairwise)
pos_score = (seq_output * pos_emb).sum(-1)
neg_score = (seq_output * neg_emb).sum(-1)
loss = -F.logsigmoid(pos_score - neg_score).mean()
```

Для full-position обучения логит считается на каждой позиции и cross-entropy усредняется по валидным (не padding) позициям.

### 2.4 Гиперпараметры (paper + типичные)

| Параметр | Значение |
|---|---|
| `kernel_size` | 3 |
| `embedding_size` (= residual_channels) | 64 (paper) → 128-256 (modern) |
| `block_num` | 8-16 |
| `dilations` | `[1, 2, 4, 8]` повторённое `block_num` раз |
| `dropout` | 0.1-0.3 |
| Loss | CE (full softmax) или BPR |
| `max_len` | 100-200 |
| `lr` | 1e-3 (Adam) |
| `batch_size` | 256 |

### 2.5 Чего нет в RecBole-имплементации, но стоит добавить

- **Tied weights** input embedding ↔ output projection (стандартный трюк ALBERT/GPT-2): экономит `|I|·d` параметров, +1-3% NDCG. RecBole этого не делает.
- **Gated activations** (GTU `tanh(x) ⊙ σ(x)` или GLU) вместо ReLU — wavenet-style, +1-3%.
- **Skip connections от каждого блока к выходу** (DenseNet-style aggregation) — +1-2%.
- **Weight tying** + label smoothing 0.1 — стандартный пакет современного training'а.

### 2.6 Открытые имплементации (для сверки)

- `github.com/fajieyuan/WSDM2019-nextitnet` — оригинальный TF от автора
- `github.com/syiswell/NextItNet-Pytorch` — популярный PyTorch port
- `recbole/model/sequential_recommender/nextitnet.py` — production-quality в RecBole (то, что разобрали выше)

### 2.7 Числа NextItNet vs SASRec на ML-20M

NextItNet и SASRec обычно идут «нос к носу» на ML-20M в честных сравнениях — разница в пределах 1-3% NDCG@10 в обе стороны в зависимости от тюнинга. NextItNet **глубже** (8-16 блоков против 2-4 у SASRec), но **дешевле по compute** на блок (свёртки vs квадратичный attention). При L=200 split всё-таки чаще побеждает SASRec за счёт O(1) attention reach.

---

## 3. StackRec и масштабирование вглубь

**Wang et al., SIGIR 2021** (arXiv:2012.07598). «StackRec: Efficient Training of Very Deep Sequential Recommender Models by Layer Stacking».

### 3.1 Главное наблюдение

В **обученном** глубоком sequential recommender (NextItNet, SASRec, GRec) скрытые представления соседних слоёв почти идентичны — косинусное сходство 0.9+. То есть глубокая сеть в каком-то смысле уже состоит из «повторов» одного и того же преобразования. Отсюда — растить глубину **из обученной shallow модели** через копирование блоков, а не учить глубокую с нуля.

### 3.2 Оператор стэкинга

1. Тренируем модель из N блоков до сходимости.
2. Строим сеть из 2N блоков. Веса инициализируются **удвоением**:
   - **Adjacent / interleaved copy** (рекомендуется): блок i → блоки 2i-1 и 2i (каждый старый блок ставится рядом со своей копией). Это **выигрывает** у appended на 0.3-0.8 п.п. NDCG@5.
   - **Sequential / appended copy**: сначала все N оригиналов, потом все N копий ([B1..BN, B1..BN]). Создаёт «провал» в середине → доучивается дольше.
3. Дообучаем (warmup на 5% шагов + основная фаза с пониженным lr).
4. Повторяем: 2N → 4N → 8N. Авторы доходят до **64 блоков** для NextItNet и до **32 блоков** для SASRec-like.

### 3.3 Цифры из статьи

Speedup vs from-scratch на той же глубине:
- 16 блоков: **1.4×**
- 32 блока: **2×**
- 64 блока: **3×**

Качество — паритет или чуть лучше, чем у from-scratch. Speedup растёт с глубиной (доля «дешёвой» предобученной начальной фазы относительно полной глубокой эпохи увеличивается).

### 3.4 Практические замечания

- LayerNorm/BatchNorm-статистики: в interleaved обе копии стартуют с одинаковыми running-mean/var — нормально.
- Learning rate после стэкинга **не повышать** — глубже сеть = риск взрыва градиента. Первые ~5% шагов warmup.
- Residual connections **обязательны** (метод не работает на plain stack).
- Для NextItNet: dilation pattern меняется при interleaved. Если оригинал был `[1,2,4,8,1,2,4,8]`, после стэкинга станет `[1,1,2,2,4,4,8,8,1,1,...]`. Это допустимо.
- Реализация на PyTorch ~50 строк: копирование `state_dict` с переименованием ключей блоков + удвоение `self.blocks` ModuleList.

### 3.5 Соседние подходы к глубине

- **DeepNorm** (Wang 2022): post-LN с домножением residual на `α=(2N)^(1/4)` и init `β=(8N)^(-1/4)`. Тренировали 1000-слойные трансформеры. Для ML-20M избыточно, но идею «отшкалировать residual» полезно знать.
- **ReZero** (Bachlechner 2020): residual `x + α_l · F(x)`, `α_l` инициализируется нулём. На старте сеть тождественная, оживает постепенно. Для SASRec на ML-20M ReZero даёт сходимость на 32 слоях без warmup.
- **Pre-LN vs Post-LN**: Pre-LN численно устойчивее, Post-LN чуть лучше по качеству в сходимости, но требует warmup на глубине >12. Для StackRec — Pre-LN сходится почти без warmup, Post-LN с warmup.
- **T-Fixup** (Huang 2020): инициализация без LayerNorm и без warmup. На практике в reco мало кто использует.
- **Stochastic Depth / LayerDrop** (Huang, Fan): с вероятностью p пропускаем целый residual-блок (forward = identity). Регуляризация + ускорение обучения (~p×) + robust к pruning. Для глубокого SASRec (32 блоков) с p=0.1 ощутимо снижает оверфит.

### 3.6 Прагматичный рецепт «как уйти на 32-64 блока в reco»

**Pre-LN + ReZero (или T-Fixup) + StackRec interleaved + LayerDrop** = современный мейнстрим 2022-2024. Все четыре техники ортогональны.

---

## 4. Efficient attention: Linear, Mamba, FNet

При L=200 (типичный SASRec) квадратичное внимание стоит 200² = 40k dot-products на голову на слой — **копейки** на A100. Linear attention / Mamba / FNet начинают давать win **только с L≥1k** (linear), L≥2k (Mamba), L≥4k (FNet). На ML-20M это больше педагогическое упражнение, чем performance gain. Но один из трёх может **повысить качество** даже на L=200 — спойлер, FNet/FMLP-Rec.

### 4.1 Linear Attention (Katharopoulos et al., ICML 2020, arXiv:2006.16236)

Идея в одной формуле. Заменить экспоненту в softmax на произвольную **неотрицательную** feature map φ: ядро `sim(q,k) = exp(q·k/√d)` → `sim(q,k) = φ(q)ᵀ φ(k)`.

Тогда выход для позиции i:
```
y_i = (Σⱼ φ(q_i)ᵀ φ(k_j) v_j) / (Σⱼ φ(q_i)ᵀ φ(k_j))
    = φ(q_i)ᵀ (Σⱼ φ(k_j) v_jᵀ) / (φ(q_i)ᵀ Σⱼ φ(k_j))
```

Сумма `S = Σⱼ φ(k_j) v_jᵀ` — матрица `d_φ × d_v`, не зависит от i. Строится за один проход `O(N · d_φ · d_v)`. Каждый запрос i стоит `O(d_φ · d_v)`. Итого **O(N) вместо O(N²)** — линейно по длине.

Для **causal** случая (что нам нужно) сумма префиксная: `S_i = Σⱼ≤ᵢ φ(k_j) v_jᵀ`. Обновляется рекуррентно: `S_i = S_{i-1} + φ(k_i) v_iᵀ`. **Это RNN с линейной динамикой состояния** — отсюда название статьи «Transformers are RNNs». На инференсе держим состояние размера `d_φ × d_v + d_φ`, обновляем по одному токену за O(d_φ · d_v) — без необходимости пересчитывать attention над всей историей. O(1) memory, O(N) total inference compute.

Обычно `φ(x) = elu(x) + 1` (положительная, дифференцируемая). Цена: потеря выразительности softmax. На autoregressive задачах теряют 0.5-2 BLEU/NLL (softmax умеет почти-one-hot, elu+1 — нет).

**Семейство**: Performer (random features, ICLR'21), Linformer (low-rank проекция K и V), Longformer (sliding window + global tokens), ReLA (ReLU вместо softmax). На L=200 разница между ними в шумах кросс-валидации.

### 4.2 Mamba / Selective SSM (Gu & Dao, 2023, arXiv:2312.00752)

State-Space Models в непрерывном времени: `ẋ(t) = A x(t) + B u(t)`, `y(t) = C x(t)`. После дискретизации:
```
h_t = Ā h_{t-1} + B̄ x_t
y_t = C h_t
```

Если `Ā, B̄, C` статичны — рекуррентность разворачивается в свёртку с ядром `K = (CB̄, CĀB̄, CĀ²B̄, ...)`, считается через FFT за `O(N log N)`. Это линия S4 (предшественник Mamba).

**Selectivity (главная идея Mamba)**: матрицы B, C и шаг дискретизации Δ становятся **функцией текущего входа** x_t (через линейные проекции). Рекуррентность теперь content-aware: для важных токенов сеть «запоминает» (Δ→большое), для незначимых — «пропускает» (Δ→малое). Аналог forget/input gate в LSTM, но в SSM-параметризации. Цена: ядро K зависит от позиции, нельзя одним FFT.

Решение — **hardware-aware parallel scan** (associative scan, реализован в CUDA так, что промежуточные h_t живут в SRAM/registers и не идут через HBM). Wall-clock как у FlashAttention при O(N) FLOPs vs O(N²) у трансформера.

В reco — **Mamba4Rec** (Liu et al., 2024) ставит Mamba-блок вместо self-attention в SASRec-стиле. На ML-1M, Beauty, Video показывает паритет или +0.5-1.5% NDCG@10 при меньшем inference latency на длинных последовательностях. На ML-20M эффект скромнее — длина сессии ~50-100 не «родная» зона выгоды.

**SSM vs Transformer tradeoff** (блог Albert Gu, goombalab.github.io/blog/2025/tradeoffs/): Transformer хорош для прецизионной ассоциативной памяти по всей истории (точное обращение к конкретному прошлому токену). SSM лучше для устойчивой интеграции длинного контекста с подавлением шума, хуже на retrieval. Reco — пограничный случай.

### 4.3 FNet (Lee-Thorp et al., 2021, arXiv:2105.03824)

Самый радикальный вариант: attention выкидывается полностью, заменяется на **2D FFT** (без обучаемых параметров). На каждом слое:
```
y = Re(F_seq(F_hidden(x)))
```
F_seq — DFT по позициям, F_hidden — DFT по hidden dim. Real-часть берётся для возврата в R. Дальше обычный FFN + LayerNorm + residual.

Интуиция: attention тоже линейный mixer токенов (после фиксации softmax-весов). FFT — фиксированный, но «глобальный» mixer. После него FFN добавляет нелинейность, обучение фокусируется на per-position преобразованиях.

Скорость: FFT через cuFFT — O(N log N), параллелится идеально. На L≥512 — в 2-7× быстрее BERT. На L≥4k — до 12× быстрее. Память O(N).

Качество: на GLUE FNet ≈ **92-97% от BERT**. Гибрид «3 FNet-слоя + 1 attention сверху» — до 99% BERT при сохранении ускорения. **Pattern «дешёвый mixer на ранних слоях + точный mixer на поздних»**.

**FMLP-Rec** (Zhou et al., WWW 2022) — учитываемый filter в частотной области (FFT → element-wise умножение на обучаемый комплексный фильтр → IFFT) вместо attention в SASRec. На стандартных бенчмарках обгоняет SASRec на 1-3% NDCG@10 при меньшем числе параметров. Самый чистый перенос FNet в reco. Может **выиграть** у SASRec на ML-20M даже при L=200.

### 4.4 Что выбрать для проекта на L=200

При L=200 efficient attention **не даёт wall-clock win** — FlashAttention выжимает максимум из плотного attention, а linear attention / Mamba на коротких длинах либо не догоняют по latency, либо уступают на 10-30%. Реально efficient attention начинает выигрывать с L≥1k.

**Прагматичный план — педагогический бенчмарк**: SASRec baseline + 3-4 варианта на той же ML-20M-постановке:
1. **Linear attention** (elu+1, ~30 строк PyTorch) — наглядность RNN-эквивалентности.
2. **Mamba4Rec** через готовую `mamba-ssm` библиотеку — модно, обсудить selective-механизм.
3. **FNet/FMLP-Rec** — самое поучительное; возможно, единственный с шансом **выиграть** SASRec по качеству на L=200 (за счёт неявной регуляризации фильтра в частотной области).
4. **Гибрид** «3 FNet-слоя + 1 attention сверху» — пятый вариант, часто бьёт чистый SASRec по trade-off.

Метрики: NDCG@10, HR@10, latency на batch=256+L=200, peak VRAM. Дополнительно — **synthetic L=2000 эксперимент** (искусственно удлинить сессии) — показать, что выбранная архитектура обобщается на следующее поколение reco с long lifetime user history. Это и есть главный аргумент для защиты.

**Если коротко**: на L=200 побеждает **SASRec + StackRec** (глубже, не шире/быстрее). Linear attention и Mamba — для демонстрации scaling-свойств. Единственный с реальным шансом победить на ML-20M — **FMLP-Rec**.

---

## 5. Протокол MovieLens-20M

### 5.1 Датасет

20,000,263 рейтинга, 138,493 пользователей, 27,278 фильмов, ratings 1-5, timestamps. Файлы: `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `genome-scores.csv`, `genome-tags.csv`. По правилам GroupLens у каждого пользователя ≥ 20 рейтингов уже на стороне создателей — то есть «нулевая» k-core фильтрация уже сделана, что отличает ML-20M от Amazon (где приходится резать жёстко).

### 5.2 Перевод в implicit feedback

Два способа:

1. **Все рейтинги → события** (SASRec, gSASRec, eSASRec): факт того, что юзер поставил оценку, уже сигнал «он смотрел». Сохраняет всю выборку, не привязывает к субъективному порогу.
2. **Бинаризация по rating ≥ 4** (NCF, MultiVAE-style): выкидывает примерно половину строк, сильно меняет распределение длин последовательностей.

Для команды берём **первый вариант** — он канон gSASRec и совпадает с SASRec/eSASRec. Только тройка `(user, item, ts)` с сортировкой по timestamp.

### 5.3 Фильтрация: 5-core

«Повторяй до сходимости — выкинь юзеров с <5 событий, выкинь items с <5 событий». Удаление юзера меняет счётчики items и наоборот → одна итерация не даёт фиксированную точку. Типичная реализация (RecBole) крутит цикл пока за итерацию ничего не выкинуто.

На ML-20M это почти no-op для юзеров (минимум там 20), но заметно режет хвост по фильмам — фильмы с двумя-тремя просмотрами уходят. `|I|` падает с 27k примерно до 18-20k (точная цифра зависит от того, делалась ли пред-фильтрация по rating ≥ 4).

В оригинальном SASRec/gSASRec/eSASRec — именно 5-core. Брать тот же. Иначе цифры не сравнимы.

### 5.4 Train/val/test split

**Leave-one-out (LOO)** — у каждого юзера последний по времени item → test, предпоследний → val, всё что раньше → train. Это то, что используют SASRec, BERT4Rec, gSASRec, S3Rec и почти всё что выросло из SASRec-кода. Плюсы: ровно один positive на юзера в test и в val, метрики HR/NDCG считаются тривиально, train-выборка максимальна. Минус: разные юзеры тестируются в разные моменты времени → формальный leakage по будущему.

**Time-based / global temporal split** — фиксируем глобальную дату (последние N дней / последние X% событий) → test, чуть раньше → val. Устраняет leakage, но у части юзеров в test нет событий (cold), у других — несколько positive'ов (тогда метрики усредняются по событиям, не по юзерам).

**Канон gSASRec** — **leave-one-out + full-catalog evaluation**. Берём это. Та же постановка, что в репе TOPAPEC.

### 5.5 Метрики

В sequential rec в test у каждого юзера **ровно один правильный item** (last item при LOO), формулы упрощаются.

| Метрика | Определение | Замечание |
|---|---|---|
| **HitRate@K = Recall@K** | `1` если true item в топ-K, иначе `0`; среднее по юзерам | Для one-positive Recall = HR (\|relevant\| = 1) |
| **NDCG@K** | `1 / log₂(rank + 1)` если rank ≤ K, иначе `0` | IDCG = 1 (true item на ранге 1), нормализация встроена |
| **MRR(@K)** | `1 / rank` | Без K — самая «информативная», нет порога |
| **Precision@K** | `HR@K / K` | Линейно связана с HR, обычно опускается как избыточная |
| **Coverage@K** | `\|⋃ top-K(u)\| / \|I\|` | Diversity-метрика, не accuracy |

### 5.6 Главная контроверсия: sampled vs full-catalog

**Sampled-100** (S3Rec-style, оригинальный SASRec): true item ранжируется среди 100 случайных негативов. Быстро (O(101) на юзера). Но **Krichene & Rendle KDD 2020** доказали: sampled metrics **не сохраняют ранжирование моделей**. Нелинейность top-K фильтра подавляет различия в хвосте.

**Full-catalog** (gSASRec и далее): ранг true item среди ВСЕХ items. Медленнее, но честно. После 2021 — стандарт.

**Realistic / filtered**: из кандидатов вырезаются items, уже виденные юзером. Ближе к продакшну, но хуже воспроизводится между статьями. gSASRec обычно **не** делает этой фильтрации, чтобы числа были сравнимы с историческими таблицами SASRec/BERT4Rec.

**Для команды берём: full-catalog leave-one-out, метрики HR@10 + NDCG@10 (опционально MRR без K), `max_len = 200`**.

### 5.7 Реализация препроцессинга в PyTorch

Шаги:
1. Прочитать `ratings.csv` → `DataFrame[user, item, ts]`.
2. Применить итеративный 5-core до фиксированной точки.
3. Перенумеровать `item_id` в плотные индексы 0..N−1 (0 резервируется под padding → 1..N), юзеров — в 0..U−1.
4. Сгруппировать по юзеру, отсортировать список items по ts.
5. Для последовательности `[i₁, ..., iₙ]`: `i_n` → test, `i_{n-1}` → val, `[i₁, ..., i_{n-2}]` → train.
6. Обрезать train-префикс до `max_len = 200` справа (последние 200 событий, не первые), укоротить — паддинг слева нулями до 200.
7. Для обучения: shifted-sequence target — подаём `[i₁, ..., i_t]`, предсказываем `i_{t+1}` на каждой позиции (autoregressive teacher forcing), loss суммируется по всем валидным позициям с маской по padding.

Инференс: для юзера u подаём всю его train+val последовательность (последние 200), модель отдаёт hidden state последней позиции, скор `score = h · W_item.T` размерности `|I|`, маскируем padding-индекс (опционально — уже виденные items), берём top-K, считаем ранг true `i_n`, обновляем HR@10 и NDCG@10. Усреднение по всем тестовым юзерам.

---

## 6. Что проверить руками в репе TOPAPEC/esasrec

**Не клонировал — нужно сделать первым делом.** Без этих 4 точек подтверждения цифры из чужой статьи не воспроизводятся, и сравнительная таблица команды разваливается на третьем знаке после запятой.

| Файл | Что проверить |
|---|---|
| `preprocess.py` / `data/prepare.py` | точные `min_user`, `min_item`, `max_len` |
| `configs/*.yaml` или `args.py` | дефолты для ML-20M; интерпретация `max_train_users` / `max_eval_users` (семплинг с фиксированным seed?) |
| `evaluate.py` / `trainer.py` | full-catalog или sampled? есть ли фильтрация уже виденных? |
| split-функция | LOO или time-based? нет ли скрытого leakage через `val ⊂ train`? |
| `loss.py` | точная формула gBCE (если есть) / sampled softmax / BCE |
| `model.py` | LiGR-блоки: pre-norm + gated residual? gates обучаемые скаляры или вектора? |
| CLI флаги | `python -m esasrec.train --help` — что доступно из коробки |

**Команды (на сервере)**:
```bash
git clone https://github.com/TOPAPEC/esasrec.git ~/esasrec_ref
cd ~/esasrec_ref
ls -la
cat README.md
python -m esasrec.train --help 2>&1 | head -100
find . -name "*.py" | head -30
```

И только после этого — сидим, читаем 4 ключевых файла, и фиксируем протокол на бумаге, а не на интуиции.
