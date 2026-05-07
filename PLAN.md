# План работ

> **Статус 2026-05-07.** Решение пользователя: вместо поэтапной реализации (этап 0 → 1 → 2 → ...) сделана **сразу финальная production-grade версия со всеми топ-идеями встроенными изначально**. Этапы 1-4 ниже оставлены как историческая дорожная карта; их фактическое выполнение покрыто `PROGRESS.md` (актуальное состояние) и `NOTES.md` (откуда взято и почему).

Делаем по этапам. Каждый этап — отдельная сессия (см. memory: «дробить сессии — одна глава = один чат»).

## Этап 0 — Setup и проверка инфраструктуры (1-2 часа)

- [ ] Подключиться к GPU-серверу `ssh SkeletonDadGaming@176.109.79.43 -p 2222`
- [ ] Проверить `nvidia-smi`, `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] Создать приватный github-репо `ml_recsys`
- [ ] Сгенерить ssh-ключ на сервере, добавить в github
- [ ] `git clone` на сервер в `~/recsys_project`
- [ ] Создать venv с `--system-site-packages` (чтобы видеть глобальный torch)
- [ ] `pip install numpy pandas scikit-learn tqdm matplotlib`
- [ ] Скачать ML-20M в `~/recsys_project/data/ml-20m/`
- [ ] **Клонировать TOPAPEC/esasrec в `~/esasrec_ref`** для сверки (не работаем там, только читаем)
- [ ] Прочитать 4 ключевых файла в TOPAPEC/esasrec и записать дословный протокол в `RESEARCH.md` секцию 6

**Гейт**: получены HR@10/NDCG@10 цифры дефолтного запуска `python -m esasrec.train --epochs 1 --max-train-users 1000`. Это smoke-test, что инфра работает.

## Этап 1 — Воспроизведение baseline'ов (1-2 недели)

Цель: получить сопоставимые цифры на одном протоколе для двух моделей (SASRec и/или NextItNet).

### Этап 1a — SASRec через TOPAPEC/esasrec

- [ ] Запустить дефолтный SASRec (не eSASRec) на полном ML-20M с `--epochs 100 --batch-size 128 --amp`
- [ ] Сверить полученные метрики с paper-числом gSASRec (NDCG@10 ≈ 0.137-0.141)
- [ ] Если расхождение >10% — отлаживать (проверить препроцессинг, сплит, evaluation, loss)
- [ ] Зафиксировать чекпойнт + конфиг + метрики в `runs/sasrec_baseline/`

**Самопроверка по математике (на листочке, как просил Данил):**
- [ ] Выписать формулу gBCE и осознать роль `t`
- [ ] Выписать формулу attention с causal mask, понять why диагональная маска
- [ ] Объяснить что такое позиционные эмбеддинги и зачем они нужны
- [ ] Объяснить роль PointWiseFFN vs роль Attention (smoothing vs token mixing)
- [ ] Расписать формулы NDCG@K, HR@K, MRR — в чём разница

### Этап 1b — NextItNet (свой код)

Cтрого по `RESEARCH.md` секция 2. По мотивам RecBole, с улучшениями.

- [ ] Реализовать `ResidualBlock_b` с causal padding + LayerNorm + ReLU
- [ ] Реализовать `NextItNet` модель: embedding → стек блоков → final_layer
- [ ] Добавить **tied weights** (input emb = output projection — RecBole не делает, мы добавим)
- [ ] Реализовать full-position training (loss на каждой валидной позиции)
- [ ] Препроцессинг — **тот же**, что использовали для SASRec (один pickle на оба эксперимента)
- [ ] Тренировка: 16 блоков, dilations `[1,2,4,8]×4`, kernel=3, d=128, lr=1e-3 Adam, 100 epochs
- [ ] Сверить метрики — NextItNet должен быть в пределах ±3% NDCG@10 от SASRec на ML-20M

**Самопроверка по NextItNet:**
- [ ] Объяснить **dilated convolution** на пальцах: формула, receptive field растёт как `Σ dilation_i × (k-1)`
- [ ] Объяснить ответ на контрольный вопрос Данила: за счёт чего модель ловит длинные зависимости (receptive field растёт экспоненциально, но информация физически проходит через каждый слой → нужна глубина + хороший gradient flow через residual connections; vs attention с O(1) hops)
- [ ] Расписать softmax loss и shifted target

**Гейт перед этапом 2**: на руках 2 сопоставимые таблицы метрик (SASRec и NextItNet) на одном протоколе ML-20M, с зафиксированными чекпойнтами и конфигами.

## Этап 2 — Улучшения по IDEAS.md (2-3 недели)

Берём топ-5 идей из IDEAS.md, добавляем по одной с фиксированными остальными гиперпараметрами.

### Очередность

1. **Полная full-catalog evaluation с filter-seen** — fix эвалюации, базовая гигиена. Чтобы все следующие изменения честно мерились.
2. **gBCE → sampled softmax с logQ correction** — ожидаемый прирост +1-2% NDCG.
3. **Side info: genome-scores + жанры + декада** — ожидаемый +2-4% NDCG. День работы.
4. **Современный training stack**: AdamW + warmup-cosine + bf16 + grad clip + EMA — +3-5%.
5. **Архитектурный апгрейд**: Pre-LN + RoPE + SwiGLU + LiGR — +3-6%.
6. **SSE-PT + embedding dropout** — гигиенический +2-3%.

После каждого шага — фиксировать чекпойнт + метрики + diff конфига в `runs/<имя_эксперимента>/`. Это и будет история улучшений для финальной презентации.

**Гейт перед этапом 3**: NDCG@10 ≥ 0.14, HR@10 ≥ 0.34. Если нет — отлаживать, не идти дальше.

## Этап 3 — Scaling по StackRec (1 неделя)

По `RESEARCH.md` секция 3. Только для SASRec — для NextItNet это уже paper-baseline.

- [ ] Тренировать 4-блочный SASRec до сходимости
- [ ] Реализовать **interleaved stacking**: 4 → 8 блоков, копирование `state_dict` с переименованием
- [ ] Дообучить с warmup 5%
- [ ] Повторить 8 → 16
- [ ] Сравнить:
  - 4 блока from-scratch
  - 16 блоков from-scratch
  - 16 блоков через StackRec (4 → 8 → 16)
- [ ] Wall-clock и финальные метрики обоих 16-блочных вариантов
- [ ] Записать в `RESEARCH.md` дополнение: получился ли paper-speedup ×2-3

**Гейт перед этапом 4**: понимание получилось / не получилось у StackRec и почему.

## Этап 4 — Efficient attention (опционально, по желанию Данила)

По `RESEARCH.md` секция 4. Цель — педагогическое сравнение.

- [ ] Реализовать **Linear Attention** (elu+1) — замена attention в SASRec, всё остальное идентично
- [ ] Использовать готовый **Mamba** через `pip install mamba-ssm`, замена блоков
- [ ] Реализовать **FNet** (FFT-mixer) или **FMLP-Rec** (learnable freq filter)
- [ ] Сравнить на ML-20M (L=200): NDCG@10, HR@10, latency, peak VRAM
- [ ] **Synthetic L=2000 эксперимент** — искусственно удлинить сессии (повторение/конкатенация юзеров) и показать, как меняется latency vs SASRec при росте L

## Подытог: финальная таблица для защиты

| Архитектура | Loss | Side info | Train stack | NDCG@10 | HR@10 | Wall-clock | Параметры |
|---|---|---|---|---|---|---|---|
| SASRec baseline | BCE-1neg | ❌ | дефолт | ? | ? | ? | ? |
| SASRec + gBCE | gBCE n=256 | ❌ | дефолт | ? | ? | ? | ? |
| SASRec + gBCE + side | gBCE n=256 | ✅ | дефолт | ? | ? | ? | ? |
| SASRec + всё | sampled SM | ✅ | modern | ? | ? | ? | ? |
| eSASRec | sampled SM | ❌ | modern | ? | ? | ? | ? |
| NextItNet baseline | CE full SM | ❌ | дефолт | ? | ? | ? | ? |
| NextItNet + всё | sampled SM | ✅ | modern | ? | ? | ? | ? |
| StackRec SASRec 16b | sampled SM | ✅ | modern | ? | ? | ? | ? |
| FMLP-Rec | sampled SM | ✅ | modern | ? | ? | ? | ? |
| Linear Attn SASRec | sampled SM | ✅ | modern | ? | ? | ? | ? |
| Mamba4Rec | sampled SM | ✅ | modern | ? | ? | ? | ? |
| Ансамбль | — | ✅ | — | ? | ? | ? | ? |

После каждого «?» становится числом — записываем в `PROGRESS.md` (заведём отдельно по факту).

## Что критично сразу написать Данилу

После Setup (этап 0):
- Подтвердить, что ты — один в команде
- Уточнить дедлайны этапов 1, 2, 3
- Спросить какой именно протокол evaluation (LOO + full-catalog или другой) считается каноном для команды
- Узнать слот созвона на вторник
