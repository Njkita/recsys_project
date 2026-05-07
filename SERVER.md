# Серверы и workflow

## GPU-сервер (выдан Данилом, общий)

```
host:        x32-gpu-01
external IP: 176.109.79.43:2222
location:    Коломна, Московская обл (МТС, AS60490)
OS:          Ubuntu 22.04.5 LTS, kernel 6.8.0-106
user:        SkeletonDadGaming (нерут, своя учётка)
home:        /home/SkeletonDadGaming
```

**Железо:**
- GPU: **NVIDIA A100-SXM4-80GB**, driver 570.211.01, CUDA 12.8, **сейчас полностью свободна** (0 MiB used, 0 процессов)
- CPU: Intel Xeon Gold 6348 @ 2.6GHz, 32 vCPU (1 thread/core, 32 socket — это виртуалка с проброшенными ядрами, NUMA 2 ноды)
- RAM: 157 GB total, 153 GB available, swap 8 GB
- Диск: 784 GB ext4 (LVM), 125 GB свободно (84% занято — сервер коммунальный, в /home кэши других)

**Софт:**
- Python 3.10.12 (`/usr/bin/python3`), pip 22.0.2
- torch 2.5.1+cu121 уже установлен глобально (но без numpy — ставим в свой venv)
- git 2.34.1
- nvcc нет (для PyTorch не нужен)

**Машина общая.** A100 одна на несколько команд. Сейчас свободна, но возможна очередь — следить через `nvidia-smi`.

## Локальная машина

Где сейчас работаем: `/z_hw_/recsys_project`. Это рабочая копия, отсюда пушим на GPU-сервер.

## Workflow синка

Самый чистый путь — **git** через приватный репо на GitHub. Один раз настраиваем, дальше push/pull.

**Шаг 1 (локально, один раз):** инициализация и приватный репо.

```bash
cd /z_hw_/recsys_project
git init -b main
git add .
git commit -m "feat: production sweep on ML-20M"
git remote add origin git@github.com:<логин>/ml_recsys.git
git push -u origin main
```

**Шаг 2 (на сервере, один раз):** клонировать и одной командой поднять окружение.

```bash
ssh SkeletonDadGaming@176.109.79.43 -p 2222
ssh-keygen -t ed25519 -C "x32-gpu-01"           # ключ для github
cat ~/.ssh/id_ed25519.pub                       # добавить в github → ssh keys
git clone git@github.com:<логин>/ml_recsys.git ~/recsys_project
cd ~/recsys_project

bash scripts/setup_server.sh                    # venv + системный torch + numpy/pandas/yaml
bash scripts/download_data.sh                   # ml-20m.zip, ~190 MB
bash scripts/preprocess.sh                      # 5-core + LOO → data/processed.pkl
bash scripts/install_mamba.sh                   # опционально, для Mamba4Rec

# полный sweep — ~24-72ч
bash scripts/train_all.sh
# или одна модель
bash scripts/train.sh configs/sasrec.yaml runs/sasrec_modern
# benchmark latency / VRAM / throughput
bash scripts/benchmark.sh
```

**Цикл работы:** локально пишем → `git commit -am "msg" && git push` → на сервере `git pull` → запускаем.

## Альтернатива: VSCode Remote-SSH

Так как ты в VSCode — можно поставить расширение **Remote - SSH** и открыть workspace прямо на сервере (`SkeletonDadGaming@176.109.79.43:2222`). Тогда правки идут сразу на сервере, git нужен только для бэкапа/истории. Удобно если тестируешь много мелких изменений.

## Альтернатива: rsync (без git)

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='artifacts' \
  -e "ssh -p 2222" /z_hw_/recsys_project/ SkeletonDadGaming@176.109.79.43:~/recsys_project/
```

## Куда складывать датасет ML-20M

На сервере диск 84% занят. ratings.csv (ML-20M) ~500 MB. Лучше класть **не в home** (кэши других пользователей могут забить), а в `~/recsys_project/data/` с тем расчётом что артефакты тренировки могут разрастись до 10–20 GB. Если будет тесно — `du -sh ~` периодически и чистить старые чекпойнты.

Скачать прямо на сервер:
```bash
mkdir -p ~/recsys_project/data && cd ~/recsys_project/data
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
# получим ml-20m/ratings.csv ~ 500MB
```

## Изоляция от арбитражного прода

Этот проект **не пересекается** с твоим HFT/Solana окружением. GPU-сервер отдельный, в Москве (а не Frankfurt/NL), своя учётка. Никакие зависимости/конфиги/cron-задачи от dex-dex/highload_system здесь не нужны и не используются.