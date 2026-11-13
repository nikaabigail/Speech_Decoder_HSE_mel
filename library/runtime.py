# -*- coding: utf-8 -*-
"""
Единые runtime-утилиты для test bench (Фаза 0):
  - DEVICE      : cuda если доступна, иначе cpu (device-agnostic запуск);
  - set_seed()  : фиксация генераторов случайности для воспроизводимости;
  - make_split(): честный train/val/test сплит по файлам (без утечки val==test).

Никаких тяжёлых зависимостей кроме numpy/torch.
"""
import os
import random

import numpy as np
import torch

# --- Устройство: GPU если есть, иначе CPU. Заменяет жёсткие .cuda(). ---
_DEVICE_REQUEST = os.environ.get("ECOG_DEVICE", "auto")
if _DEVICE_REQUEST not in ("auto", "cpu", "cuda"):
    raise ValueError("ECOG_DEVICE must be auto, cpu or cuda")
if _DEVICE_REQUEST == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is unavailable")
DEVICE = torch.device("cuda" if _DEVICE_REQUEST == "cuda" or
                      (_DEVICE_REQUEST == "auto" and torch.cuda.is_available()) else "cpu")


def device_str():
    """Человекочитаемое описание устройства, например 'cuda (NVIDIA GeForce RTX 4090)'."""
    if DEVICE.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    return "cpu (GPU не найдена!)"

# Сид по умолчанию (можно переопределить через переменную окружения BENCH_SEED)
SEED = int(os.environ.get("BENCH_SEED", "42"))


def set_seed(seed=SEED):
    """Фиксирует random/numpy/torch для воспроизводимости. Возвращает использованный seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # детерминизм cuDNN: чтобы разброс по сидам отражал СИД, а не аппаратный шум
        # (важно для парного мульти-сид сравнения целей mel vs Whisper)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def make_split(n_files, test_start):
    """
    Честный сплит по индексам файлов.

      test  = files[test_start:]   -- ОТЛОЖЕННЫЙ тест, только для финального отчёта;
                                       НЕ участвует в early stopping / выборе модели.
      val   = последний train-файл -- для early stopping (выбора модели).
      train = остальные ранние файлы.

    Это убирает баг исходного кода, где X_val и X_test были одним и тем же срезом,
    из-за чего early stopping подсматривал в тестовую выборку.

    Возвращает (train_idx, val_idx, test_idx) — списки индексов файлов.
    """
    # подстраховка для краевых случаев (например, debug с малым числом файлов)
    test_start = max(1, min(int(test_start), n_files - 1))
    test_idx = list(range(test_start, n_files))
    if test_start >= 2:
        val_idx = [test_start - 1]
        train_idx = list(range(0, test_start - 1))
    else:
        # доступен лишь один обучающий файл (только debug/smoke): val совпадает с train.
        # Для реальных метрик такой режим не использовать.
        val_idx = [0]
        train_idx = [0]
    return train_idx, val_idx, test_idx
