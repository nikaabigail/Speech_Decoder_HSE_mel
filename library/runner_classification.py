"""Historical frozen hidden-state word classification only."""
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
from .runner_common import MODEL_DUMPS_DIR, RESULTS_DIR, REGRESSION_MODE, WORDS_REMAP, data_generator
from . import bench_models_regression, bench_models_classification
from .runtime import DEVICE, SEED, set_seed, make_split, device_str
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import copy
import sklearn.preprocessing

MAX_ITERATIONS_COUNT = 10_000
METRIC_ITERATIONS = 1_000
EARLY_STOP_STEPS = 5_000
HIDDEN_STRIDE = 10


def process_batch(bench_model, generator, is_train, iteration, max_words_length):
    loss_function = nn.CrossEntropyLoss()

    if is_train:
        bench_model.model.train()
    else:
        bench_model.model.eval()

    x_batch, y_batch = next(generator)
    x_batch = prepare_x_batch_for_net(x_batch, max_words_length)
    non_silent_indexes = np.where(y_batch != 0)[0]

    assert x_batch.shape[0] == y_batch.shape[0]

    x_batch = torch.FloatTensor(x_batch).to(DEVICE)
    y_batch = torch.LongTensor(y_batch).to(DEVICE)

    if is_train:
        bench_model.optimizer.zero_grad()

    y_predicted = bench_model.model(x_batch)
    assert not torch.any(torch.isnan(y_predicted))

    loss = loss_function(y_predicted, y_batch)

    if is_train:
        loss.backward()
        bench_model.optimizer.step()

    assert y_predicted.shape[0] == y_batch.shape[0], f"{y_predicted.shape[0]} != {y_batch.shape[0]}"

    metrics = {}

    y_predicted_numpy = y_predicted.cpu().detach().numpy().argmax(axis=1)
    y_batch_numpy = y_batch.cpu().detach().numpy()

    metrics["loss"] = float(loss.cpu().detach().numpy())

    metrics["accuracy"] = float(np.mean(y_predicted_numpy == y_batch_numpy))
    if len(non_silent_indexes) > 0:
        metrics["accuracy (without silent class)"] = float(np.mean(y_predicted_numpy[non_silent_indexes] == y_batch_numpy[non_silent_indexes]))

    for key, value in metrics.items():
        bench_model.logger.add_value(key, is_train, value, iteration)

    return metrics


def load_words_info(filepath):
    phrases_info = []
    with open(filepath, encoding="utf-8") as phrases_file:
        for row in phrases_file:
            row = row.strip()
            if len(row) == 0:
                continue
            splitted_row = row.split("\t")
            splitted_row[0] = int(splitted_row[0])
            splitted_row[1] = int(splitted_row[1])
            assert splitted_row[2] in WORDS_REMAP, f"phrase {splitted_row[2]} not allowed"
            phrases_info.append(splitted_row)
    return phrases_info


def get_random_predictions(bench_model, generator, iterations, max_words_length):
    Y_batch = []
    Y_predicted = []
    for index, (x_batch, y_batch) in enumerate(generator):
        x_batch = prepare_x_batch_for_net(x_batch, max_words_length)
        x_batch = torch.FloatTensor(x_batch).to(DEVICE)
        y_predicted = bench_model.model(x_batch).cpu().detach().numpy().argmax(axis=1)
        assert x_batch.shape[0] == y_predicted.shape[0]
        Y_predicted.append(y_predicted)
        Y_batch.append(y_batch)
        if index > iterations:
            break

    Y_predicted = np.concatenate(Y_predicted, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    return Y_batch, Y_predicted


def get_full_predictions(bench_model, X, Y, max_words_length, batch_size):
    """ДЕТЕРМИНИРОВАННЫЙ один проход по (X, Y): каждый трайл ровно один раз
    (в отличие от get_random_predictions с np.random.choice = выборкой С ВОЗВРАЩЕНИЕМ,
    которая раздувает мнимую точность n до ~50k из ~сотни реальных трайлов). Даёт
    честный point-estimate и истинное n для CI."""
    y_true, y_pred = [], []
    bench_model.model.eval()
    with torch.no_grad():
        for x_batch, y_batch in batch_iterator(X, Y, batch_size):
            xb = prepare_x_batch_for_net(x_batch, max_words_length)
            xb = torch.FloatTensor(xb).to(DEVICE)
            pred = bench_model.model(xb).cpu().detach().numpy().argmax(axis=1)
            y_true.extend(list(np.asarray(y_batch)))
            y_pred.extend(list(pred))
    return np.array(y_true), np.array(y_pred)


def _wilson_ci(k, n, z=1.96):
    """95% Wilson доверительный интервал для доли k/n (без scipy)."""
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [max(0.0, center - half), min(1.0, center + half)]


def predict_regression_hidden(bench_model_regression, ecog, stride=HIDDEN_STRIDE):
    """
    Извлекает внутреннее представление энкодера регрессии (features_scaled,
    "hidden state перед FC") по всей записи, прорежённое во времени в `stride` раз
    (Фаза 2.1, вариант A). Возвращает массив (ceil(T_down/stride), hidden_dim),
    выровненный по таймбейзу T_down: строка k соответствует таймпоинту k*stride.
    """
    lag_b = bench_model_regression.LAG_BACKWARD
    lag_f = bench_model_regression.LAG_FORWARD
    gen = data_generator(ecog, [], bench_model_regression.BATCH_SIZE, lag_b, lag_f, shuffle=False, infinite=False)
    bench_model_regression.model.eval()

    hidden_dim = bench_model_regression.model.final_out_features
    t_down = ecog.shape[0]
    n_sub = (t_down + stride - 1) // stride
    x_sub = np.zeros((n_sub, hidden_dim), dtype="float32")

    global_i = lag_b  # индекс центра первого валидного окна в таймбейзе T_down
    for ecog_batch in gen:
        ecog_batch = torch.FloatTensor(ecog_batch).to(DEVICE)
        with torch.no_grad():
            h = bench_model_regression.model(ecog_batch, return_hidden=True).cpu().numpy()
        b = h.shape[0]
        idx = np.arange(global_i, global_i + b)
        mask = (idx % stride) == 0
        if mask.any():
            x_sub[idx[mask] // stride] = h[mask]
        global_i += b
    return x_sub


def prepare_frames(x, words_info, downsampling_coef):
    x_frames = []
    classes = []

    last_phrase_end = 0
    for phrase_start, phrase_end, phrase in words_info:
        if last_phrase_end != 0 and last_phrase_end != phrase_start:
            assert last_phrase_end < phrase_start
            x_frames.append(x[int(last_phrase_end / downsampling_coef):int(phrase_start / downsampling_coef)])
            classes.append(WORDS_REMAP['silent'])

        x_frames.append(x[int(phrase_start / downsampling_coef):int(phrase_end / downsampling_coef)])
        classes.append(WORDS_REMAP[phrase])
        last_phrase_end = phrase_end

    return x_frames, classes


def fix_class_imbalance(X, Y):
    non_silent_indexes = np.where(Y != 0)[0]
    possible_classes = len(set(Y))
    silent_indexes = np.where(Y == 0)[0]

    selected_silent_indexes = np.random.choice(silent_indexes, int(len(non_silent_indexes) * 1.0 / possible_classes))
    selected_indexes = np.sort(np.concatenate([selected_silent_indexes, non_silent_indexes]))

    X = X[selected_indexes]
    Y = Y[selected_indexes]

    assert len(X) == len(Y)
    return X, Y


def batch_iterator(X, Y, batch_size=1):
    length = len(X)
    for index in range(0, length, batch_size):
        current_slice = slice(index, min(index + batch_size, length))
        yield copy.deepcopy(X[current_slice]), copy.deepcopy(Y[current_slice])


def random_iterator(X, Y, batch_size=1):
    random_core = np.arange(0, len(X))
    while True:
        current_indexes = np.random.choice(random_core, batch_size)
        yield copy.deepcopy(X[current_indexes]), copy.deepcopy(Y[current_indexes])


def prepare_x_batch_for_net(x_batch, max_words_length):
    x_batch = [x.transpose() for x in x_batch]
    for i in range(len(x_batch)):
        if x_batch[i].shape[1] >= max_words_length:
            x_batch[i] = x_batch[i][:, :max_words_length]
        else:
            x_batch[i] = np.pad(x_batch[i], pad_width=[(0, 0), (0, max_words_length - x_batch[i].shape[1])])
    return np.array(x_batch)


def split_name(filename_):
    filename = copy.deepcopy(filename_).split("/")[-1]
    filename = ".".join(filename.split(".")[:-1])
    mode, patient, model_name, date = filename.split("___")
    return mode, patient, model_name, date


def get_words_filepath(data_filepath):
    return ".".join(data_filepath.split(".")[:-1]) + "_words.txt"


def calc_max_words_length(X):
    lenth_list = []
    for file_x in X:
        for x in file_x:
            lenth_list.append(x.shape[0])
    return int(np.percentile(lenth_list, 95))


def _idx2word():
    return {v: k for k, v in WORDS_REMAP.items()}


def _train_word_classifier(X, Y, output_size, patient, regression_bench_model_name, date, tag, is_debug, classifier_bench_class=None):
    """
    Train the retained hidden-state word classifier.

    Вход:
      X, Y -- списки по файлам: X[i] -- object-массив кадров (мел-спектрограмма
              слова/тишины), Y[i] -- массив классов.
      tag  -- метка режима для имени файлов/результата:
              'classification_hidden' -- frozen encoder hidden -> word.

    Делает: честный сплит, обучение Mel2WordHidden с early stopping,
    accuracy(train/val/test) + confusion matrix + per-word recall, запись JSON.
    """
    # Historical behavior: width uses duration metadata from all records.
    # Preserved for old checkpoints; this is not metadata-blind evaluation.
    max_words_length = calc_max_words_length(X)
    if classifier_bench_class is None:
        classifier_bench_class = bench_models_classification.Mel2WordHidden
    bench_model = classifier_bench_class(output_size, patient, regression_bench_model_name)

    test_start_file_index = bench_model.TEST_START_FILE_INDEX if not is_debug else 1

    # Честный сплит: test = отложенные файлы (только отчёт), val = последний
    # train-файл (early stopping), train = остальные. Убирает баг val == test.
    train_idx, val_idx, test_idx = make_split(len(X), test_start_file_index)
    print(f"[split] train_files={train_idx} val_files={val_idx} test_files={test_idx}")

    X_train = np.concatenate([X[i] for i in train_idx], axis=0)
    Y_train = np.concatenate([Y[i] for i in train_idx], axis=0)

    X_val = np.concatenate([X[i] for i in val_idx], axis=0)
    Y_val = np.concatenate([Y[i] for i in val_idx], axis=0)

    X_test = np.concatenate([X[i] for i in test_idx], axis=0)
    Y_test = np.concatenate([Y[i] for i in test_idx], axis=0)

    assert X_train.shape[0] == Y_train.shape[0]
    assert X_val.shape[0] == Y_val.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]


    batch_size = bench_model.BATCH_SIZE
    train_generator = random_iterator(X_train, Y_train, batch_size)
    val_generator = random_iterator(X_val, Y_val, batch_size)
    test_generator = random_iterator(X_test, Y_test, batch_size)

    max_metric = -float("inf")
    effective_tag = tag
    model_filename = f"{effective_tag}___{patient['name']}___{regression_bench_model_name}___{date}"
    model_path = f"{MODEL_DUMPS_DIR}/{model_filename}.pth"
    max_iterations_count = MAX_ITERATIONS_COUNT if not is_debug else 1_000
    best_iteration = 0

    for iteration in range(max_iterations_count):
        process_batch(bench_model, train_generator, True, iteration, max_words_length)
        with torch.no_grad():
            process_batch(bench_model, val_generator, False, iteration, max_words_length)
            is_last_iteration = iteration == (max_iterations_count - 1)
            if iteration % 1000 == 0 or is_last_iteration:
                smoothed_metric = bench_model.logger.get_smoothed_value("accuracy")
                if smoothed_metric >= max_metric:
                    max_metric = smoothed_metric
                    best_iteration = iteration
                    torch.save(bench_model.model.state_dict(), model_path)
                else:
                    assert iteration >= best_iteration
                    if (iteration - best_iteration) > EARLY_STOP_STEPS:
                        print(f"Stopping model. Iteration {iteration} {round(smoothed_metric, 2)}. Best iteration {best_iteration} {round(max_metric, 2)}.")
                        break

    bench_model.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    bench_model.model.eval()

    result = {}
    result["train_accuracy"] = float(accuracy_score(*get_random_predictions(bench_model, train_generator, METRIC_ITERATIONS, max_words_length)))
    result["val_accuracy"] = float(accuracy_score(*get_random_predictions(bench_model, val_generator, METRIC_ITERATIONS, max_words_length)))
    y_true_test, y_pred_test = get_random_predictions(bench_model, test_generator, METRIC_ITERATIONS, max_words_length)
    result["test_accuracy"] = float(accuracy_score(y_true_test, y_pred_test))

    # --- ЧЕСТНАЯ метрика теста: детерминированный один проход + истинное n + Wilson CI ---
    # (test_accuracy выше — выборка С ВОЗВРАЩЕНИЕМ -> раздутое мнимое n; ниже — каждый
    #  тест-трайл ровно один раз. Аддитивно: старое поле не трогаем для совместимости.)
    yt_full, yp_full = get_full_predictions(bench_model, X_test, Y_test, max_words_length, batch_size)
    n_test = int(len(yt_full))
    k_correct = int((yt_full == yp_full).sum()) if n_test else 0
    result["test_accuracy_full"] = float(accuracy_score(yt_full, yp_full)) if n_test else None
    result["n_test_trials"] = n_test
    result["test_ci95"] = _wilson_ci(k_correct, n_test)

    # --- Отчётность Фазы 1: confusion matrix 27x27 + per-word recall ---
    n_classes = len(WORDS_REMAP)
    cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(n_classes)))
    idx2word = _idx2word()
    row_sums = cm.sum(axis=1)
    per_word_recall = {}
    for i in range(n_classes):
        per_word_recall[idx2word.get(i, str(i))] = (float(cm[i, i] / row_sums[i]) if row_sums[i] > 0 else None)
    result["test_confusion_matrix"] = cm.tolist()
    result["test_per_word_recall"] = per_word_recall
    result["chance_level"] = 1.0 / n_classes

    result["train_logs"] = bench_model.logger.train_logs
    result["val_logs"] = bench_model.logger.test_logs
    result["iterations"] = iteration
    result["config"] = {
        "mode": tag,
        "control": "none",
        "augment": "none",
        "regression_model": regression_bench_model_name,
        "patient": patient["name"],
        "seed": SEED,
        "device": str(DEVICE),
        "split": {"train_files": train_idx, "val_files": val_idx, "test_files": test_idx},
        "best_iteration": best_iteration,
        "max_words_length": int(max_words_length),
    }

    with open(f'{RESULTS_DIR}/{model_filename}.json', 'w') as result_file:
        json.dump(result, result_file)

    print(f"[result] {tag}: train_acc={result['train_accuracy']:.3f} "
          f"val_acc={result['val_accuracy']:.3f} test_acc={result['test_accuracy']:.3f} "
          f"(chance={result['chance_level']:.3f})")
    return result


def run_hidden(regression_bench_model_name, patient, is_debug=False):
    """Train a word head on frozen, stride10 encoder hidden states."""
    assert hasattr(bench_models_regression, regression_bench_model_name)
    set_seed(SEED)
    print(f"[runtime] device={device_str()} | seed={SEED} | HIDDEN-STATE -> word (stride={HIDDEN_STRIDE})")
    bench_regression_model = getattr(bench_models_regression, regression_bench_model_name)(patient)

    ecog_preprocessed_cache = []
    for filepath in patient["files_list"]:
        with h5py.File(filepath, 'r') as input_file:
            data = input_file['RawData']['Samples'][()]
        ecog = data[:, patient["ecog_channels"]].astype("double")
        x = bench_regression_model.preprocess_ecog(ecog, patient["sampling_rate"]).astype("float32")
        ecog_preprocessed_cache.append(x)
        if is_debug and len(ecog_preprocessed_cache) >= 2:
            break

    all_models_files = []
    for filename in os.listdir(MODEL_DUMPS_DIR):
        if not filename.endswith(".pth"):
            continue
        mode, patient_name, model_name, date = split_name(filename)
        if mode == REGRESSION_MODE and model_name == regression_bench_model_name and patient_name == patient["name"]:
            all_models_files.append(filename)

    for regression_model_filename in all_models_files:
        bench_regression_model = getattr(bench_models_regression, regression_bench_model_name)(patient)
        print("Start File:", regression_model_filename)
        regression_model_file_path = f"{MODEL_DUMPS_DIR}/{regression_model_filename}"
        _, patient_name, _, date = split_name(regression_model_filename)
        assert patient_name == patient["name"]
        bench_regression_model.model.load_state_dict(torch.load(regression_model_file_path, map_location=DEVICE, weights_only=True))

        for parameter in bench_regression_model.model.parameters():
            parameter.requires_grad_(False)

        hidden_dim = bench_regression_model.model.final_out_features
        eff_downsampling = bench_regression_model.downsampling_coef * HIDDEN_STRIDE

        X = []
        Y = []
        for index, filepath in enumerate(patient["files_list"]):
            with h5py.File(filepath, 'r') as input_file:
                data = input_file['RawData']['Samples'][()]
            words_info = load_words_info(get_words_filepath(filepath))

            ecog = ecog_preprocessed_cache[index]
            x = predict_regression_hidden(bench_regression_model, ecog, HIDDEN_STRIDE)
            x = sklearn.preprocessing.scale(x, copy=False)
            x_frames, classes = prepare_frames(x, words_info, eff_downsampling)
            x_frames = np.array(x_frames, dtype=object)
            classes = np.array(classes)
            assert x_frames.shape[0] == classes.shape[0]
            x_frames, classes = fix_class_imbalance(x_frames, classes)
            X.append(x_frames)
            Y.append(classes)
            if is_debug and len(X) >= 2:
                break

        _train_word_classifier(
            X, Y, hidden_dim, patient,
            regression_bench_model_name, date, "classification_hidden", is_debug,
            classifier_bench_class=bench_models_classification.Mel2WordHidden
        )
