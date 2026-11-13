"""Historical MSE stage; augmentation and word-supervision branches removed."""
import json
import datetime
import h5py
import numpy as np
import torch
import torch.nn as nn
from .runner_common import data_generator
from . import bench_models_regression, whisper_target
from .runtime import DEVICE, SEED, set_seed, make_split, device_str

MAX_ITERATIONS_COUNT = 400_000
METRIC_ITERATIONS = 1_000
EARLY_STOP_STEPS = 10_000


def corr_multiple(x, y):
    assert x.shape[1] == y.shape[1]
    return [np.corrcoef(x[:, i],  y[:, i], rowvar=False)[0, 1] for i in range(x.shape[1])]


def process_batch(bench_model, generator, is_train, iteration):
    loss_function = nn.MSELoss()

    if is_train:
        bench_model.model.train()
    else:
        bench_model.model.eval()

    batch = next(generator)
    x_batch, y_batch = batch


    y_batch_speech_indexes = bench_model.detect_voice(y_batch)

    assert x_batch.shape[0] == y_batch.shape[0]
    x_batch = torch.FloatTensor(x_batch).to(DEVICE)
    y_batch = torch.FloatTensor(y_batch).to(DEVICE)

    if is_train:
        bench_model.optimizer.zero_grad()

    y_predicted = bench_model.model(x_batch)
    assert not torch.any(torch.isnan(y_predicted))

    loss = loss_function(y_predicted, y_batch)


    if is_train:
        loss.backward()
        bench_model.optimizer.step()

    assert y_predicted.shape[0] == y_batch.shape[0], f"{y_predicted.shape[0]} != {y_batch.shape[0]}"
    assert y_predicted.shape[1] == y_batch.shape[1], f"{y_predicted.shape[1]} != {y_batch.shape[1]}"

    metrics = {}

    y_predicted_numpy = y_predicted.cpu().detach().numpy()
    y_batch_numpy = y_batch.cpu().detach().numpy()

    metrics["loss"] = float(loss.cpu().detach().numpy())

    metrics["correlation"] = float(np.nanmean(corr_multiple(y_predicted_numpy, y_batch_numpy)))

    if np.any(y_batch_speech_indexes):
        metrics["correlation_speech"] = float(np.nanmean(corr_multiple(y_predicted_numpy[y_batch_speech_indexes], y_batch_numpy[y_batch_speech_indexes])))


    for key, value in metrics.items():
        bench_model.logger.add_value(key, is_train, value, iteration)

    return metrics


def get_random_predictions(model, generator, iterations):
    Y_batch = []
    Y_predicted = []
    for index, batch in enumerate(generator):
        x_batch, y_batch = batch[0], batch[1]
        x_batch = torch.FloatTensor(x_batch).to(DEVICE)
        y_predicted = model(x_batch).cpu().detach().numpy()
        assert x_batch.shape[0]==y_predicted.shape[0]
        Y_predicted.append(y_predicted)
        Y_batch.append(y_batch)
        if index > iterations:
            break

    Y_predicted = np.concatenate(Y_predicted, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    return Y_batch, Y_predicted


def run_regression(bench_model_name, patient, runs_count=1, is_debug=False):
    assert hasattr(bench_models_regression, bench_model_name), f"No such model:{bench_model_name}"
    set_seed(SEED)
    print(f"[runtime] device={device_str()} | seed={SEED}")
    bench_model = getattr(bench_models_regression, bench_model_name)(patient)


    X = []
    Y = []

    for filepath in patient["files_list"]:
        with h5py.File(filepath,'r') as input_file:
            data = input_file['RawData']['Samples'][()]

        ecog = data[:, patient["ecog_channels"]].astype("double")
        sound = data[:, patient["sound_channel"]].astype("double")

        x = bench_model.preprocess_ecog(ecog, patient["sampling_rate"]).astype("float32")
        y = bench_model.preprocess_sound(sound, patient["sampling_rate"], x.shape[0]).astype("float32")

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        assert x.shape[0] == y.shape[0]

        X.append(x)
        Y.append(y)


        if is_debug and len(X) >= 2:
            break

    test_start_file_index = bench_model.TEST_START_FILE_INDEX if not is_debug else 1

    # Честный сплит: test = отложенные файлы (только отчёт), val = последний train-файл
    # (для early stopping), train = остальные. Убирает баг val == test.
    train_idx, val_idx, test_idx = make_split(len(X), test_start_file_index)
    print(f"[split] train_files={train_idx} val_files={val_idx} test_files={test_idx}")

    # Whisper-цель: снижение размерности (z-score [-> PCA]) — ФИТ ТОЛЬКО НА TRAIN (без утечки).
    # Фитим на подвыборке кадров train-файлов (без гигантского concat), затем трансформируем ВСЕ Y.
    if getattr(bench_model, "IS_WHISPER_TARGET", False):
        rng = np.random.RandomState(SEED)
        per = max(1, whisper_target.PCA_FIT_MAX_ROWS // max(1, len(train_idx)))
        sample = np.concatenate(
            [Y[i][rng.choice(Y[i].shape[0], min(per, Y[i].shape[0]), replace=False)] for i in train_idx],
            axis=0)
        reducer = whisper_target.fit_reducer(sample, bench_model.USE_PCA, bench_model.PCA_COMPONENTS, seed=SEED)
        Y = [whisper_target.apply_reducer(reducer, y) for y in Y]
        assert Y[0].shape[1] == bench_model.OUTPUT_SIZE, f"reduced {Y[0].shape[1]} != OUTPUT_SIZE {bench_model.OUTPUT_SIZE}"
        print(f"[whisper] target reduced -> {Y[0].shape[1]} dims "
              f"(use_pca={bench_model.USE_PCA}, model={bench_model.WHISPER_MODEL}, layer={bench_model.WHISPER_LAYER})")

    X_train = np.concatenate([X[i] for i in train_idx], axis=0)
    Y_train = np.concatenate([Y[i] for i in train_idx], axis=0)

    X_val = np.concatenate([X[i] for i in val_idx], axis=0)
    Y_val = np.concatenate([Y[i] for i in val_idx], axis=0)

    X_test = np.concatenate([X[i] for i in test_idx], axis=0)
    Y_test = np.concatenate([Y[i] for i in test_idx], axis=0)


    batch_size = bench_model.BATCH_SIZE
    lag_backward = bench_model.LAG_BACKWARD
    lag_forward = bench_model.LAG_FORWARD

    train_generator = data_generator(X_train, Y_train, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True)
    val_generator = data_generator(X_val, Y_val, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True)
    
    max_iterations_count = MAX_ITERATIONS_COUNT if not is_debug else 1_000
    
    
    for run_iteration in range(runs_count):
        print("Run iteration:", run_iteration)
        best_iteration = 0
        max_metric = -float("inf")
        max_metric_speech = -float("inf")
        bench_model = getattr(bench_models_regression, bench_model_name)(patient)


        model_filename = f"regression___{patient['name']}___{bench_model.__class__.__name__}___{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        model_path =  f"model_dumps/{model_filename}.pth"
        for iteration in range(max_iterations_count):
            process_batch(bench_model, train_generator, True, iteration)
            with torch.no_grad():
                metrics = process_batch(bench_model, val_generator, False, iteration)
                is_last_iteration = iteration == (max_iterations_count - 1)
                if iteration % 1000 == 0 or is_last_iteration:
                    smoothed_metric = bench_model.logger.get_smoothed_value("correlation")
                    smoothed_metric_speech = bench_model.logger.get_smoothed_value("correlation_speech")
                    if smoothed_metric >= max_metric or smoothed_metric_speech >= max_metric_speech:
                        max_metric = max(smoothed_metric, max_metric)
                        max_metric_speech = max(smoothed_metric_speech, max_metric_speech)
                        best_iteration = iteration
                        torch.save(bench_model.model.state_dict(), model_path)
                    else:
                        assert iteration >= best_iteration
                        if (iteration - best_iteration) > EARLY_STOP_STEPS:
                            print(f"Stopping model. Iteration {iteration} {round(smoothed_metric, 2)} {round(smoothed_metric_speech, 2)}. Best iteration {best_iteration} {round(max_metric, 2)} {round(max_metric_speech, 2)}.")
                            break
        bench_model.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        bench_model.model.eval()

        test_generator = data_generator(X_test, Y_test, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True)

        result = {}
        result["train_corr"] = np.mean(corr_multiple(*get_random_predictions(bench_model.model, train_generator, METRIC_ITERATIONS)))
        result["val_corr"] = np.mean(corr_multiple(*get_random_predictions(bench_model.model, val_generator, METRIC_ITERATIONS)))
        result["test_corr"] = np.mean(corr_multiple(*get_random_predictions(bench_model.model, test_generator, METRIC_ITERATIONS)))
        result["train_logs"] = bench_model.logger.train_logs
        result["val_logs"] = bench_model.logger.test_logs
        result["iterations"] = iteration
        result["config"] = {
            "mode": "regression",
            "model": bench_model_name,
            "patient": patient["name"],
            "seed": SEED,
            "device": str(DEVICE),
            "split": {"train_files": train_idx, "val_files": val_idx, "test_files": test_idx},
            "best_iteration": best_iteration,
            "augment": "none",
            "multitask": False,
            "word_weight": None,
            "selection_metric": "correlation",
        }

        with open(f'results/{model_filename}.json', 'w') as result_file:
            json.dump(result, result_file)
