import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, '..')
from lstm import AVERT_LSTM
import itertools
import tensorflow as tf
import keras
import pickle

grid = {
    "n_past": [3, 6, 12], # 36h, 72h, 144h at 12H sampling
    "n_future": [1, 2, 4], # 12h, 24h, 48h ahead
    "n_neurons": [400, 600, 800],
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "opt": ["adam"],
    "momentum": [0.2, 0.4],
    "activation": ["tanh"]
}
fname = 'lstm_classification_results_1'

grid = {
    "n_past": [3, 6, 12], # 36h, 72h, 144h at 12H sampling
    "n_future": [1, 2, 4], # 12h, 24h, 48h ahead
    "n_neurons": [400, 600, 800],
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "opt": ["adam", "sgd"],
    "momentum": [0.2, 0.4],
    "activation": ["tanh", "relu"]
}
fname = 'lstm_classification_results_2'

processed_dpath = 'processed_data/'

plt.rcParams.update(
    {
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "pdf.fonttype": 42,
        "font.family": "Avenir",
        "font.size": 18,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.pad": 6.5,
        "ytick.major.pad": 6.5,
    }
)


# load data
df = pd.read_csv(processed_dpath+f'dataInterpolated.csv', index_col=0)
df.index = pd.to_datetime(df.index)

eps = 1e-6
for s in ["VPCC_RSAM", "VPPC_RSAM", "VPNC_RSAM", "VPRS_RSAM"]:
    df[f"log_{s}"] = np.log(df[s].clip(lower=eps))

df = df.resample('12h').mean()
df['Eruption_Activity_Label'] = (df['Eruption_Activity'] > 0).astype(int)
input_vars = ['Eruption_Activity_Label', 'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size']
predict_vars = ['Eruption_Activity_Label']


def evaluate_run(lstm, eruption_col="Eruption_Activity_Label", eruption_thresh=0.0):
    # true values in timestamps as predictions
    y_true = lstm.df_y.loc[lstm.test_index]
    y_pred = lstm.df_yhat

    # align indices/cols
    y_true = y_true[y_pred.columns].loc[y_pred.index]

    # regression metrics
    mae_all = (y_true - y_pred).abs().mean().mean() # mean over time then over features ? 
    mae_eruption = (y_true[eruption_col] - y_pred[eruption_col]).abs().mean()

    # check how well it separates activity vs no-activity (threshold on truth)
    y_true_evt = (y_true[eruption_col] > eruption_thresh).astype(int).to_numpy()
    y_score = y_pred[eruption_col].to_numpy()

    # auc / roc 
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true_evt, y_score) if len(np.unique(y_true_evt)) > 1 else np.nan
    except Exception:
        auc = np.nan

    return {
        "mae_all": float(mae_all),
        "mae_eruption": float(mae_eruption),
        "auc_eruption": float(auc),
        "eruption_rate_test": float(y_true_evt.mean()),
    }


def run_one(df, input_vars, predict_vars,
            n_past, n_future, n_epochs,
            n_divide, n_neurons, learning_rate, momentum,
            opt="adam", activation="tanh",
            task_type="binary",
            seed=42):

    # Reproducibility-ish
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Fresh instance each run
    lstm = AVERT_LSTM(df, input_vars=input_vars, predict_vars=predict_vars)

    lstm.CreateModel(
        n_past=n_past,
        n_future=n_future,
        n_epochs=n_epochs,
        n_divide=n_divide,
        n_neurons=n_neurons,
        learning_rate=learning_rate,
        momentum=momentum,
        opt=opt,
        activation=activation,
        task_type=task_type,
    )
    lstm.Fit(plot=False)
    lstm.Predict()

    metrics = evaluate_run(lstm)
    return metrics


def grid_search(df, input_vars, predict_vars, grid, base_kwargs):
    results = []
    combos = list(itertools.product(*[grid[k] for k in grid.keys()]))
    ncombos = len(combos)

    print(f"number of combos: {len(combos)}")
    for ii, vals in enumerate(combos):
        print(ii/ncombos*100)
        params = dict(zip(grid.keys(), vals))

        if params.get("opt", "adam") == "adam" and "momentum" in params and params["momentum"] not in (0.0, 0.9):
            # no momentum for adam 
            pass

        # clear state between runs
        keras.backend.clear_session()

        try:
            metrics = run_one(
                df, input_vars, predict_vars,
                **params,
                **base_kwargs,
            )
            row = {**params, **metrics}
            results.append(row)
            print(row)
        except Exception as e:
            row = {**params, "error": str(e)}
            results.append(row)
            print("failed: ", row)

        # save each time
        with open(f'{fname}.pkl' , 'wb') as f:
            pickle.dump(pd.DataFrame(results), f)

    return pd.DataFrame(results)


base_kwargs = dict(
    n_epochs=300,
    n_divide=0.75,
    task_type="binary",
    seed=42,
)

results = grid_search(df, input_vars, predict_vars, grid, base_kwargs)

# Rank by eruption MAE (lower is better), then overall MAE
# results_clean = results[~results.columns.isin(["error"])].copy()
results_clean = results.sort_values(["mae_eruption", "mae_all"])
display(results_clean.head(10))

