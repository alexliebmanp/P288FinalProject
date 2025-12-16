# %% 

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, '..')
from lstm import AVERT_LSTM

import itertools
import tensorflow as tf
import keras

processed_dpath = '../processed_data/'

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
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

# %% 

# lets now try to train a model with just a subset of data, say a two month period
#
# for context, the example 1 variable LSTM in LSTMI.ipynb had 643,200+3,208 paramters in it, and the length of the training set was 168 time steps. This ran 150 epochs in a reasonable time. 

df = pd.read_csv(processed_dpath+f'dataInterpolated.csv', index_col=0)
df.index = pd.to_datetime(df.index)

eps = 1e-6
df['log_VPCC_RSAM'] = np.log(df['VPCC_RSAM'].clip(lower=eps))
df['log_VPPC_RSAM'] = np.log(df['VPPC_RSAM'].clip(lower=eps))
df['log_VPNC_RSAM'] = np.log(df['VPNC_RSAM'].clip(lower=eps))
df['log_VPRS_RSAM'] = np.log(df['VPRS_RSAM'].clip(lower=eps))

# cut the range and resample to one data point per 12 hours
df = df.resample('12H').mean()
df['Eruption_Activity_Label'] = (df['Eruption_Activity'] > 0).astype(int)

# %% 


def evaluate_run(lstm, target_col=None):
    y_true = lstm.df_y.loc[lstm.test_index]
    y_pred = lstm.df_yhat

    # align indices/cols
    y_true = y_true[y_pred.columns].loc[y_pred.index]

    err = y_true - y_pred

    mae_by_col = err.abs().mean()
    rmse_by_col = np.sqrt((err ** 2).mean())

    out = {
        "mae_all": float(mae_by_col.mean()),
        "rmse_all": float(rmse_by_col.mean()),
    }

    # optionally report one column separately
    if target_col is not None:
        out[f"mae_{target_col}"] = float(mae_by_col[target_col])
        out[f"rmse_{target_col}"] = float(rmse_by_col[target_col])

    return out


def run_one(df, input_vars, predict_vars,
            n_past, n_future, n_epochs,
            n_divide, n_neurons, learning_rate, momentum,
            opt="adam", activation="tanh",
            task_type="regression"):

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
    lstm.Fit()
    lstm.Predict()

    metrics = evaluate_run(lstm)
    return metrics


def grid_search(df, input_vars, predict_vars, grid, base_kwargs):
    results = []
    combos = list(itertools.product(*[grid[k] for k in grid.keys()]))

    for vals in combos:
        params = dict(zip(grid.keys(), vals))

        if params.get("opt", "adam") == "adam" and params.get("momentum", 0.0) != 0.0:
            continue

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

    return pd.DataFrame(results)

# %% 
# %% 

df = pd.read_csv(processed_dpath+f'dataInterpolated.csv', index_col=0)
df.index = pd.to_datetime(df.index)

eps = 1e-6
for s in ["VPCC_RSAM", "VPPC_RSAM", "VPNC_RSAM", "VPRS_RSAM"]:
    df[f"log_{s}"] = np.log(df[s].clip(lower=eps))

agg = {c: "mean" for c in df.columns}
agg["Eruption_Activity"] = "max" # or "sum"
df = df.resample("12H").agg(agg)

# cut the range and resample to one data point per 12 hours
input_vars = ['Eruption_Activity', 'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size']
predict_vars = ['Eruption_Activity', 'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size']

# define model and plot data
lstm = AVERT_LSTM(df, input_vars=input_vars, predict_vars=predict_vars)
# use 36 hours to forecast the next 12 hours
lstm.CreateModel(n_past=3, n_future=1, n_epochs=200, n_divide=0.75, n_neurons=400, learning_rate=1e-4, momentum=0.4, opt="adam", activation='tanh', task_type='regression')
lstm.model.summary()
lstm.Fit()
lstm.Predict()

# %% 

lstm.PlotData([lstm.df_y, lstm.df_yhat], log_vars=[])

# %%

# %% 

df = pd.read_csv(processed_dpath+f'dataInterpolated.csv', index_col=0)
df.index = pd.to_datetime(df.index)

eps = 1e-6
for s in ["VPCC_RSAM", "VPPC_RSAM", "VPNC_RSAM", "VPRS_RSAM"]:
    df[f"log_{s}"] = np.log(df[s].clip(lower=eps))

agg = {c: "mean" for c in df.columns}
agg["Eruption_Activity"] = "max" # or "sum"
df = df.resample("12H").agg(agg)

input_vars = [
    'Eruption_Activity',
    'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM',
    'VPNC_Intensity', 'CO2_ppm', 'lake_size'
]

predict_vars = [
    'Eruption_Activity',
    'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM',
    'VPNC_Intensity', 'CO2_ppm', 'lake_size'
]

base_kwargs = dict(
    n_epochs=300,
    n_divide=0.75,
    activation="tanh",
    task_type="regression",
)

grid = {
    "n_past": [3, 6, 12], # 36h, 72h, 144h at 12H sampling
    "n_future": [1, 2, 4], # 12h, 24h, 48h ahead
    "n_neurons": [64, 128, 256, 512],
    "learning_rate": [1e-4],
    "opt": ["adam", "sgd"],
    "momentum": [0.0, 0.2, 0.4],
}

results = grid_search(df, input_vars, predict_vars, grid, base_kwargs)

results_clean = results.sort_values(["rmse_all"])
print(results_clean.head(10))

with open('regression_experiment_results_multiparameter_rescaled.pkl', 'wb') as f: 
    pickle.dump(results, f)


# %% 

# load dataset
with open("regression_experiment_results_multiparameter_rescaled.pkl", "rb") as f:
    results = pickle.load(f)
results_dict = results.copy()

results_clean = results.sort_values(["rmse_all"])
print(results_clean.head(10))


# %% 

best = results_clean.iloc[0]

df = pd.read_csv(processed_dpath+f'dataInterpolated.csv', index_col=0)
df.index = pd.to_datetime(df.index)

eps = 1e-6
for s in ["VPCC_RSAM", "VPPC_RSAM", "VPNC_RSAM", "VPRS_RSAM"]:
    df[f"log_{s}"] = np.log(df[s].clip(lower=eps))

agg = {c: "mean" for c in df.columns}
agg["Eruption_Activity"] = "max" # or "sum"
df = df.resample("12H").agg(agg)

# cut the range and resample to one data point per 12 hours
input_vars = ['Eruption_Activity', 'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size']
predict_vars = ['Eruption_Activity', 'log_VPCC_RSAM', 'log_VPPC_RSAM', 'log_VPNC_RSAM', 'log_VPRS_RSAM', 'VPNC_Intensity', 'CO2_ppm', 'lake_size']

# define model and plot data
lstm = AVERT_LSTM(df, input_vars=input_vars, predict_vars=predict_vars)
lstm.CreateModel(n_past=best.n_past, n_future=best.n_future, n_epochs=300, n_divide=0.75, n_neurons=128, learning_rate=1e-4, momentum=0.0, opt="adam", activation='tanh', task_type='regression')
lstm.model.summary()
lstm.Fit(plot=True)
lstm.Predict()

# %% 

lstm.PlotData([lstm.df_y, lstm.df_yhat], log_vars=[], figsave='lstm_regression', labels=['Observed', 'Predictions'])
# plt.savefig('lstm.pdf')

# %% 


def plot_distribution_comparisons(
    df_obs,
    df_pred,
    columns=None,
    log_vars=None,
    bins=40,
    figsave="dist_compare",
):
    """
    Overlay observed vs predicted distributions for each variable.

    df_obs, df_pred: DataFrames with same columns; indices can differ (will align).
    columns: subset of columns to plot (default: intersection).
    log_vars: list of columns to plot in log10-space for distribution comparison (optional).
    bins: histogram bins (int or array-like).
    """

    df_obs = df_obs.copy()
    df_pred = df_pred.copy()

    # choose columns + align
    if columns is None:
        columns = [c for c in df_obs.columns if c in df_pred.columns]
    df_obs = df_obs[columns]
    df_pred = df_pred[columns]

    # align on index
    common_idx = df_obs.index.intersection(df_pred.index)
    df_obs = df_obs.loc[common_idx]
    df_pred = df_pred.loc[common_idx]

    if log_vars is None:
        log_vars = []

    n = len(columns)
    ncols = len(columns)
    scale = 3
    fig, ax = plt.subplots(math.ceil(ncols / 2), 2,
                           figsize=(8 * scale, 0.75 * scale * ncols), constrained_layout=True)
    ax = np.array(ax).reshape(-1)  # flat

    for i, c in enumerate(columns):
        a = df_obs[c].to_numpy(dtype=float)
        b = df_pred[c].to_numpy(dtype=float)

        # drop NaNs
        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]

        # log10 transform for distribution comparison
        if c in log_vars:
            eps = 1e-12
            a = np.log10(np.clip(a, eps, None))
            b = np.log10(np.clip(b, eps, None))
            xlabel = f"log10({c})"
        else:
            xlabel = c

        # robust shared bins so the two histograms are directly comparable
        if len(a) == 0 or len(b) == 0:
            ax[i].text(0.5, 0.5, "No data after alignment", ha="center", va="center",
                       transform=ax[i].transAxes)
            ax[i].set_title(c)
            continue

        lo = np.nanmin(np.r_[a, b])
        hi = np.nanmax(np.r_[a, b])
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        edges = np.linspace(lo, hi, bins + 1) if isinstance(bins, int) else bins

        ax[i].hist(a, bins=edges, density=True, alpha=0.3)
        ax[i].hist(b, bins=edges, density=True, alpha=0.3)

        ax[i].hist(a, bins=edges, density=True, histtype="step", linewidth=2, color='tab:blue')
        ax[i].hist(b, bins=edges, density=True, histtype="step", linewidth=2, color='tab:orange')

        # quick summary stats in-panel (optional but useful)
        ax[i].set_title(c)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel("Density")

    for j in range(n, len(ax)):
        fig.delaxes(ax[j])

    # one legend for the figure
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, loc="upper right", frameon=False)

    plt.savefig(figsave + ".pdf")
    plt.show()

# %% 

import math 
cols = [
    "Eruption_Activity",
    "log_VPCC_RSAM", "log_VPPC_RSAM", "log_VPNC_RSAM", "log_VPRS_RSAM",
    "VPNC_Intensity", "CO2_ppm", "lake_size"
]

plot_distribution_comparisons(
    df_obs=lstm.df_y,
    df_pred=lstm.df_yhat,
    columns=cols,
    log_vars=[],          # keep empty if you already stored log_* columns
    bins=50,
    figsave="lstm_dist_compare"
)

# %%
