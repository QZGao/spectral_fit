import json

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from pulsar_spectra.spectral_fit import robust_cost_function, huber_loss_function, migrad_simplex_scan

from catalogue import Dataset
from env import Env
from models import is_good_fit
from plots import plot


def fit_aic(dataset: Dataset, model_name: str, env: Env, dataset_plot: Dataset = None, output: bool = False):
    model = env.model_dict[model_name]['model']
    labels = env.model_dict[model_name]['labels']
    start_params = env.model_dict[model_name]['start_params'].copy()
    limits = env.model_dict[model_name]['limits'].copy()

    # Add the reference frequency to the parameters, since pulsar_spectra model functions use it as an input parameter
    start_params.append(dataset.v0)
    limits.append(None)

    # Decide based on the current dataset by default
    for i in range(len(limits)):
        if limits[i] == 'dynamic':
            limits[i] = (np.min(dataset.X), np.max(dataset.X))
        elif limits[i] == 'dynamic_expanded':
            limits[i] = (np.min(dataset.X), np.max(dataset.X) * 1.5)
    for i in range(len(start_params)):
        if start_params[i] == 'xmid':
            start_params[i] = np.median(dataset.X)
        elif start_params[i] == 'xmax':
            start_params[i] = np.max(dataset.X)

    # AIC fitting procedure borrowed from pulsar_spectra by Nick Swainston
    # https://github.com/NickSwainston/pulsar_spectra/blob/373f65d866c3bf162fd6d93780235cfbd81849b5/pulsar_spectra/spectral_fit.py#L420-L425
    least_squares = LeastSquares(dataset.X, dataset.Y, dataset.YERR, model)
    least_squares.loss = huber_loss_function
    m = Minuit(least_squares, *tuple(start_params))
    m.fixed["v0"] = True  # Not part of the fit
    m = migrad_simplex_scan(m, limits, model_name)

    # Calculate AIC
    k = len(start_params) - 1
    beta = robust_cost_function(model(dataset.X, *m.values), dataset.Y, dataset.YERR)
    if env.args.aic_no_corr:
        # AIC without correction term
        aic = 2 * beta + 2 * k
    else:
        if len(dataset.X) - k - 1 != 0:
            # AIC with correction term
            aic = 2 * beta + 2 * k + (2 * k * (k + 1)) / (len(dataset.X) - k - 1)
        else:
            # AIC calculation is not possible if the number of data points is no more than the number of parameters plus 1
            # See the following link for pulsar_spectra's implementation
            # https://github.com/NickSwainston/pulsar_spectra/blob/373f65d866c3bf162fd6d93780235cfbd81849b5/pulsar_spectra/spectral_fit.py#L414-L418
            raise ValueError("For AIC calculation, the number of data points must be more than the number of parameters plus 1.")
            # This error will not actually be raised, because the number of data points is checked beforehand in fit.py

    # Calculate the parameter estimates
    median, plus_minus = [], []
    for i in range(len(labels)):
        median.append(m.values[i])
        plus_minus.append(m.errors[i])

    # Goodness of fit
    params_dict = {k: v for k, v in zip(labels, median)}
    fitted_model = lambda v: model(v, *median, dataset.v0)
    good_fit = is_good_fit(dataset, params_dict, fitted_model, env)

    # Save the results
    with open(f'{env.outdir}/{dataset.jname}/{model_name}_results.json', 'w', encoding='utf-8-sig') as f:
        json.dump({
            'aic': aic,
            'param_estimates': {
                'median': median,
                'plus_minus': plus_minus,
            },
            'dataset': dataset.to_dict(),
            'model': model_name,
            'params': labels,
            'start_params': env.model_dict[model_name]['start_params'],
            'limits': env.model_dict[model_name]['limits'],
            'good_fit': str(good_fit),
        }, f, ensure_ascii=False, indent=4)

    # Plots
    plot(
        dataset=dataset if dataset_plot is None else dataset_plot,
        model=model,
        model_name_cap=env.model_dict[model_name]['name'],
        labels=labels,
        param_estimates=(median, plus_minus),
        iminuit_result=m,
        aic=aic,
        good_fit=good_fit,
        outpath=f'{env.outdir}/{dataset.jname}/{model_name}_result',
        env=env,
    )
