import json
import pickle
from pathlib import Path

import numpy as np
from dynesty import DynamicNestedSampler
from dynesty.utils import quantile as _quantile
from dynesty.utils import resample_equal
from scipy.special import logsumexp
import scipy.stats as ss

from catalogue import Dataset
from env import Env
from plots import plot_corner, plot


def log_likelihood(p, model, labels, dataset: Dataset, env: Env):
    Y_model = model(
        dataset.X,
        {labels[i]: p[i] for i in range(len(p))},
        dataset.v0,
    )

    err = dataset.YERR
    if env.args.err_all:  # All with the same systematic error
        err = dataset.combined_err(p[-1])
    elif env.args.err_thresh:  # With systematic error for points with yerr/y < threshold
        err = dataset.combined_err(p[-1], thresh=env.args.err_thresh)

    if dataset.len < 4 or env.args.gaussian_patch:  # Patch formula for dataset with only 2 or 3 points
        return np.sum(np.log(np.exp(- ((dataset.Y - Y_model) / err) ** 2)))
    if env.args.gaussian:  # Gaussian likelihood
        return ss.norm.logpdf(dataset.Y, loc=Y_model, scale=err).sum()
    else:  # Student's-t likelihood
        return ss.t.logpdf(dataset.Y, loc=Y_model, scale=err, df=dataset.len - 1).sum()


def uniform_ppf(q, low, high):
    return low + q * (high - low)


def log_uniform_ppf(q, low, high):
    return 10 ** (uniform_ppf(q, np.log10(low), np.log10(high)))


def prior_transform(u, priors):
    p = np.zeros_like(u)
    for i, q in enumerate(priors):
        if q[0] > q[1]:
            raise ValueError(f"Invalid prior: {q}")
        if q[2] == 'uniform':        # Uniform prior
            p[i] = uniform_ppf(u[i], q[0], q[1])
        elif q[2] == 'log_uniform':  # Log-uniform prior
            if q[0] <= 0:
                raise ValueError(f"Invalid lower bound for log_uniform prior: {q[0]}")
            p[i] = log_uniform_ppf(u[i], q[0], q[1])
        else:
            raise ValueError(f"Unknown prior type: {q[2]}")
    return p


def prior_translate(priors, dataset: Dataset, env: Env) -> list:
    for i in range(len(priors)):
        if env.args.fixed_freq_prior:
            # Decide based on the whole picture
            if priors[i] == 'dynamic' or priors[i] == 'dynamic_expanded':
                priors[i] = (10., 515250., 'log_uniform')
        else:
            # Decide based on the current dataset
            if priors[i] == 'dynamic':
                priors[i] = (np.min(dataset.X), np.max(dataset.X), 'log_uniform')
            elif priors[i] == 'dynamic_expanded':
                priors[i] = (np.min(dataset.X), np.max(dataset.X) * 1.5, 'log_uniform')
    return priors


def fit_bayesian(dataset: Dataset, model_name: str, env: Env, dataset_plot: Dataset = None):
    model = env.model_dict[model_name]['model']
    labels = env.model_dict[model_name]['labels'].copy()
    priors = env.model_dict[model_name]['priors'].copy()
    priors = prior_translate(priors, dataset, env)

    # Add a prior for the systematic error
    if env.args.err_all or env.args.err_thresh:
        priors.append((0., dataset.max_yerr_y, 'uniform'))  # up to 50% of the flux density
        labels.append('σ')

    dres = None
    if not env.args.override and Path(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl').exists():
        try:
            with open(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl', 'rb') as f:
                dres = pickle.load(f)
        except:
            print(f'Failed to load {env.outdir}/{dataset.jname}/{model_name}_dres.pkl')
            dres = None

    if dres is None:
        sampler = DynamicNestedSampler(
            loglikelihood=lambda *args: log_likelihood(*args, model=model, labels=labels, dataset=dataset, env=env),
            prior_transform=lambda *args: prior_transform(*args, priors=priors),
            ndim=len(priors),
            nlive=2000,
            bound='multi',
            sample='rwalk',
        )
        sampler.run_nested(
            print_progress=False,
        )

        dres = sampler.results
        with open(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl', 'wb') as f:
            pickle.dump(dres, f)

    weights = np.exp(dres['logwt'] - dres['logz'][-1])
    samp_all = resample_equal(dres.samples, weights)

    # Calculate the log evidence and its error
    log_evidences = np.array(dres['logz'])
    log_evidence = logsumexp(log_evidences, b=1. / len(log_evidences))
    log_errs = np.array(dres['logzerr'])
    log_evidence_err = 0.5 * logsumexp(log_errs * 2, b=1. / len(log_errs))

    # Calculate the parameter estimates
    posteriors = {labels[i]: samp_all[:, i] for i in range(len(labels))}
    median = np.array([_quantile(posteriors[label], [0.5])[0] for label in labels]).tolist()
    lower = [_quantile(posteriors[label], [0.5 - 0.3413])[0] for label in labels]
    upper = [_quantile(posteriors[label], [0.5 + 0.3413])[0] for label in labels]
    plus = (np.array(upper) - np.array(median)).tolist()
    minus = (np.array(median) - np.array(lower)).tolist()

    # Save the results
    with open(f'{env.outdir}/{dataset.jname}/{model_name}_results.json', 'w', encoding='utf-8-sig') as f:
        json.dump({
            'log_evidence': float(log_evidence),
            'log_evidence_err': float(log_evidence_err),
            'param_estimates': {
                'median': median,
                'plus': plus,
                'minus': minus,
            },
            'dataset': dataset.to_dict(),
            'model': model_name,
            'params': labels,
            'priors': np.array(priors).tolist(),
            # Format changed to numpy.int32 by dynesty, so convert them back for JSON compatibility
        }, f, ensure_ascii=False, indent=4)

    # Plots
    plot_corner(
        samples=samp_all,
        labels=labels,
        priors=priors,
        outpath=f'{env.outdir}/{dataset.jname}/{model_name}_corner',
    )
    plot(
        dataset=dataset if dataset_plot is None else dataset_plot,
        model=model,
        model_name_cap=env.model_dict[model_name]['name'],
        labels=labels,
        samples=samp_all,
        param_estimates=(median, plus, minus),
        log_evidence=log_evidence,
        log_evidence_err=log_evidence_err,
        outpath=f'{env.outdir}/{dataset.jname}/{model_name}_result',
        env=env,
        display_info=dataset_plot is None,
    )
