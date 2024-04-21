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

    if dataset.len <= 4 and dataset.min_yerr_y > 0.1:  # Patch formula for very vague dataset
        return np.sum(np.log(np.exp(- ((dataset.Y - Y_model) / err) ** 2)))
    if env.args.gaussian:  # Gaussian likelihood
        return ss.norm.logpdf(dataset.Y, loc=Y_model, scale=err).sum()
    else:  # Student's-t likelihood
        return ss.t.logpdf(dataset.Y, loc=Y_model, scale=err, df=dataset.len - 1).sum()


def prior_transform(u, priors):
    p = np.zeros_like(u)
    for i, q in enumerate(priors):
        p[i] = q[0] + u[i] * (q[1] - q[0])
    return p


def fit_bayesian(dataset: Dataset, model_name: str, env: Env):
    model = env.model_dict[model_name]['model']
    labels = env.model_dict[model_name]['labels'].copy()
    priors = env.model_dict[model_name]['priors'].copy()
    for i in range(len(priors)):
        if env.args.fixed_freq_prior:
            # Decide based on the whole picture
            if priors[i] == 'dynamic' or priors[i] == 'dynamic_expanded':
                priors[i] = (10., 515250.)
        else:
            # Decide based on the current dataset
            if priors[i] == 'dynamic':
                priors[i] = (np.min(dataset.X), np.max(dataset.X))
            elif priors[i] == 'dynamic_expanded':
                priors[i] = (np.min(dataset.X), np.max(dataset.X) * 1.5)

    # Add a prior for the systematic error
    if not env.args.no_err:
        priors.append((0., dataset.max_yerr_y))  # up to 50% of the flux density
        labels.append('σ')

    if not env.args.override and Path(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl').exists():
        with open(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl', 'rb') as f:
            dres = pickle.load(f)

    else:
        sampler = DynamicNestedSampler(
            loglikelihood=lambda *args: log_likelihood(*args, model=model, labels=labels, dataset=dataset, env=env),
            prior_transform=lambda *args: prior_transform(*args, priors=priors),
            ndim=len(priors),
            sample='rwalk',
            bound='multi',
            nlive=2000,
        )
        sampler.run_nested(
            print_progress=False,
            maxiter=20000,
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
            'priors': np.array(priors).tolist(),  # Format changed to numpy.int32 by dynesty, so convert them back for JSON compatibility
        }, f, ensure_ascii=False, indent=4)

    # Plots
    plot_corner(
        samples=samp_all,
        labels=labels,
        scales=['log' if p == 'dynamic' else 'linear' for p in priors],
        outpath=f'{env.outdir}/{dataset.jname}/{model_name}_corner',
    )
    plot(
        dataset=dataset,
        model=model,
        model_name_cap=env.model_dict[model_name]['name'],
        labels=labels,
        samples=samp_all,
        param_estimates=(median, plus, minus),
        log_evidence=log_evidence,
        log_evidence_err=log_evidence_err,
        outpath=f'{env.outdir}/{dataset.jname}/{model_name}_result',
        env=env,
    )
