import json
import pickle
from pathlib import Path

import numpy as np
from dynesty import DynamicNestedSampler
from dynesty.utils import quantile as _quantile
from dynesty.utils import resample_equal
from matplotlib import pyplot as plt
from scipy.special import logsumexp
import scipy.stats as ss

from catalogue import Dataset
from env import Env
from models import is_good_fit
from plots import plot_corner, plot


def log_likelihood(p, model, labels, dataset: Dataset, env: Env):
    Y_model = model(
        dataset.X,
        {labels[i]: p[i] for i in range(len(p))},
        dataset.v0,
    )

    err = dataset.YERR.copy()
    if env.args.efac or env.args.equad:
        ref_set = list(set(dataset.REF))
        for ref in ref_set:
            if env.args.efac:
                err[dataset.REF == ref] *= p[labels.index('e_{\mathrm{fac,\,' + ref.replace('_', '\_') + '}}')]
            if env.args.equad:
                err[dataset.REF == ref] = np.sqrt(
                    err[dataset.REF == ref] ** 2 +
                    p[labels.index('e_{\mathrm{quad,\,' + ref.replace('_', '\_') + '}}')] ** 2 * dataset.Y[dataset.REF == ref] ** 2
                )

    if dataset.len < 4 or env.args.gaussian_patch:  # Patch formula for dataset with only 2 or 3 points
        return np.sum(np.log(np.exp(- ((dataset.Y - Y_model) / err) ** 2)))
    if env.args.gaussian:  # Gaussian likelihood
        return ss.norm.logpdf(dataset.Y, loc=Y_model, scale=err).sum()
    else:  # Cauchy likelihood
        return ss.cauchy.logpdf(dataset.Y, loc=Y_model, scale=err).sum()


def uniform_ppf(q, low, high):
    return low + q * (high - low)


def log_uniform_ppf(q, low, high):
    return 10 ** (uniform_ppf(q, np.log10(low), np.log10(high)))


def prior_transform(u, priors):
    p = np.zeros_like(u)
    for i, q in enumerate(priors):
        if q[0] > q[1]:
            raise ValueError(f"Invalid prior: {q}")
        if q[2] == 'uniform':  # Uniform prior
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


def fit_bayesian(dataset: Dataset, model_name: str, env: Env, dataset_plot: Dataset = None, output: bool = False):
    model = env.model_dict[model_name]['model']
    labels = env.model_dict[model_name]['labels'].copy()
    priors = env.model_dict[model_name]['priors'].copy()
    priors = prior_translate(priors, dataset, env)

    # Add a prior for the systematic error
    if env.args.efac or env.args.equad or env.args.efac_qbound:
        ref_set = list(set(dataset.REF))
        if env.args.efac_qbound:
            by_ref = dataset.by_ref()
            for ref in ref_set:
                min_yerr_y = np.min(by_ref[ref]['YERR'] / by_ref[ref]['Y'])
                efac_deduced = np.sqrt(1. + env.args.efac_qbound**2 / min_yerr_y**2)  # sqrt(original^2 + equad^2) = original * efac
                labels.append('e_{\mathrm{fac,\,' + ref.replace('_', '\_') + '}}')
                priors.append((1., efac_deduced, 'log_uniform'))
        else:
            for ref in ref_set:
                if env.args.efac:
                    labels.append('e_{\mathrm{fac,\,' + ref.replace('_', '\_') + '}}')
                    priors.append((1., env.args.efac, 'uniform'))
                if env.args.equad:
                    labels.append('e_{\mathrm{quad,\,' + ref.replace('_', '\_') + '}}')
                    priors.append((0., env.args.equad, 'uniform'))


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
            print_progress=output,
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

    # Goodness of fit
    params_dict = {k: v for k, v in zip(labels, median)}
    fitted_model = lambda v: model(v, params_dict, dataset.v0)
    good_fit = is_good_fit(dataset, params_dict, fitted_model, env)

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
            'good_fit': str(good_fit),
            # Format changed to numpy.int32 by dynesty, so convert them back for JSON compatibility
        }, f, ensure_ascii=False, indent=4)

    # Plots
    if env.args.corner:
        plot_corner(
            samples=samp_all,
            labels=labels,
            priors=priors,
            env=env,
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
        good_fit=good_fit,
        outpath=f'{env.outdir}/{dataset.jname}/{model_name}_result',
        env=env,
        display_info=dataset_plot is None,
    )
