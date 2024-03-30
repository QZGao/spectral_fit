import traceback
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from sys import exit
import multiprocessing
import json
import pickle
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from scipy.special import logsumexp
from dynesty import DynamicNestedSampler
from dynesty.utils import quantile as _quantile
from dynesty.utils import resample_equal
import warnings

from plots import plot_corner, plot
from models import get_models
from catalogue import get_catalogue

warnings.filterwarnings("ignore")
console = Console()


class Env:
    def __init__(self, args, model_dict, catalogue, outdir):
        self.args = args
        self.model_dict = model_dict
        self.catalogue = catalogue
        self.outdir = outdir


def log_likelihood(p, model, labels, dataset, env: Env):
    Y_model = model(
        dataset.X,
        {labels[i]: p[i] for i in range(len(p))},
        dataset.v0,
    )

    if env.args.no_err:  # Without systematic error
        combined_err = dataset.YERR
    elif env.args.err_all:  # All with the same systematic error
        combined_err = dataset.combined_err(p[-1])
    else:  # With systematic error for points with yerr/y < threshold
        combined_err = dataset.combined_err(p[-1], threshold=env.args.err_threshold)

    return np.sum(np.log(
        np.exp(- (dataset.Y - Y_model) ** 2 / (combined_err * combined_err)) * (2 * np.pi) ** -0.5 / combined_err))


def prior_transform(u, priors):
    p = np.zeros_like(u)
    for i, q in enumerate(priors):
        p[i] = q[0] + u[i] * (q[1] - q[0])
    return p


def fit_dataset(dataset, model_name, env: Env):
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
                priors[i] = (dataset.X.min(), dataset.X.max())
            elif priors[i] == 'dynamic_expanded':
                priors[i] = (dataset.X.min(), dataset.X.max() * 1.5)

    # Add a prior for the systematic error
    if not env.args.no_err:
        priors.append((0., dataset.max_yerr_y()))  # up to 50% of the flux density
        labels.append('σ')

    if Path(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl').exists():
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
        )

        dres = sampler.results
        with open(f'{env.outdir}/{dataset.jname}/{model_name}_dres.pkl', 'wb') as f:
            pickle.dump(dres, f)

    weights = np.exp(dres['logwt'] - dres['logz'][-1])
    samp_all = resample_equal(dres.samples, weights)

    # Compute the log evidence and its error
    log_evidences = np.array(dres['logz'])
    log_evidence = logsumexp(log_evidences, b=1. / len(log_evidences))
    log_errs = np.array(dres['logzerr'])
    log_evidence_err = 0.5 * logsumexp(log_errs * 2, b=1. / len(log_errs))

    # Compute the parameter estimates
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
            'priors': priors,
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


def fit(jname, model_name, env: Env):
    try:
        dataset = env.catalogue.get_pulsar(jname, nparams=len(env.model_dict[model_name]['labels']), args=env.args)
        if dataset is None:  # Not in the catalogue, or not enough data to fit
            return

        Path(f'{env.outdir}/{jname}').mkdir(parents=True, exist_ok=True)

        fit_dataset(dataset, model_name, env=env)
    except Exception as e:
        console.log(f"Error: {jname} {model_name}\n{e}\n{traceback.format_exc()}", style='red')


def parse_args():
    parser = ArgumentParser()

    # Data used for fitting
    parser.add_argument('--jname', help="Specific pulsar name")
    parser.add_argument('--model', help="Specific model name", default='simple_power_law;'
                                                                       'broken_power_law;'
                                                                       'log_parabolic_spectrum;'
                                                                       'high_frequency_cut_off_power_law;'
                                                                       'low_frequency_turn_over_power_law')
    parser.add_argument('--fixed_freq_prior', help="Use fixed frequency prior", action='store_true')

    # Dataset
    parser.add_argument('--lit_set', help="Customize literature list", type=str, default=None)
    parser.add_argument('--atnf', help="Include ATNF pulsar catalogue", action='store_true')
    parser.add_argument('--jan_set', help="Use Jankowski et al. (2018)'s (reproduced) dataset", action='store_true')

    # Dealing with outliers
    # 1) Remove outliers, set minimum YERR / Y, or set all YERR / Y to a value
    parser.add_argument('--outliers_rm', help="Remove outliers", action='store_true')
    parser.add_argument('--outliers_set_min', help="Set minimum YERR / Y (when --outliers_rm is not set)", type=float)
    parser.add_argument('--outliers_set_all', help="Don't believe in any of the YERRs and set them all to this value",
                        type=float)

    # 2) Use additional systematic error
    parser.add_argument('--no_err', help="Do not use systematic error", action='store_true')
    parser.add_argument('--err_all', help="Use systematic error for all points", action='store_true')
    parser.add_argument('--err_threshold', help="Threshold for systematic error", type=float, default=0.1)

    # Jan et al. (2018)'s method
    parser.add_argument('--aic', help="Use Jankowski et al. (2018)'s method instead", action='store_true')

    # Output & multiprocessing
    parser.add_argument('--label', help="Output directory label (when --outdir is not set)")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument('--override', help="Override finished jobs", action='store_true')
    parser.add_argument('--nproc', help="Number of processes", type=int, default=multiprocessing.cpu_count() - 1 or 1)

    return parser.parse_args()


def get_jobs(env: Env) -> list:
    args = env.args
    job_list = []
    if args.jname:
        if ';' in args.jname:
            jname_list = args.jname.split(';')
            for jname in jname_list:
                for model_name in env.model_dict:
                    job_list.append((jname, model_name, env))
        else:
            if args.jname not in env.catalogue.cat_dict:
                console.log(f"Error: {args.jname} not found in the catalogue.", style='red')
                exit()
            for model_name in env.model_dict:
                job_list.append((args.jname, model_name, env))
    else:
        for jname in env.catalogue.cat_dict:
            for model_name in env.model_dict:
                job_list.append((jname, model_name, env))

    # Remove finished jobs
    if args.outdir and not args.override:
        for jname, model_name, env in job_list[:]:
            # Check if the job is finished
            if not Path(f'{outdir}/{jname}/{model_name}_result.png').exists() or not Path(
                    f'{outdir}/{jname}/{model_name}_results.json').exists():
                continue
            if not args.aic:
                if not Path(f'{outdir}/{jname}/{model_name}_dres.pkl').exists() or not Path(
                        f'{outdir}/{jname}/{model_name}_corner.png').exists():
                    continue

            # Check if the result file is readable
            with open(f'{outdir}/{jname}/{model_name}_results.json', 'r', encoding='utf-8-sig') as f:
                try:
                    data = json.load(f)
                    if ('aic' if args.aic else 'log_evidence') in data:
                        job_list.remove((jname, model_name, env))
                except json.JSONDecodeError:
                    pass

    return job_list


if __name__ == '__main__':
    args = parse_args()

    if args.outdir:
        outdir = args.outdir
    else:
        outdir = f'output/outdir_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        if args.label:
            outdir += f'_{args.label.replace(" ", "_")}'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    console.log(f"outdir: {outdir}")

    env = Env(args, get_models(args.model), get_catalogue(args), outdir)
    job_list = get_jobs(env)
    if len(job_list) == 0:
        console.log("All jobs finished.")
        exit()
    console.log(f"Number of jobs: {len(job_list)}")
    with Progress() as progress:
        task = progress.add_task("[cyan]Fitting", total=len(job_list))

        with multiprocessing.Pool(processes=args.nproc) as pool:
            for jname, model_name, env in job_list:
                pool.apply_async(
                    fit,
                    args=(jname, model_name, env),
                    callback=lambda _: progress.update(task, advance=1),
                )
            pool.close()
            pool.join()
    console.log("All jobs finished.")
