import json
import multiprocessing
import traceback
import warnings
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from sys import exit

import numpy as np
from rich.console import Console
from rich.progress import Progress

from aic import fit_aic
from bayesian import fit_bayesian
from catalogue import get_catalogue
from env import Env
from models import get_models

warnings.filterwarnings("ignore")
console = Console()


def fit(jname: str, model_name: str, env: Env):
    try:
        dataset = env.catalogue.get_pulsar(jname, args=env.args)
        if dataset is None:  # Not in the catalogue, or no data points
            return

        if not env.args.no_requirements:
            # Check if the dataset meets the requirements
            if len(np.unique(dataset.X)) < max(4, len(env.model_dict[model_name]['labels'])):
                console.log(f"{jname} {model_name}: Not enough data points to fit.", style='yellow')
                return
            if np.max(dataset.X) / np.min(dataset.X) < 2:
                console.log(f"{jname} {model_name}: Not enough frequency range to fit.", style='yellow')
                return

        if env.args.aic and len(dataset.X) <= len(env.model_dict[model_name]['labels']) + 1:
            # AIC calculation is not possible if the number of data points is no more than the number of parameters plus 1
            # See the following link for pulsar_spectra's implementation
            # https://github.com/NickSwainston/pulsar_spectra/blob/373f65d866c3bf162fd6d93780235cfbd81849b5/pulsar_spectra/spectral_fit.py#L414-L418
            console.log(f'{jname} {model_name}: Not enough data points to calculate AIC.', style='yellow')
            return

        Path(f'{env.outdir}/{jname}').mkdir(parents=True, exist_ok=True)

        if env.args.aic:
            fit_aic(dataset, model_name, env=env)
        else:
            fit_bayesian(dataset, model_name, env=env)
    except Exception as e:
        console.log(f"Error: {jname} {model_name}\n{e}\n{traceback.format_exc()}", style='red')


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Data used for fitting
    parser.add_argument('--jname', help="Specific pulsar name")
    parser.add_argument('--model', help="Specific model name", default='simple_power_law;'
                                                                                    'broken_power_law;'
                                                                                    'log_parabolic_spectrum;'
                                                                                    'high_frequency_cut_off_power_law;'
                                                                                    'low_frequency_turn_over_power_law')

    # Dataset
    parser.add_argument('--lit_set', help="Customize literature list", type=str, default=None)
    parser.add_argument('--atnf', help="Include ATNF pulsar catalogue", action='store_true')
    parser.add_argument('--atnf_ver', help="ATNF pulsar catalogue version", type=str, default='1.54')
    parser.add_argument('--jan_set', help="Use Jankowski et al. (2018)'s (reproduced) dataset", action='store_true')

    # Fitting
    parser.add_argument('--no_requirements', help="Do not require the dataset to have at least 4 points and a frequency range of at least 2", action='store_true')
    parser.add_argument('--fixed_freq_prior', help="Use fixed frequency prior", action='store_true')

    # Dealing with outliers
    # 1) Remove outliers, set minimum YERR / Y, or set all YERR / Y to a value
    parser.add_argument('--outliers_rm', help="Remove outliers", action='store_true')
    parser.add_argument('--outliers_min', help="Set minimum YERR / Y (when --outliers_rm is not set)", type=float)
    parser.add_argument('--outliers_all', help="Don't believe in any of the YERRs and set them all to this value",
                        type=float)

    # 2) Use additional systematic error
    parser.add_argument('--no_err', help="Do not use systematic error", action='store_true')
    parser.add_argument('--err_all', help="Use systematic error for all points", action='store_true')
    parser.add_argument('--err_thresh', help="Threshold for systematic error", type=float, default=0.1)

    # Jan et al. (2018)'s method, a.k.a. AIC method in pulsar_spectra
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
            if not Path(f'{env.outdir}/{jname}/{model_name}_result.png').exists() or not Path(
                    f'{env.outdir}/{jname}/{model_name}_results.json').exists():
                continue
            if not args.aic:
                if not Path(f'{env.outdir}/{jname}/{model_name}_dres.pkl').exists() or not Path(
                        f'{env.outdir}/{jname}/{model_name}_corner.png').exists():
                    continue

            # Check if the result file is readable
            with open(f'{env.outdir}/{jname}/{model_name}_results.json', 'r', encoding='utf-8-sig') as f:
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
        outdir = f'output/{args.outdir}'
    else:
        outdir = f'output/outdir_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        if args.label:
            outdir += f'_{args.label.replace(" ", "_")}'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    console.log(f"outdir: {outdir}")

    env = Env(args, get_models(args.model, aic=args.aic), get_catalogue(args), outdir)
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
