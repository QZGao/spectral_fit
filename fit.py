import json
from pebble import ProcessPool
from multiprocessing import cpu_count
from concurrent.futures import TimeoutError
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
from catalogue import get_catalogue_from_args, Dataset
from env import Env
from models import get_models

warnings.filterwarnings("ignore")
console = Console()


def fit(jname: str, model_name: str, env: Env, dataset: Dataset = None, dataset_plot: Dataset = None, output: bool = False):
    try:
        if dataset is None:
            dataset = env.catalogue.get_pulsar(jname, args=env.args)
        if dataset is None:  # Not in the catalogue, or no data points
            return

        if env.args.no_requirements:
            # Check if there are at least 2 points
            if len(np.unique(dataset.X)) < 2:
                console.log(f"{jname} {model_name}: Not enough data points to fit.", style='yellow')
                return
        else:
            # Check if the dataset meets the requirements
            if len(np.unique(dataset.X)) < max(4, len(env.model_dict[model_name]['labels'])):
                console.log(f"{jname} {model_name}: Not enough data points to fit.", style='yellow')
                return
            if np.max(dataset.X) / np.min(dataset.X) < 2:
                console.log(f"{jname} {model_name}: Not enough frequency range to fit.", style='yellow')
                return

        if env.args.aic and (not env.args.aic_no_corr) and len(dataset.X) <= len(env.model_dict[model_name]['labels']) + 1:
            # AIC calculation is not possible if the number of data points is no more than the number of parameters plus 1
            # See the following link for pulsar_spectra's implementation
            # https://github.com/NickSwainston/pulsar_spectra/blob/373f65d866c3bf162fd6d93780235cfbd81849b5/pulsar_spectra/spectral_fit.py#L414-L418
            console.log(f'{jname} {model_name}: Not enough data points to calculate AIC.', style='yellow')
            return

        Path(f'{env.outdir}/{jname}').mkdir(parents=True, exist_ok=True)

        if env.args.aic:
            fit_aic(dataset, model_name, env=env, dataset_plot=dataset_plot, output=output)
        else:
            fit_bayesian(dataset, model_name, env=env, dataset_plot=dataset_plot, output=output)
    except Exception as e:
        # Most likely due to very wrong data points, can't fit at all
        console.log(f"Error: {jname} {model_name}\n{e}\n{traceback.format_exc()}", style='red')

        # Clean up
        # if Path(f'{env.outdir}/{jname}/{model_name}_dres.pkl').exists():
        #     Path(f'{env.outdir}/{jname}/{model_name}_dres.pkl').unlink()
        # if env.args.pdf:
        #     if Path(f'{env.outdir}/{jname}/{model_name}_result.pdf').exists():
        #         Path(f'{env.outdir}/{jname}/{model_name}_result.pdf').unlink()
        #     if Path(f'{env.outdir}/{jname}/{model_name}_corner.png').exists():
        #         Path(f'{env.outdir}/{jname}/{model_name}_corner.png').unlink()
        # else:
        #     if Path(f'{env.outdir}/{jname}/{model_name}_result.png').exists():
        #         Path(f'{env.outdir}/{jname}/{model_name}_result.png').unlink()
        #     if Path(f'{env.outdir}/{jname}/{model_name}_corner.png').exists():
        #         Path(f'{env.outdir}/{jname}/{model_name}_corner.png').unlink()
        # if Path(f'{env.outdir}/{jname}/{model_name}_results.json').exists():
        #     Path(f'{env.outdir}/{jname}/{model_name}_results.json').unlink()


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Data used for fitting
    parser.add_argument('--jname', help="Specific pulsar name")
    parser.add_argument('--model', help="Specific model name", default='simple_power_law;'
                                                                       'broken_power_law;'
                                                                       'log_parabolic_spectrum;'
                                                                       'high_frequency_cut_off_power_law;'
                                                                       'low_frequency_turn_over_power_law;'
                                                                       'double_turn_over_spectrum')

    # Dataset
    parser.add_argument('--lit_set', help="Customize literature list", type=str, default=None)
    parser.add_argument('--atnf', help="Include ATNF pulsar catalogue", action='store_true')
    parser.add_argument('--atnf_ver', help="ATNF pulsar catalogue version", type=str, default='1.54')
    parser.add_argument('--jan_set', help="Use Jankowski et al. (2018)'s (reproduced) dataset", action='store_true')
    parser.add_argument('--print_lit', help="Print literature list", action='store_true')
    parser.add_argument('--refresh', help="Refresh the catalogue", action='store_true')

    # Fitting
    parser.add_argument('--no_requirements', help="Do not require the dataset to have at least 4 points and a frequency range of at least 2", action='store_true')
    parser.add_argument('--fixed_freq_prior', help="Use fixed frequency prior", action='store_true')

    # Dealing with outliers (not applicable to AIC method)
    # 1) Use Gaussian distribution
    parser.add_argument('--gaussian', help="Use Gaussian distribution", action='store_true')
    parser.add_argument('--gaussian_patch', help="Use simplified Gaussian distribution", action='store_true')

    # 2) Remove outliers, set minimum YERR / Y, or set all YERR / Y to a value
    parser.add_argument('--outliers_rm', help="Remove outliers", action='store_true')
    parser.add_argument('--outliers_min', help="Set minimum YERR / Y (when --outliers_rm is not set)", type=float)
    parser.add_argument('--outliers_min_plus', help="Add a systematic error to the uncertainty instead of replacing it", type=float)
    parser.add_argument('--outliers_all', help="Don't believe in any of the YERRs and set them all to this value",
                        type=float)

    # 3) Use additional systematic error
    parser.add_argument('--equad', help="Add an additional systematic error", type=float)
    parser.add_argument('--efac', help="Multiply by a systematic error factor", type=float)
    parser.add_argument('--efac_qbound', help="Multiply by a systematic error factor that is determined by equad bound", type=float)

    # Jan et al. (2018)'s method, a.k.a. AIC method in pulsar_spectra
    parser.add_argument('--aic', help="Use Jankowski et al. (2018)'s method instead", action='store_true')
    parser.add_argument('--aic_no_corr', help="Do not apply the correction term in AIC calculation", action='store_true')

    # Output & multiprocessing
    parser.add_argument('--label', help="Output directory label (when --outdir is not set)")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument('--override', help="Override finished jobs", action='store_true')
    parser.add_argument("--no_checkpoint", help="Save pickle dump file", action='store_true')
    parser.add_argument("--no_plot", help="Plot results", action='store_true')
    parser.add_argument('--nproc', help="Number of processes", type=int, default=cpu_count() - 1 or 1)
    parser.add_argument('--corner', help="Plot corner plot", action='store_true')
    parser.add_argument('--pdf', help="Save the plot as a PDF file", action='store_true')

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
        args.outdir = f'output/{args.outdir}'
    else:
        args.outdir = f'output/outdir_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        if args.label:
            args.outdir += f'_{args.label.replace(" ", "_")}'
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    console.log(f"outdir: {args.outdir}")
    with open(f'{args.outdir}/config.json', 'w', encoding='utf-8-sig') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)  # Save arguments

    env = Env(args, get_models(args.model, aic=args.aic), get_catalogue_from_args(args))
    if args.print_lit:
        env.catalogue.print_lit(args.outdir)
        exit()

    job_list = get_jobs(env)
    if len(job_list) == 0:
        console.log("All jobs finished.")
        exit()
    console.log(f"Number of jobs: {len(job_list)}")

    if len(job_list) == 1:
        fit(*job_list[0], output=True)

    else:
        with Progress() as progress:
            task = progress.add_task("[cyan]Fitting", total=len(job_list))
            job_jnames = [job[0] for job in job_list]
            job_models = [job[1] for job in job_list]
            job_envs = [job[2] for job in job_list]

            with ProcessPool(max_workers=args.nproc) as pool:
                future = pool.map(fit, job_jnames, job_models, job_envs, timeout=600)
                iterator = future.result()
                while True:
                    try:
                        next(iterator)
                    except StopIteration:
                        break
                    except TimeoutError as e:
                        console.log(f"TimeoutError: {e}")
                    finally:
                        progress.update(task, advance=1)
    console.log("All jobs finished.")
