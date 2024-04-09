import json
import os
import pickle
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image
from dynesty.utils import resample_equal
from rich.console import Console
from rich.progress import Progress

from catalogue import get_catalogue
from models import get_models


def map_list_to_dict(keys: list, values: list):
    return {keys[i]: values[i] for i in range(len(keys))}


def get_plot(outdir: str, jname: str | list[str], model: str | list[str]):
    if not os.path.exists(f'output/{outdir}'):
        raise FileNotFoundError(f"Directory output/{outdir} does not exist.")

    if isinstance(jname, str):
        jname = [jname]
    if isinstance(model, str):
        model = [model]

    for j in jname:
        for m in model:
            if not os.path.exists(f'output/{outdir}/{j}/{j}_{m}_result.png'):
                raise FileNotFoundError(f"File output/{outdir}/{j}/{j}_{m}_result.png does not exist.")

            img = Image.open(f'output/{outdir}/{j}/{j}_{m}_result.png')
            img.show()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument('-v', '--var', help="Variable to process (default: log_evidence)", default='log_evidence')
    parser.add_argument('-p', '--plot', help="Extract pictures", default='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = f'output/{args.outdir}'
    console = Console()
    console.log(f"Extracting {args.var} from {args.outdir}")

    with (Progress() as progress):
        if args.var in ['chi_squared', 'reduced_chi_squared']:
            catalogue = get_catalogue()
            models = {}
        if args.plot:
            folders = args.plot.split(';')
        else:
            folders = [f for f in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, f))]
        task = progress.add_task("Processing", total=len(folders))
        for folder in folders:
            progress.update(task, description=f"Processing {args.outdir}")
            if args.plot:
                files = [f for f in os.listdir(os.path.join(outdir, folder)) if f.endswith('.pdf')]
            elif args.var in ['chi_squared', 'reduced_chi_squared']:
                files = [f for f in os.listdir(os.path.join(outdir, folder)) if f.endswith('.pkl')]
            else:
                files = [f for f in os.listdir(os.path.join(outdir, folder)) if f.endswith('.json')]

            if len(files) == 0:
                console.log(f'Warning: {args.outdir} does not contain any results.', style='yellow')

            if args.plot:
                for file in files:
                    # Copy file to new directory
                    os.makedirs(f'output/plots_from_{args.outdir}', exist_ok=True)
                    shutil.copy(os.path.join(outdir, folder, file), f'output/plots_from_{args.outdir}/{folder}_{file}')

            elif args.var in ['chi_squared', 'reduced_chi_squared']:
                pulsar = catalogue.get_pulsar(folder)
                for file in files:
                    model_name = file.removesuffix("_dres.pkl")
                    if model_name not in models:
                        models[model_name] = get_models(model_name)[model_name]

                    # Open pkl file of sampler.results
                    with open(os.path.join(outdir, folder, file), 'rb') as f:
                        dres = pickle.load(f)

                    # Resample samples based on weights
                    weights = np.exp(dres['logwt'] - dres['logz'][-1])
                    samp_all = resample_equal(dres.samples, weights)

                    # Calculate the chi-squared
                    chi_squared_all = []
                    for samp in samp_all:
                        params = map_list_to_dict(models[model_name]['labels'], samp)
                        Y_model = models[model_name]['model'](pulsar.X, params, pulsar.v0)
                        chi_squared = np.sum(np.power((pulsar.Y - Y_model) / pulsar.YERR, 2))
                        chi_squared_all.append(chi_squared)
                    chi_squared_all = np.array(chi_squared_all)

                    # Calculate reduced chi-squared
                    if args.var == 'reduced_chi_squared':
                        dof = pulsar.dof(n_params=len(models[model_name]['labels']))
                        if dof == 0:
                            chi_squared_all = None
                        else:
                            chi_squared_all /= dof

                    Path(f'output/{args.var}_from_{args.outdir}').mkdir(parents=True, exist_ok=True)
                    with open(f'output/{args.var}_from_{args.outdir}/{folder}_{model_name}_{args.var}.pkl', 'wb') as f:
                        pickle.dump(chi_squared_all, f)

            else:
                result_dict = {}
                for file in files:
                    model_name = file.removesuffix("_results.json")
                    # Open file and extract variable
                    with open(os.path.join(outdir, folder, file), 'r', encoding='utf-8-sig') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            console.log(f'Error: {folder}/{file} is not a valid json file.', style='red')
                            continue

                        if folder not in result_dict:
                            result_dict[folder] = {}
                        result_dict[folder][model_name] = data[args.var]

                with open(f'output/results_{args.var}_from_{args.outdir}.json', 'w') as f:
                    json.dump(result_dict, f, indent=4)
                console.log(f"Results saved to results_{args.var}_from_{args.outdir}.json")

            progress.advance(task)
