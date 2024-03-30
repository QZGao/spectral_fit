import json
import os
import shutil
from argparse import ArgumentParser, Namespace

from rich.console import Console
from rich.progress import Progress


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument('-v', '--var', help="Variable to process (default: log_evidence)", default='log_evidence')
    parser.add_argument('-p', '--plot', help="Extract pictures", default='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = os.path.join('output', args.outdir)
    console = Console()
    console.log(f"Extracting {args.var} from {outdir}")

    result_dict = {}
    with (Progress() as progress):
        if args.plot:
            folders = args.plot.split(';')
        else:
            folders = [f for f in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, f))]
        task = progress.add_task("Processing", total=len(folders))
        for folder in folders:
            progress.update(task, description=f"Processing {folder}")
            if args.plot:
                files = [f for f in os.listdir(os.path.join(outdir, folder)) if f.endswith('.pdf')]
            else:
                files = [f for f in os.listdir(os.path.join(outdir, folder)) if f.endswith('.json')]

            if len(files) == 0:
                console.log(f'Warning: {folder} does not contain any results.', style='yellow')

            if args.plot:
                for file in files:
                    # Copy file to new directory
                    os.makedirs(f'plots_from_{outdir}', exist_ok=True)
                    shutil.copy(os.path.join(outdir, folder, file), f'plots_from_{outdir}/{folder}_{file}')
            else:
                for file in files:
                    # Open file and extract variable
                    with open(os.path.join(outdir, folder, file), 'r', encoding='utf-8-sig') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            console.log(f'Error: {folder}/{file} is not a valid json file.', style='red')
                            continue

                        if folder not in result_dict:
                            result_dict[folder] = {}
                        result_dict[folder][file.removeprefix(folder + "_").removesuffix("_results.json")] = data[
                            args.var]

            progress.advance(task)

    # Save results to file
    if not args.plot:
        with open(f'results_{args.var}_from_{outdir}.json', 'w') as f:
            json.dump(result_dict, f, indent=4)
        console.log(f"Results saved to results_{args.var}_from_{outdir}.json")
