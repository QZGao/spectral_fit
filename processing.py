from rich.progress import Progress
from rich.console import Console
console = Console()
import os
import json
from argparse import ArgumentParser
import shutil


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument('-v', '--var', help="Variable to process (default: log_evidence)", default='log_evidence')
    parser.add_argument('-p', '--plot', help="Extract pictures", default='')
    args = parser.parse_args()

    console.log(f"Extracting {args.var} from {args.outdir}")

    result_dict = {}
    with (Progress() as progress):
        if args.plot:
            folders = args.plot.split(';')
        else:
            folders = [f for f in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir, f))]
        task = progress.add_task("Processing", total=len(folders))
        for folder in folders:
            progress.update(task, description=f"Processing {folder}")
            if args.plot:
                files = [f for f in os.listdir(os.path.join(args.outdir, folder)) if f.endswith('.pdf')]
            else:
                files = [f for f in os.listdir(os.path.join(args.outdir, folder)) if f.endswith('.json')]

            # Check if there are less than 5 json files
            if len(files) == 0:
                console.log(f'Warning: {folder} does not contain any results.', style='yellow')
            # elif len(files) < 5:
            #     log_str = f'Warning: {folder} only contains: '
            #     for i in range(len(files)):
            #         log_str += f'{files[i].removeprefix(folder+"_").removesuffix("_results.json")}'
            #         log_str += '.\n' if i == len(files)-1 else ', '
            #     console.log(log_str, style='yellow')

            if args.plot:
                for file in files:
                    # Copy file to new directory
                    os.makedirs(f'plots_from_{args.outdir}', exist_ok=True)
                    shutil.copy(os.path.join(args.outdir, folder, file), f'plots_from_{args.outdir}/{folder}_{file}')
            else:
                for file in files:
                    # Open file and extract variable
                    with open(os.path.join(args.outdir, folder, file), 'r', encoding='utf-8-sig') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            console.log(f'Error: {folder}/{file} is not a valid json file.', style='red')
                            continue

                        if folder not in result_dict:
                            result_dict[folder] = {}
                        result_dict[folder][file.removeprefix(folder+"_").removesuffix("_results.json")] = data[args.var]

            progress.advance(task)

    # Save results to file
    if not args.plot:
        with open(f'results_{args.var}_from_{args.outdir}.json', 'w') as f:
            json.dump(result_dict, f, indent=4)
        console.log(f"Results saved to results_{args.var}_from_{args.outdir}.json")