from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()
from sys import exit
import multiprocessing
import json
from traceback import TracebackException
import pickle
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
from pulsar_spectra.spectral_fit import robust_cost_function, huber_loss_function, migrad_simplex_scan, \
    propagate_flux_n_err
from pulsar_spectra.models import simple_power_law, broken_power_law, high_frequency_cut_off_power_law, \
    low_frequency_turn_over_power_law, double_turn_over_spectrum, log_parabolic_spectrum
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt
from cycler import cycler
import warnings
warnings.filterwarnings("ignore")

from models import get_models

args = None
model_dict = None




# Use the built-in catalogue
with open('catalogue_fluxes_0202_good.pkl', 'rb') as f:
    cat_dict = pickle.load(f)

# Use Jankowski's catalogue
# with open('catalogue_fluxes_0202_jankowski.pkl', 'rb') as f:
#     cat_dict = pickle.load(f)

# Use the built-in citation dictionary
with open('catalogue_fluxes_0202_citation_dict.pkl', 'rb') as f:
    citation_dict = pickle.load(f)


def model_settings(print_models=False):
    # fit starting value, min and max
    # constant
    c_s = 1.
    c_min = 0.
    c_max = None
    # spectral index
    a_s = -1.6
    a_min = -8.
    a_max = 3.
    # Beta, he smoothness of the turn-over
    beta_s = 1.
    beta_min = 0.1
    beta_max = 2.1
    # High frequency cut off frequency
    vc_s = 4e9
    vc_both = None # will set the cut off frequency based on the data set's frequency range
    # Lof frequency turn over frequency peak
    vpeak_s = 100e6
    vpeak_min = 10e6
    vpeak_max = 2e9

    model_dict = {
        # Name: [model_function, short_name, start_params, mod_limits]
        "simple_power_law" : [
            simple_power_law,
            "simple pl",
            # (a, c)
            (a_s, c_s),
            [(a_min, a_max), (c_min, c_max)],
        ],
        "broken_power_law" : [
            broken_power_law,
            "broken pl",
            #(vb, a1, a2, c)
            (1e9, a_s, a_s, c_s),
            [(50e6, 5e9), (a_min, a_max), (a_min, a_max), (c_min, c_max)],
        ],
        "log_parabolic_spectrum" : [
            log_parabolic_spectrum,
            "lps",
            #(a, b, c)
            (-1, -1., c_s),
            [(-5, 2), (-5, 2), (None, c_max)],
        ],
        "high_frequency_cut_off_power_law" : [
            high_frequency_cut_off_power_law,
            "pl hard cut-off",
            #(vc, a, c)
            (vc_s, a_s, c_s),
            [vc_both, (a_min, 0.), (c_min, c_max)],
        ],
        "low_frequency_turn_over_power_law" : [
            low_frequency_turn_over_power_law,
            "pl low turn-over",
            #(vpeak, a, c, beta)
            (vpeak_s, a_s, c_s, beta_s),
            [(vpeak_min, vpeak_max), (a_min, 0.), (c_min, c_max) , (beta_min, beta_max)],
        ],
        # "double_turn_over_spectrum" : [
        #     double_turn_over_spectrum,
        #     "double turn over spectrum",
        #     #(vc, vpeak, a, beta, c)
        #     (vc_s, vpeak_s, a_s, beta_s, c_s),
        #     [(vc_both), (vpeak_min, vpeak_max), (a_min, 0.), (beta_min, beta_max), (c_min, c_max)],
        # ],
    }

    if print_models:
        # Print the models dictionary which is useful for debuging new models
        for mod in model_dict.keys():
            print(f"\n{mod}")
            model_function, short_name, start_params, mod_limits, model_function_integrate = model_dict[mod]
            print(f"    model_function:           {model_function.__name__}")
            print(f"    model_function_integrate: {model_function_integrate.__name__}")
            print(f"    short_name:               {short_name}")
            print(f"    start_params:             {start_params}")
            print(f"    mod_limits:               {mod_limits}")

    return model_dict


marker_types = [("#006ddb", "o", 6),  # blue circle
                ("#24ff24", "^", 7),  # green triangle
                ("r", "D", 5),  # red diamond
                ("#920000", "s", 5.5),  # maroon square
                ("#6db6ff", "p", 6.5),  # sky blue pentagon
                ("#ff6db6", "*", 9),  # pink star
                ("m", "v", 7),  # purple upside-down triangle
                ("#b6dbff", "d", 7),  # light blue thin diamond
                ("#009292", "P", 7.5),  # turqoise thick plus
                ("#b66dff", "h", 7),  # lavender hexagon
                ("#db6d00", ">", 7),  # orange right-pointing triangle
                ("c", "H", 7),  # cyan sideways hexagon
                ("#ffb6db", "X", 7.5),  # light pink thick cross
                ("#004949", "<", 7),  # dark green right-pointing triangle
                ("k", "x", 7),  # black thin cross
                ("y", "s", 5.5),  # yellow square
                ("#009292", "^", 7),  # turquoise triangle
                ("k", "d", 7),  # black thin diamond
                ("#b6dbff", "*", 9),  # light blue star
                ("y", "P", 7.5)]  # yellow thick plus


para_dict = {
    'simple_power_law': ['α', 'b', 'ν_0'],
    'broken_power_law': ['ν_b', 'α_1', 'α_2', 'b', 'ν_0'],
    'log_parabolic_spectrum': ['a', 'b', 'c', 'ν_0'],
    'high_frequency_cut_off_power_law': ['ν_c', 'α', 'b', 'ν_0'],
    'low_frequency_turn_over_power_law': ['ν_c', 'α', 'b', 'β', 'ν_0'],
    # 'double_turn_over_spectrum': ['ν_{c1}', 'ν_{c2}', 'α', 'β', 'b', 'ν_0']
}

name_dict = {
    'simple_power_law': 'simple power law',
    'broken_power_law': 'broken power law',
    'log_parabolic_spectrum': 'log-parabolic spectrum',
    'high_frequency_cut_off_power_law': 'high-frequency cut-off power law',
    'low_frequency_turn_over_power_law': 'low-frequency turn-over power law',
    # 'double_turn_over_spectrum': 'double turn-over spectrum'
}


def plot(dataset, model, model_name, iminuit_result, aic, outpath):
    fig, ax = plt.subplots(figsize=(5 * 4 / 3, 5))

    jname, X, Y, YERR, REF, v0 = dataset

    fit_info = f"{name_dict[model_name]}\n$\\mathrm{{AIC}} = {aic:.2f}$"
    params = para_dict[model_name]
    for i in range(len(params)):
        if params[i].startswith('ν'):
            fit_info += f"\n${params[i]} = {iminuit_result.values[i] / 1e6:.2f} \pm {iminuit_result.errors[i] / 1e6:.2f}$ MHz"
        elif len(f'{iminuit_result.values[i]:.2f}') > 8:
            fit_info += f"\n${params[i]} = {iminuit_result.values[i]:.2e} \pm {iminuit_result.errors[i]:.2e}$"
        else:
            fit_info += f"\n${params[i]} = {iminuit_result.values[i]:.2f} \pm {iminuit_result.errors[i]:.2f}$"
    # if 'DM' in cat_dict[jname]:
    #     if 'DMERR' in cat_dict[jname]:
    #         fit_info += f"\n$\\mathrm{{DM}} = {cat_dict[jname]['DM']:.2f} \pm {cat_dict[jname]['DMERR']:.2f}$"
    #     else:
    #         fit_info += f"\n$\\mathrm{{DM}} = {cat_dict[jname]['DM']:.2f}$"

    custom_cycler = (cycler(color=[p[0] for p in marker_types])
                     + cycler(marker=[p[1] for p in marker_types])
                     + cycler(markersize=np.array([p[2] for p in marker_types]) * .7))
    ax.set_prop_cycle(custom_cycler)

    for g in np.unique(REF):
        ix = np.where(REF == g)
        ax.errorbar(X[ix], Y[ix], yerr=YERR[ix],
                    linestyle='None',
                    mec='k',
                    markeredgewidth=.5,
                    elinewidth=.7,
                    capsize=1.5,
                    label=citation_dict.get(g, g))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(linestyle=':')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    fitted_freq = np.logspace(np.log10(X.min() / 2.), np.log10(X.max() * 2.), 1000)
    fitted_flux, fitted_flux_prop = propagate_flux_n_err(fitted_freq, model, iminuit_result)

    ax.plot(fitted_freq, fitted_flux, marker="None", linewidth=1, color='tab:orange')
    if iminuit_result.valid and fitted_flux_prop[0] is not None:
        ax.fill_between(fitted_freq, fitted_flux - fitted_flux_prop, fitted_flux + fitted_flux_prop, color='tab:orange',
                    alpha=0.2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Frequency $ν$ (MHz)')
    ax.set_ylabel('Flux density $S$ (mJy)')
    ax.set_title(jname)

    # Fit info in the lower left corner
    ax.text(0.05, 0.05, fit_info, transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(facecolor='none', edgecolor='none'), linespacing=1.5)

    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.savefig(outpath + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(outpath + '.pdf', bbox_inches='tight')
    plt.close()


def fit_dataset(dataset, model_name, outdir):
    jname, X, Y, YERR, REF, v0 = dataset

    v0_Hz = v0 * 1e6
    freqs_Hz = np.array(X) * 1e6
    fluxs_Jy = np.array(Y) / 1e3
    flux_errs_Jy = np.array(YERR) / 1e3

    # Load model settings
    model_dict = model_settings()
    model_function = model_dict[model_name][0]
    start_params = model_dict[model_name][2]
    mod_limits = model_dict[model_name][3]
    # Add the reference frequency
    start_params += (v0_Hz,)
    mod_limits += [None]

    if (model_name == "high_frequency_cut_off_power_law" or model_name == "double_turn_over_spectrum") and mod_limits[
        0] is None:
        # will set the cut off frequency based on the data set's frequency range
        mod_limits[0] = (max(freqs_Hz), 10 * max(freqs_Hz))
        console.log(f"HFCO cut off frequency limits (Hz): {mod_limits[0]}")
        # Replace vc start param with max frequency
        temp_params = list(start_params)
        temp_params[0] = max(freqs_Hz)
        start_params = tuple(temp_params)

    # Fit model
    least_squares = LeastSquares(freqs_Hz, fluxs_Jy, flux_errs_Jy, model_function)
    least_squares.loss = huber_loss_function
    m = Minuit(least_squares, *start_params)
    m.fixed["v0"] = True  # fix the reference frequency
    m = migrad_simplex_scan(m, mod_limits, model_name)

    # Calculate AIC
    k = len(start_params)-1
    beta = robust_cost_function(model_function(freqs_Hz, *m.values), fluxs_Jy, flux_errs_Jy)
    if len(freqs_Hz) - k - 1 != 0:
        aic = 2 * beta + 2 * k + (2 * k * (k + 1)) / (len(freqs_Hz) - k - 1)
    else:
        aic = 2 * beta + 2 * k

    # Save the results
    with open(f'{outdir}/{jname}/{model_name}_results.json', 'w', encoding='utf-8-sig') as f:
        json.dump({
            'aic': aic,
            'param_estimates': {
                'value': m.values.to_dict(),
                'error': m.errors.to_dict(),
            },
            'dataset': {
                'jname': jname,
                'X': X.tolist(),
                'Y': Y.tolist(),
                'YERR': YERR.tolist(),
                'REF': REF.tolist(),
                'v0': v0,
            },
            'model': model_name,
            'params': m.parameters,
            'start_params': start_params,
            'limits': mod_limits,
        }, f, ensure_ascii=False, indent=4)

    # Plots
    plot(
        dataset=dataset,
        model=model_function,
        model_name=model_name,
        iminuit_result=m,
        aic=aic,
        outpath=f'{outdir}/{jname}/{model_name}_result',
    )


def fit(jname, model_name, outdir):
    # Use the built-in catalogue
    X = np.array(cat_dict[jname]['X'])
    Y = np.array(cat_dict[jname]['Y'])
    YERR = np.array(cat_dict[jname]['YERR'])
    REF = np.array(cat_dict[jname]['REF'])

    # If there are less unique Xs than the number of parameters, skip
    Xs_num = len(np.unique(X))
    params_num = len(model_settings()[model_name][2])
    if Xs_num < params_num:
        return

    # If Xs do not span at least a factor of 2, skip
    if X.max() / X.min() < 2:
        return

    v0 = 10 ** ((np.log10(X.max()) + np.log10(X.min())) / 2)  # central frequency
    dataset = jname, X, Y, YERR, REF, v0

    Path(f'{outdir}/{jname}').mkdir(parents=True, exist_ok=True)
    try:
        fit_dataset(dataset, model_name, outdir=outdir)
    except Exception as e:
        console.log(f"Error: {jname} {model_name} {e}\n{''.join(TracebackException.from_exception(e).format())}", style='red')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument('--override', help="Override finished jobs", action='store_true')
    parser.add_argument('--jname', help="Specific pulsar name")
    parser.add_argument('--label', help="Output directory label (not compatible with --outdir)", default="AIC")
    args = parser.parse_args()

    model_dict = model_settings()

    if args.outdir:
        base_outdir = args.outdir
    else:
        base_outdir = f'outdir_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        if args.label:
            base_outdir += f'_{args.label.replace(" ", "_")}'
    Path(base_outdir).mkdir(parents=True, exist_ok=True)
    console.log(f"outdir: {base_outdir}")

    job_list = []
    if args.jname:
        if ';' in args.jname:
            jname_list = args.jname.split(';')
            for jname in jname_list:
                for model_name in model_dict:
                    job_list.append((jname, model_name, base_outdir))
        else:
            if args.jname not in cat_dict:
                console.log(f"Error: {args.jname} not found in the catalogue.", style='red')
                exit()
            for model_name in model_dict:
                job_list.append((args.jname, model_name, base_outdir))
    else:
        for jname in cat_dict:
            for model_name in model_dict:
                job_list.append((jname, model_name, base_outdir))

    # Remove finished jobs
    if args.outdir and not args.override:
        for jname, model_name, outdir in job_list[:]:
            if (Path(f'{outdir}/{jname}/{model_name}_result.png').exists() and
                Path(f'{outdir}/{jname}/{model_name}_results.json').exists()):
                with open(f'{outdir}/{jname}/{model_name}_results.json', 'r', encoding='utf-8-sig') as f:
                    try:
                        data = json.load(f)
                        if 'aic' in data:
                            job_list.remove((jname, model_name, outdir))
                    except json.JSONDecodeError:
                        pass

    if len(job_list) == 0:
        console.log("All jobs finished.")
        exit()
    console.log(f"Number of jobs: {len(job_list)}")

    with Progress() as progress:
        task = progress.add_task("[cyan]Fitting", total=len(job_list))

        # Single core
        # count = 0
        # for jname, model_name, outdir in job_list:
        #     fit(jname, model_name, outdir)
        #     progress.update(task, advance=1)
        #     count += 1
        #     if count > 5:
        #         exit()

        # Multiprocessing
        with multiprocessing.Pool() as pool:
            for jname, model_name, outdir in job_list:
                pool.apply_async(
                    fit,
                    args=(jname, model_name, outdir),
                    callback=lambda _: progress.update(task, advance=1),
                )
            pool.close()
            pool.join()

    console.log("All jobs finished.")
