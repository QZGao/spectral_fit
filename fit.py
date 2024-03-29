import itertools
import traceback
from pathlib import Path
from rich.progress import Progress
from rich.console import Console
console = Console()
from sys import exit
import multiprocessing
import json
import pickle
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.special import logsumexp
from dynesty import DynamicNestedSampler
from corner import corner
from dynesty.utils import quantile as _quantile
from dynesty.utils import resample_equal
from ultranest.plot import PredictionBand
import warnings
warnings.filterwarnings("ignore")


with open('catalogue_fluxes_0202_good.pkl', 'rb') as f:
    cat_dict = pickle.load(f)
with open('catalogue_fluxes_0202_citation_dict.pkl', 'rb') as f:
    citation_dict = pickle.load(f)


# simple power law
def simple_power_law(v, p: dict, v0, **kwargs):
    x = v / v0
    return p['b'] * np.power(x, p['α'])


# broken power law
def broken_power_law(v, p: dict, v0, **kwargs):
    x = v / v0
    xb = p['ν_b'] / v0
    y1 = p['b'] * np.power(x, p['α_1'])
    y2 = p['b'] * np.power(x, p['α_2']) * np.power(xb, p['α_1'] - p['α_2'])
    return np.where(x <= xb, y1, y2)


# double broken power law
def double_broken_power_law(v, p: dict, v0, **kwargs):
    x = v / v0
    xb1 = p['ν_{b1}'] / v0
    xb2 = p['ν_{b2}'] / v0
    y1 = p['b'] * np.power(x, p['α_1'])
    y2 = p['b'] * np.power(x, p['α_2']) * np.power(xb1, p['α_1'] - p['α_2'])
    y3 = p['b'] * np.power(x, p['α_3']) * np.power(xb1, p['α_1'] - p['α_2']) * np.power(xb2, p['α_2'] - p['α_3'])
    return np.where(x <= xb1, y1, np.where(x <= xb2, y2, y3))


# logarithmic parabolic spectrum
def log_parabolic_spectrum(v, p: dict, v0, **kwargs):
    x = np.log10(v / v0)
    return np.power(10., p['a'] * x * x + p['b'] * x + p['c'])


# high-frequency cut-off power law (Jankowski et al. 2018 ver.)
def high_frequency_cut_off_power_law_jan(v, p: dict, v0, **kwargs):
    x = v / v0
    xc = p['ν_c'] / v0
    return p['b'] * np.power(x, -2.) * (1. - x / xc)


# high-frequency cut-off power law
def high_frequency_cut_off_power_law(v, p: dict, v0, **kwargs):
    x = v / v0
    xc = p['ν_c'] / v0
    return p['b'] * np.power(x, p['α']) * (1. - x / xc)


# low-frequency turn-over power law
def low_frequency_turn_over_power_law(v, p: dict, v0, **kwargs):
    x = v / v0
    xc = v / p['ν_c']
    return p['b'] * np.power(x, p['α']) * np.exp(p['α'] * np.power(xc, -p['β']) / p['β'])


# double turn-over spectrum
def double_turn_over_spectrum(v, p: dict, v0, **kwargs):
    x = v / v0
    xc1 = p['ν_{c1}'] / v0
    xc2 = v / p['ν_{c2}']
    y1 = p['b'] * np.power(x, p['α']) * (1. - x / xc1) * np.exp(p['α'] / p['β'] * np.power(xc2, -p['β']))
    y2 = 0.
    return np.where(x <= xc1, y1, y2)


model_dict = {
    "simple_power_law" : {
        "model": simple_power_law,
        "labels": ['α', 'b'],
        "priors": [(-10., 0.), (0., 1e3)],
    },
    "broken_power_law" : {
        "model": broken_power_law,
        "labels": ['ν_b', 'α_1', 'α_2', 'b'],
        "priors": ['dynamic', (-5., 5.), (-5., 5.), (0., 1e3)],
    },
    # "double_broken_power_law" : {
    #     "model": double_broken_power_law,
    #     "labels": ['ν_{b1}', 'ν_{b2}', 'α_1', 'α_2', 'α_3', 'b'],
    #     "priors": ['dynamic', 'dynamic', (-5., 5.), (-5., 5.), (-5., 5.), (0., 1e3)],
    # },
    "log_parabolic_spectrum" : {
        "model": log_parabolic_spectrum,
        "labels": ['a', 'b', 'c'],
        "priors": [(-5., 2.), (-5., 2.), (-10., 10.)],
    },
    "high_frequency_cut_off_power_law" : {
        "model": high_frequency_cut_off_power_law,
        "labels": ['ν_c', 'α', 'b'],
        "priors": ['dynamic_expanded', (-20., 5.), (0., 1e3)],
    },
    # "high_frequency_cut_off_power_law_jan" : {
    #     "model": high_frequency_cut_off_power_law_jan,
    #     "labels": ['ν_c', 'b'],
    #     "priors": ['dynamic_expanded', (0., 1e3)],
    # },
    "low_frequency_turn_over_power_law" : {
        "model": low_frequency_turn_over_power_law,
        "labels": ['ν_c', 'α', 'b', 'β'],
        "priors": ['dynamic', (-20., 5.), (0., 1e3), (0., 2.1)],
    },
    # "double_turn_over_spectrum" : {
    #     "model": double_turn_over_spectrum,
    #     "labels": ['ν_{c1}', 'ν_{c2}', 'α', 'b', 'β'],
    #     "priors": ['dynamic_expanded', 'dynamic', (-20., 5.), (0., 1e3), (0., 2.1)],
    # },
}

name_dict = {
    'simple_power_law': 'simple power law',
    'broken_power_law': 'broken power law',
    'log_parabolic_spectrum': 'log-parabolic spectrum',
    'high_frequency_cut_off_power_law': 'high-frequency cut-off power law',
    # 'high_frequency_cut_off_power_law_jan': 'high-frequency cut-off power law',
    'low_frequency_turn_over_power_law': 'low-frequency turn-over power law',
    # 'double_turn_over_spectrum': 'double turn-over spectrum',
}

marker_types = [("#006ddb", "o", 6),    # blue circle
                ("#24ff24", "^", 7),    # green triangle
                ("r",       "D", 5),    # red diamond
                ("#920000", "s", 5.5),  # maroon square
                ("#6db6ff", "p", 6.5),  # sky blue pentagon
                ("#ff6db6", "*", 9),    # pink star
                ("m",       "v", 7),    # purple upside-down triangle
                ("#b6dbff", "d", 7),    # light blue thin diamond
                ("#009292", "P", 7.5),  # turqoise thick plus
                ("#b66dff", "h", 7),    # lavender hexagon
                ("#db6d00", ">", 7),    # orange right-pointing triangle
                ("c",       "H", 7),    # cyan sideways hexagon
                ("#ffb6db", "X", 7.5),  # light pink thick cross
                ("#004949", "<", 7),    # dark green right-pointing triangle
                ("k",       "x", 7),    # black thin cross
                ("y",       "s", 5.5),  # yellow square
                ("#009292", "^", 7),    # turquoise triangle
                ("k",       "d", 7),    # black thin diamond
                ("#b6dbff", "*", 9),    # light blue star
                ("y",       "P", 7.5)]  # yellow thick plus


def plot_corner(samples, labels, scales, outpath):
    fig, axes = plt.subplots(len(labels), len(labels), figsize=(3*len(labels), 3*len(labels)))
    corner(
        samples,
        bins=100,
        smooth=0.9,
        smooth1d=0.9,
        color='tab:blue',
        labels=[f"${label}$" for label in labels],
        label_kwargs=dict(fontsize=14),
        show_titles=True,
        title_fmt='.2f',
        title_kwargs=dict(fontsize=14),
        quantiles=[0.5-0.3413, 0.5, 0.5+0.3413],
        use_math_text=True,
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        fig=fig,
        axes_scale=scales,
    )
    plt.savefig(outpath + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot(dataset, model, model_name, labels, samples, param_estimates, log_evidence, log_evidence_err, outpath):
    fig, ax = plt.subplots(figsize=(5*4/3, 5))

    jname, X, Y, YERR, REF, v0 = dataset
    median, plus, minus = param_estimates

    custom_cycler = (cycler(color = [p[0] for p in marker_types])
                    + cycler(marker = [p[1] for p in marker_types])
                    + cycler(markersize = np.array([p[2] for p in marker_types])*.7))
    ax.set_prop_cycle(custom_cycler)
    prop_cycle = ax._get_lines.prop_cycler

    for g in np.unique(REF):
        ix = np.where(REF == g)

        if labels[-1] == 'σ':

            # Systematic error for all points
            # combined_err = np.sqrt(YERR[ix]**2 + median[-1]**2 * Y[ix]**2)

            # Systematic error only for the YERR / Y < 10% cases
            combined_err = np.where(YERR[ix] / Y[ix] < 0.1, np.sqrt(YERR[ix]**2 + median[-1]**2 * Y[ix]**2), YERR[ix])

            # Systematic error only for the YERR / Y < 20% cases
            # combined_err = np.where(YERR[ix] / Y[ix] < 0.2, np.sqrt(YERR[ix]**2 + median[-1]**2 * Y[ix]**2), YERR[ix])

            prop_cycle, prop_cycle_copy = itertools.tee(prop_cycle)
            eb = ax.errorbar(X[ix], Y[ix], yerr=combined_err,
                        linestyle='None',
                        mec='k',
                        markeredgewidth=.5,
                        elinewidth=.7,
                        capsize=1.5,
                        label='_nolegend_',
                        alpha=0.5,
                        **next(prop_cycle_copy))
            # Set the line dashed
            eb[-1][0].set_linestyle('--')

        ax.errorbar(X[ix], Y[ix], yerr=YERR[ix],
                    linestyle='None',
                    mec='k',
                    markeredgewidth=.5,
                    elinewidth=.7,
                    capsize=1.5,
                    label=citation_dict.get(g, g),
                    **next(prop_cycle))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(linestyle=':')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plot_x = np.logspace(np.log10(X.min() / 2.), np.log10(X.max() * 2.), 1000)
    band = PredictionBand(plot_x)
    for sample_array in samples:
        p = {labels[i]: sample_array[i] for i in range(len(sample_array))}
        band.add(model(plot_x, p, v0))

    fit_info = f"{name_dict[model_name]}\n$\ln z = {log_evidence:.2f} \pm {log_evidence_err:.2f}$"
    for i in range(len(labels)):
        if labels[i].startswith('ν'):
            fit_info += f"\n${labels[i]} = {median[i]:.2f}^{{+{plus[i]:.2f}}}_{{-{minus[i]:.2f}}}$ MHz"
        elif len(f'{median[i]:.2f}') > 8:
            fit_info += f"\n${labels[i]} = {median[i]:.2e}^{{+{plus[i]:.2e}}}_{{-{minus[i]:.2e}}}$"
        else:
            fit_info += f"\n${labels[i]} = {median[i]:.2f}^{{+{plus[i]:.2f}}}_{{-{minus[i]:.2f}}}$"
    fit_info += f"\n$ν_0 = {v0:.2f}$ MHz"
    # if 'DM' in cat_dict[jname]:
    #     if 'DMERR' in cat_dict[jname]:
    #         fit_info += f"\n$\\mathrm{{DM}} = {cat_dict[jname]['DM']:.2f} \pm {cat_dict[jname]['DMERR']:.2f}$"
    #     else:
    #         fit_info += f"\n$\\mathrm{{DM}} = {cat_dict[jname]['DM']:.2f}$"

    band.line(color='tab:orange', marker='', linewidth=1)
    band.shade(color='tab:orange', alpha=0.2)

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


def log_likelihood(p, model, labels, dataset):
    jname, X, Y, YERR, REF, v0 = dataset
    Y_model = model(
        X,
        {labels[i]: p[i] for i in range(len(p))},
        v0,
    )

    # Without systematic error
    combined_err = YERR

    # All with the same systematic error
    # combined_err = np.sqrt(YERR**2 + p[-1]**2 * Y**2)

    # Systematic error only for the YERR / Y < 10% cases
    combined_err = np.where(YERR / Y < 0.1, np.sqrt(YERR**2 + p[-1]**2 * Y**2), YERR)

    # Systematic error only for the YERR / Y < 20% cases
    # combined_err = np.where(YERR / Y < 0.2, np.sqrt(YERR**2 + p[-1]**2 * Y**2), YERR)

    return np.sum(np.log(np.exp( - (Y - Y_model) ** 2 / (combined_err * combined_err) ) * (2 * np.pi) ** -0.5 / combined_err))


def prior_transform(u, priors):
    p = np.zeros_like(u)
    for i, q in enumerate(priors):
        p[i] = q[0] + u[i] * (q[1] - q[0])
    return p


def fit_dataset(dataset, model_name, outdir):
    jname, X, Y, YERR, REF, v0 = dataset

    model = model_dict[model_name]['model']
    labels = model_dict[model_name]['labels'].copy()
    priors = model_dict[model_name]['priors'].copy()
    for i in range(len(priors)):

        # Decide based on the current dataset
        if priors[i] == 'dynamic':
            priors[i] = (X.min(), X.max())
        elif priors[i] == 'dynamic_expanded':
            priors[i] = (X.min(), X.max() * 1.5)

        # Decide based on the whole picture
        # if priors[i] == 'dynamic' or priors[i] == 'dynamic_expanded':
        #     priors[i] = (10., 515250.)

    # Add a prior for the systematic error
    priors.append((0., (YERR / Y).max()))  # up to 50% of the flux density
    labels.append('σ')

    if Path(f'{outdir}/{jname}/{model_name}_dres.pkl').exists():
        with open(f'{outdir}/{jname}/{model_name}_dres.pkl', 'rb') as f:
            dres = pickle.load(f)

    else:
        sampler = DynamicNestedSampler(
            loglikelihood=lambda *args: log_likelihood(*args, model=model, labels=labels, dataset=dataset),
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
        with open(f'{outdir}/{jname}/{model_name}_dres.pkl', 'wb') as f:
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
    median = [_quantile(posteriors[label], [0.5])[0] for label in labels]
    lower = [_quantile(posteriors[label], [0.5-0.3413])[0] for label in labels]
    upper = [_quantile(posteriors[label], [0.5+0.3413])[0] for label in labels]
    plus = (np.array(upper) - np.array(median)).tolist()
    minus = (np.array(median) - np.array(lower)).tolist()

    # Save the results
    with open(f'{outdir}/{jname}/{model_name}_results.json', 'w', encoding='utf-8-sig') as f:
        json.dump({
            'log_evidence': float(log_evidence),
            'log_evidence_err': float(log_evidence_err),
            'param_estimates': {
                'median': median,
                'plus': plus,
                'minus': minus,
            },
            'dataset': {
                'jname': str(jname),
                'X': X.tolist(),
                'Y': Y.tolist(),
                'YERR': YERR.tolist(),
                'REF': REF.tolist(),
                'v0': float(v0),
            },
            'model': model_name,
            'params': labels,
            'priors': priors,
        }, f, ensure_ascii=False, indent=4)

    # Plots
    plot_corner(
        samples=samp_all,
        labels=labels,
        scales=['log' if p == 'dynamic' else 'linear' for p in priors],
        outpath=f'{outdir}/{jname}/{model_name}_corner',
    )
    plot(
        dataset=dataset,
        model=model,
        model_name=model_name,
        labels=labels,
        samples=samp_all,
        param_estimates=(median, plus, minus),
        log_evidence=log_evidence,
        log_evidence_err=log_evidence_err,
        outpath=f'{outdir}/{jname}/{model_name}_result',
    )


def fit(jname, model_name, outdir):
    X = np.array(cat_dict[jname]['X'])
    Y = np.array(cat_dict[jname]['Y'])
    YERR = np.array(cat_dict[jname]['YERR'])
    REF = np.array(cat_dict[jname]['REF'])

    # Remove outlier YERRs
    # YERR_Y = YERR / Y
    # while np.median(YERR_Y) / np.min(YERR_Y) >= 10.:
    #     ix = np.where(YERR_Y == np.min(YERR_Y))
    #     X = np.delete(X, ix)
    #     Y = np.delete(Y, ix)
    #     YERR = np.delete(YERR, ix)
    #     REF = np.delete(REF, ix)
    #     YERR_Y = YERR / Y

    # Set minimum YERR to 10% of Y
    # YERR = np.where(YERR / Y < 0.1, 0.1 * Y, YERR)

    # Set minimum YERR to 20% of Y
    # YERR = np.where(YERR / Y < 0.2, 0.2 * Y, YERR)

    # Don't believe in any of the YERRs
    # YERR = np.ones_like(Y) * 0.5 * Y

    # If there are less unique Xs than the number of parameters, skip
    if len(np.unique(X)) < len(model_dict[model_name]['labels']):
        return

    # If Xs do not span at least a factor of 2, skip
    if X.max() / X.min() < 2:
        return

    v0 = 10**((np.log10(X.max())+np.log10(X.min()))/2) # central frequency
    dataset = jname, X, Y, YERR, REF, v0

    Path(f'{outdir}/{jname}').mkdir(parents=True, exist_ok=True)
    try:
        fit_dataset(dataset, model_name, outdir=outdir)
    except Exception as e:
        console.log(f"Error: {jname} {model_name}\n{e}\n{traceback.format_exc()}", style='red')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument('--override', help="Override finished jobs", action='store_true')
    parser.add_argument('--jname', help="Specific pulsar name")
    parser.add_argument('--label', help="Output directory label (not compatible with --outdir)")
    args = parser.parse_args()

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
            if Path(f'{outdir}/{jname}/{model_name}_result.png').exists() and Path(f'{outdir}/{jname}/{model_name}_results.json').exists() and Path(f'{outdir}/{jname}/{model_name}_dres.pkl').exists() and Path(f'{outdir}/{jname}/{model_name}_corner.png').exists():
                with open(f'{outdir}/{jname}/{model_name}_results.json', 'r', encoding='utf-8-sig') as f:
                    try:
                        data = json.load(f)
                        if 'log_evidence' in data:
                            job_list.remove((jname, model_name, outdir))
                    except json.JSONDecodeError:
                        pass

    if len(job_list) == 0:
        console.log("All jobs finished.")
        exit()
    console.log(f"Number of jobs: {len(job_list)}")

    with Progress() as progress:
        task = progress.add_task("[cyan]Fitting", total=len(job_list))
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