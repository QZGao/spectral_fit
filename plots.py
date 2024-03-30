import itertools

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from cycler import cycler
from jacobi import propagate
from ultranest.plot import PredictionBand

from env import Env


def sci_notation(num: float):
    e = np.floor(np.log10(abs(num)))
    if e == 0:
        return '%.2f' % num
    s = num / 10**e
    exponent = '%d' % e
    significand = '%d' % s if s.is_integer() else '%.2f' % s
    return f"{{{significand} \\times 10^{{{exponent}}}}}"


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


def plot(dataset, model, model_name_cap, labels, outpath, env: Env,
         param_estimates=None, samples=None, log_evidence=None, log_evidence_err=None,
         iminuit_result=None, aic=None):
    fig, ax = plt.subplots(figsize=(5*4/3, 5))

    if env.args.aic:
        median, plus_minus = param_estimates
    else:
        median, plus, minus = param_estimates

    custom_cycler = (cycler(color = [p[0] for p in MARKER_TYPES])
                    + cycler(marker = [p[1] for p in MARKER_TYPES])
                    + cycler(markersize = np.array([p[2] for p in MARKER_TYPES])*.7))
    ax.set_prop_cycle(custom_cycler)
    prop_cycle = ax._get_lines.prop_cycler

    for g in np.unique(dataset.REF):
        ix = np.where(dataset.REF == g)

        if not env.args.aic and not env.args.no_err:
            if env.args.err_all:  # All with the same systematic error
                combined_err = dataset.combined_err(median[-1], ix)
            else:  # With systematic error for points with yerr/y < threshold
                combined_err = dataset.combined_err(median[-1], ix, thresh=env.args.err_thresh)

            prop_cycle, prop_cycle_copy = itertools.tee(prop_cycle)
            eb = ax.errorbar(dataset.X[ix], dataset.Y[ix], yerr=combined_err,
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

        ax.errorbar(dataset.X[ix], dataset.Y[ix], yerr=dataset.YERR[ix],
                    linestyle='None',
                    mec='k',
                    markeredgewidth=.5,
                    elinewidth=.7,
                    capsize=1.5,
                    label=env.catalogue.citation_dict.get(g, g),
                    **next(prop_cycle))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(linestyle=':')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plot_x = np.logspace(np.log10(dataset.X.min() / 2.), np.log10(dataset.X.max() * 2.), 1000)
    if env.args.aic:
        # AIC prediction band borrowed from pulsar_spectra by Nick Swainston
        # https://github.com/NickSwainston/pulsar_spectra/blob/373f65d866c3bf162fd6d93780235cfbd81849b5/pulsar_spectra/spectral_fit.py#L85-L116
        if iminuit_result.valid:
            try:
                plot_y, plot_y_cov = propagate(lambda p: model(plot_x, *p), iminuit_result.values, iminuit_result.covariance)
            except ValueError:
                plot_y = model(plot_x, *iminuit_result.values)
                plot_yerr = [None] * len(plot_y)
            else:
                plot_yerr = np.diag(plot_y_cov) ** 0.5
        else:
            # No convariance values so use old method
            plot_y = model(plot_x, *iminuit_result.values)
            plot_yerr = [None]*len(plot_y)

        ax.plot(plot_x, plot_y, marker="None", linewidth=1, color='tab:orange')
        if iminuit_result.valid and plot_yerr[0] is not None:
            ax.fill_between(plot_x, plot_y - plot_yerr, plot_y + plot_yerr, color='tab:orange', alpha=0.2)

    else:
        band = PredictionBand(plot_x)
        for sample_array in samples:
            p = {labels[i]: sample_array[i] for i in range(len(sample_array))}
            band.add(model(plot_x, p, dataset.v0))
        band.line(color='tab:orange', marker='', linewidth=1)
        band.shade(color='tab:orange', alpha=0.2)

    if env.args.aic:
        fit_info = f"{model_name_cap}\n$\\mathrm{{AIC}} = {aic:.2f}$"
    else:
        fit_info = f"{model_name_cap}\n$\ln z = {log_evidence:.2f} \pm {log_evidence_err:.2f}$"
    for i in range(len(labels)):
        fit_info += f"\n${labels[i]} = "
        fit_info += f"{sci_notation(median[i])}" if len(f'{median[i]:.2f}') > 8 else f"{median[i]:.2f}"
        if env.args.aic:
            fit_info += f" \pm {sci_notation(plus_minus[i])}$" if len(f'{plus_minus[i]:.2f}') > 8 else f" \pm {plus_minus[i]:.2f}$"
        else:
            fit_info += f"^{{+{sci_notation(plus[i])}}}_{{-{sci_notation(minus[i])}}}$" if len(f'{plus[i]:.2f}') > 8 else f"^{{+{plus[i]:.2f}}}_{{-{minus[i]:.2f}}}$"
        if labels[i].startswith('ν'):
            fit_info += ' MHz'
    fit_info += f"\n$ν_0 = {dataset.v0:.2f}$ MHz"

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Frequency $ν$ (MHz)')
    ax.set_ylabel('Flux density $S$ (mJy)')
    ax.set_title(dataset.jname)

    # Fit info in the lower left corner
    ax.text(0.05, 0.05, fit_info, transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(facecolor='none', edgecolor='none'), linespacing=1.5)

    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.savefig(outpath + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(outpath + '.pdf', bbox_inches='tight')
    plt.close()


# Marker styles borrowed from pulsar_spectra by Nick Swainston
# https://github.com/NickSwainston/pulsar_spectra/blob/373f65d866c3bf162fd6d93780235cfbd81849b5/pulsar_spectra/spectral_fit.py#L170-L189
MARKER_TYPES = [("#006ddb", "o", 6),    # blue circle
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
