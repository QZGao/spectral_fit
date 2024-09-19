import numpy as np
from matplotlib import pyplot as plt

from catalogue import Dataset
from env import Env


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


def get_models(model_names: str | list, aic: bool = False, custom_p_b: tuple = None) -> dict:
    if aic:
        # Use the models from pulsar_spectra instead
        from pulsar_spectra.models import simple_power_law, broken_power_law, double_broken_power_law, \
            log_parabolic_spectrum, high_frequency_cut_off_power_law, low_frequency_turn_over_power_law, \
            double_turn_over_spectrum
    else:
        global simple_power_law, broken_power_law, double_broken_power_law, log_parabolic_spectrum, \
            high_frequency_cut_off_power_law, low_frequency_turn_over_power_law, double_turn_over_spectrum

    # Common parameters
    p_alpha = (-5., 5., 'uniform')  # Spectral index
    p_alpha_half = (-5., 0., 'uniform')  # Spectral index
    p_beta = (0., 2.1, 'uniform')   # Smoothness of the turn-over
    p_b = (1e-2, 1e4, 'log_uniform')  # Scaling factor
    if custom_p_b:
        p_b = custom_p_b

    # 'start_params' and 'limits' are used in AIC calculation via pulsar_spectra
    model_dict = {
        'simple_power_law' : {
            'name': 'simple power law',
            'model': simple_power_law,
            'labels': ['α', 'b'],
            'priors': [p_alpha_half, p_b],
            'start_params': [-1.6, 100.],
            'limits': [(-5., 0.), (0., None)],
        },
        'broken_power_law' : {
            'name': 'broken power law',
            'model': broken_power_law,
            'labels': ['ν_b', 'α_1', 'α_2', 'b'],
            'priors': ['dynamic', p_alpha, p_alpha_half, p_b],
            'start_params': ['xmid', 2., -1.6, 100.],
            'limits': ['dynamic', (-5., 5.), (-5., 0.), (0., None)],
        },
        'double_broken_power_law' : {
            'name': 'double broken power law',
            'model': double_broken_power_law,
            'labels': ['ν_{b1}', 'ν_{b2}', 'α_1', 'α_2', 'α_3', 'b'],
            'priors': ['dynamic', 'dynamic', p_alpha, p_alpha, p_alpha_half, p_b],
            'start_params': ['xmid', 'xmid', -1.6, -1.6, -1.6, 100.],
            'limits': ['dynamic', 'dynamic', (-5., 5.), (-5., 5.), (-5., 0.), (0., None)],
        },
        'log_parabolic_spectrum' : {
            'name': 'log-parabolic spectrum',
            'model': log_parabolic_spectrum,
            'labels': ['a', 'b', 'c'],
            'priors': [(-5., 2., 'uniform'), (-5., 2., 'uniform'), (-10., 10., 'uniform')],
            'start_params': [-1., -1., 0.],
            'limits': [(-5., 2.), (-5., 2.), (None, None)],
        },
        'high_frequency_cut_off_power_law' : {
            'name': 'high-frequency cut-off power law',
            'model': high_frequency_cut_off_power_law,
            'labels': ['ν_c', 'α', 'b'],
            'priors': ['dynamic_expanded', p_alpha_half, p_b],
            'start_params': ['xmax', -1.6, 1.],
            'limits': ['dynamic_expanded', (-5., 0.), (0., None)],
        },
        # 'high_frequency_cut_off_power_law_jan' : {
        #     'name': 'high-frequency cut-off power law',
        #     'model': high_frequency_cut_off_power_law_jan,
        #     'labels': ['ν_c', 'b'],
        #     'priors': ['dynamic_expanded', b],
        # },
        'low_frequency_turn_over_power_law' : {
            'name': 'low-frequency turn-over power law',
            'model': low_frequency_turn_over_power_law,
            'labels': ['ν_c', 'α', 'b', 'β'],
            'priors': ['dynamic', p_alpha_half, p_b, p_beta],
            'start_params': ['xmid', -1.6, 100., 1.],
            'limits': ['dynamic', (-5., 0.), (0., None) , (0., 2.1)],
        },
        'double_turn_over_spectrum' : {
            'name': 'double turn-over spectrum',
            'model': double_turn_over_spectrum,
            'labels': ['ν_{c1}', 'ν_{c2}', 'α', 'β', 'b'],
            'priors': ['dynamic_expanded', 'dynamic', p_alpha_half, p_beta, p_b],
            'start_params': ['xmax', 'xmid', -1.6, 100., 1.],
            'limits': ['dynamic_expanded', 'dynamic', (-5., 0.), (0., 2.1), (0., None)],
        },
    }

    if isinstance(model_names, str):
        model_name_list = [model_name.strip() for model_name in model_names.split(';') if model_name.strip()]
    else:
        model_name_list = model_names
    for model_name in model_name_list:
        if model_name not in model_dict:
            raise ValueError(f'Model {model_name} not found in model_dict.')
    for model_name in list(model_dict.keys()):
        if model_name not in model_name_list:
            model_dict.pop(model_name)

    return model_dict


def is_good_fit(dataset: Dataset, params_dict: dict, fitted_model, env: Env) -> bool:
    err = dataset.YERR.copy()
    if env.args.efac or env.args.equad:  # Apply the error factors
        ref_set = list(set(dataset.REF))
        for ref in ref_set:
            if env.args.efac:
                err[dataset.REF == ref] *= params_dict['e_{\mathrm{fac,\,' + ref.replace('_', '\_') + '}}']
            if env.args.equad:
                err[dataset.REF == ref] = np.sqrt(
                    err[dataset.REF == ref] ** 2 +
                    params_dict['e_{\mathrm{quad,\,' + ref.replace('_', '\_') + '}}'] ** 2 * dataset.Y[dataset.REF == ref] ** 2
                )
    err = np.maximum(err, .5 * dataset.Y)  # lower bound for the error

    fig, ax = plt.subplots()  # Create a dummy plot to calculate y limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.errorbar(dataset.X, dataset.Y, yerr=err, linestyle='None')
    ylim = ax.get_ylim()  # y limits of the dummy plot
    plt.close()

    fitted_Y = np.maximum(fitted_model(dataset.X), ylim[0] * .1)
    log_residuals = np.abs(np.log10(dataset.Y) - np.log10(fitted_Y))
    log_half_errbar = .5 * (np.log10(dataset.Y + err) - np.log10(np.maximum(dataset.Y - err, 1e-10)))
    n_outliers = np.sum(log_residuals / log_half_errbar > 3)
    good_fit = (n_outliers / len(dataset)) <= .2  # no more than 20% outliers

    return good_fit