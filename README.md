# Pulsar spectrum fitting

This repository contains code to fit pulsar radio spectra using Bayesian methods. The dataset is based on a compilation of literature data, including the latest `pulsar-spectra` catalogue (version 2.1.0) and a set of additional recent measurements. The fitting methods include various spectral models and options for handling outliers and systematic uncertainties.

## Installation

1. Prepare a Python environment between versions 3.9 and 3.10. A virtual environment is recommended.

2. Install the required packages (pandas has to be downgraded to version < 2.0 due to compatibility issues with `pulsar-spectra`):
```bash
pip install -r requirements.txt
```

3. Then, upgrade `pandas` back to the latest version:
```bash
pip install --upgrade pandas
```

## Usage

### Data compilation and fitting

Default Bayesian fitting with Cauchy likelihood:
```bash
python fit.py
```

Method used in the paper: Bayesian fitting with Gaussian likelihood, and a dynamic $e_{\text{fac}}$ that is calculated based on a 50% $e_{\text{quad}}$:
```bash
python fit.py --gaussian --efac_qbound .5
```

Catalogue options:
* `--jname <name>`: Specific pulsar name (string). Can be a single name or multiple names separated by `;`.
* `--lit_set <file>`: Customize literature list (string, default: None).
* `--atnf`: Include ATNF pulsar catalogue (flag, default: False).
* `--atnf_ver <version>`: ATNF pulsar catalogue version (string, default: `1.54`).
* `--jan_set`: Use Jankowski et al. (2018)'s (reproduced) dataset (flag, default: False).
* `--refresh`: Refresh the catalogue (flag, default: False).

Fitting behaviour and priors:
* `--model <models>`: Specific model name(s) (string). Default: `simple_power_law;broken_power_law;log_parabolic_spectrum;high_frequency_cut_off_power_law;low_frequency_turn_over_power_law;double_turn_over_spectrum` (multiple models can be given separated by `;`).
* `--no_requirements`: Do not require the dataset to have at least 4 points and a frequency range of at least 2 (flag, default: False).
* `--fixed_freq_prior`: Use fixed frequency prior (flag, default: False).

Outlier handling / likelihood options:
* `--gaussian`: Use Gaussian likelihood (default is Cauchy likelihood when omitted) (flag, default: False).
* `--gaussian_patch`: Use simplified Gaussian distribution that works better for a small dataset (flag, default: False).
* `--outliers_rm`: Remove outliers from the dataset (flag, default: False).
* `--outliers_min <value>`: Set minimum YERR / Y (when `--outliers_rm` is not set) (float).
* `--outliers_min_plus <value>`: Add a systematic error to the uncertainty instead of replacing it (float).
* `--outliers_all <value>`: Ignore reported YERRs and set all YERR/Y to this value (float).

Systematic error / extra uncertainty parameters:
* `--equad <value>`: Add an additional systematic error (float).
* `--efac <value>`: Multiply reported uncertainties by this systematic error factor (float).
* `--efac_qbound <value>`: Determine $e_{\text{fac}}$ dynamically based on an $e_{\text{quad}}$ bound (float). Example used in the paper: `--efac_qbound .5`.

AIC / Jankowski et al. (2018) method:
* `--aic`: Use Jankowski et al. (2018)'s AIC-based method instead of the Bayesian fit (flag, default: False). Code is adapted from `pulsar-spectra`.
* `--aic_no_corr`: Do not apply the small-sample correction term in the AIC calculation (flag, default: False).

Output, multiprocessing and plotting:
* `--label <label>`: Output directory label (used when `--outdir` is not set).
* `--outdir <dir>`: Output directory (if not set, a timestamped `output/outdir_YYYY-MM-DD_HH-MM-SS` is created).
* `--override`: Override finished jobs (flag, default: False).
* `--nproc <int>`: Number of parallel processes to use (int). Default: `cpu_count() - 1` or 1.
* `--no_checkpoint`: Do not save intermediate pickle dump files (flag, default: False).
* `--no_plot`: Do not create result plots (flag, default: False).
* `--corner`: Generate a corner plot (flag, default: False).
* `--pdf`: Save plots as PDF instead of PNG (flag, default: False).
* `--print_lit`: Print literature list and save it to the output directory (flag, default: False).

### Processing fitted results

Extract variables from the output files:
```bash
python processing.py <output_dir>              # default variable is 'log_evidence'
python processing.py <output_dir> --var "aic"  # extract AIC values
```

Extract estimated parameter values (including errors) from the output files:
```bash
python processing.py <output_dir> --var "param_estimates"
```

Extract frequency-flux plots from the output files (`<filter>` can be a list of pulsar names separated by `;`, or an expression regarding the number of measurements in a pulsar e.g. ">=15"'):
```bash
python processing.py <output_dir> --plot <filter>           # default format to be extracted is "png"
python processing.py <output_dir> --plot <filter> --plot_format "pdf" # specify format
```
