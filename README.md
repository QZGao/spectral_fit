## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Main dataset

Default fitting (adding systematic error to points where yerr/y ≤ threshold, and fitting with MCMC):
```bash
python fit.py                   # default threshold is 0.1
python fit.py --err_thresh 0.2  # set the threshold to 20%
```

Fitting with all yerr set to 50% of y (no additional systematic error):
```bash
python fit.py --outliers_all 0.5 --no_err
```

Fitting with AIC method:
```bash
python fit.py --aic                # AIC with correction for small sample size
python fit.py --aic --aic_no_corr  # AIC without correction for small sample size
```

### Reproduced Jankowski's dataset

Fitting with AIC method:
```bash
python fit.py --jan_set --aic                # AIC with correction for small sample size
python fit.py --jan_set --aic --aic_no_corr  # AIC without correction for small sample size
```

### Processing data

Extract variables from the output files:
```bash
python processing.py <output_dir>            # default variable is 'log_evidence'
    python processing.py <output_dir> --var "aic"  # extract AIC values
```

Extract estimated parameter values (including errors) from the output files:
```bash
python processing.py <output_dir> --var "param_estimates"
```

Extract frequency-flux plots from the output files (`<jname_list>` is a list of pulsar names, separated by `;`):
```bash
python processing.py <output_dir> --plot <jname_list>
```

### Notebooks
Several notebooks are available in the `./notebooks` directory. They contain an analysis of the results and the data. To run them, you need to have the output files in the `./output` directory. You can generate the output files by running the commands above; or, you can copy the files from the `./output/examples` directory to the `./output` directory, and the files from the `./catalogue/examples` directory to the `./catalogue` directory.
