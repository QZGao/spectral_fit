## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Main dataset

Default fitting (adding systematic error to points where yerr/y ≤ 10%):
```bash
python fit.py
```

Fitting with added systematic error to points where yerr/y ≤ 20%:
```bash
python fit.py --err_thresh 0.2
```

Fitting with all yerr set to 50% of y (no additional systematic error):
```bash
python fit.py --outliers_all 0.5 --no_err
```

Fitting with AIC method:
```bash
python fit.py --aic
```

### Reproduced Jankowski's dataset

Fitting with AIC method:
```bash
python fit.py --jan_set --aic
```

### Processing data

Extract log evidence values from the output files:
```bash
python processing.py <output_dir>
```

Extract AIC values from the output files:
```bash
python processing.py <output_dir> --var aic
```

Extract estimated parameter values (including errors) from the output files:
```bash
python processing.py <output_dir> --var param_estimates
```

Extract frequency-flux plots from the output files (`<jname_list>` is a list of pulsar names, separated by `;`):
```bash
python processing.py <output_dir> --plot <jname_list>
```

### Notebooks
Several notebooks are available in the `notebooks` directory. They contain an analysis of the results and the data. To run them, you need to have the output files in the `output` directory. You can generate the output files by running the commands above; or, you can copy the files from the `output/examples` directory to the `output` directory, and the files from the `catalogue/examples` directory to the `catalogue` directory.
