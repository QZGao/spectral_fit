## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Main dataset

Default Bayesian fitting:
```bash
python fit.py
```

Frequentist fitting (reproduction of Jankowski's method):
```bash
python fit.py --aic                # AIC with correction for small sample size
python fit.py --aic --aic_no_corr  # AIC without correction for small sample size
```

### Reproduced Jankowski's dataset

Frequentist fitting:
```bash
python fit.py --jan_set --aic                # AIC with correction for small sample size
python fit.py --jan_set --aic --aic_no_corr  # AIC without correction for small sample size
```

### Processing data

Extract variables from the output files:
```bash
python processing.py <output_dir>              # default variable is 'log_evidence'
python processing.py <output_dir> --var "aic"  # extract AIC values
```

Extract estimated parameter values (including errors) from the output files:
```bash
python processing.py <output_dir> --var "param_estimates"
```

Extract frequency-flux plots from the output files (`<filter>` can be a list of pulsar names separated by `;`, or an expression regarding the number of measurements in a pulsar e.g. `">=15"`):
```bash
python processing.py <output_dir> --plot <filter>
```

### Other parameters

See the help message for more information:
```bash
python fit.py --help
python processing.py --help
```
