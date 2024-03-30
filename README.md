## Usage
### Main dataset

Default fitting (with 10% systematic error threshold):
```bash
python fit.py
```

Fitting with 20% error threshold:
```bash
python fit.py --err_thresh 0.2
```

Fitting with all errors set to 50%:
```bash
python fit.py --err_all 0.5
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
