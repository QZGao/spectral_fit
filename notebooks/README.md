## Run locally
You can generate everything you need in the notebooks by running the following commands:

```bash
# Back to the root directory if not there
cd ..

# 10% error threshold
python fit.py --outdir "outdir_10_percent_threshold"
python processing.py "outdir_10_percent_threshold"
python processing.py "outdir_10_percent_threshold" --var "param_estimates"

# 20% error threshold
python fit.py --outdir "outdir_20_percent_threshold" --err_thresh 0.2
python processing.py "outdir_20_percent_threshold"
python processing.py "outdir_20_percent_threshold" --var "param_estimates"

# 50% unified uncertainty
python fit.py --outdir "outdir_50_percent_error" --no_err --outliers_all 0.5
python processing.py "outdir_50_percent_error"
python processing.py "outdir_50_percent_error" --var "param_estimates"

# AIC
python fit.py --outdir "outdir_aic" --aic
python processing.py "outdir_aic" --var "aic"
python processing.py "outdir_aic" --var "param_estimates"

python fit.py --outdir "outdir_aic_without" --aic --aic_no_corr
python processing.py "outdir_aic_without" --var "aic"
python processing.py "outdir_aic_without" --var "param_estimates"

# AIC with reproduced Jankowski's set
python fit.py --outdir "outdir_aic_jankowski" --aic --jan_set
python processing.py "outdir_aic_jankowski" --var "aic"
python processing.py "outdir_aic_jankowski" --var "param_estimates"

python fit.py --outdir "outdir_aic_jankowski_without" --aic --aic_no_corr --jan_set
python processing.py "outdir_aic_jankowski_without" --var "aic"
python processing.py "outdir_aic_jankowski_without" --var "param_estimates"
```

## Interactivity
Some of the notebooks contain interactive plots. To enable them, change the line `INTERACTIVE = False` to `INTERACTIVE = True` when you found it in the notebook.
