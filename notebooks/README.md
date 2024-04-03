## Run locally
All commands needed to run before running the notebooks in the directory:

```bash
python fit.py --outdir "outdir_10_percent_threshold"
python processing.py "outdir_10_percent_threshold"
python processing.py "outdir_10_percent_threshold" --var "param_estimates"

python fit.py --outdir "outdir_20_percent_threshold" --err_thresh 0.2
python processing.py "outdir_20_percent_threshold"
python processing.py "outdir_20_percent_threshold" --var "param_estimates"

python fit.py --outdir "outdir_50_percent_error" --no_err --outliers_all 0.5
python processing.py "outdir_50_percent_error"
python processing.py "outdir_50_percent_error" --var "param_estimates"

python fit.py --outdir "outdir_aic" --aic
python processing.py "outdir_aic" --var "aic"
python processing.py "outdir_aic" --var "param_estimates"

python fit.py --outdir "outdir_aic_jankowski" --aic --jan_set
python processing.py "outdir_aic_jankowski" --var "aic"
python processing.py "outdir_aic_jankowski" --var "param_estimates"
```