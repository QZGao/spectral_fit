## Run locally
### Based on existing results
`output/examples` contains the results of multiple runs of the models with different approaches. Copy the `.json` files to the parent directory `output` to use them in the notebooks. Also copy the `.pkl` files from the `catalogue/examples` directory to their parent directory `catalogue`.

### Run from scratch
Or, you can generate the results from scratch by running the following commands:

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

## Interactivity
Some of the notebooks contain interactive plots. To enable them, change the line `INTERACTIVE = False` to `INTERACTIVE = True` when you found it in the notebook.
