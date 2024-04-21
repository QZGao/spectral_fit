## Run locally
You can generate everything you need in the notebooks by running the following commands:

```bash
# Back to the root directory if not there
cd ..

# Bayesian
python fit.py --outdir "outdir_st"
python processing.py "outdir_st"
python processing.py "outdir_st" --var "param_estimates"

# Frequentist
python fit.py --outdir "outdir_aic" --aic
python processing.py "outdir_aic" --var "aic"
python processing.py "outdir_aic" --var "param_estimates"

python fit.py --outdir "outdir_aic_without" --aic --aic_no_corr
python processing.py "outdir_aic_without" --var "aic"
python processing.py "outdir_aic_without" --var "param_estimates"

# Frequentist with reproduced Jankowski's set, only run the five models that Jankowski used
python fit.py --outdir "outdir_aic_jankowski" --aic --jan_set --model "simple_power_law;broken_power_law;log_parabolic_spectrum;low_frequency_turn_over_power_law;high_frequency_cut_off_power_law"
python processing.py "outdir_aic_jankowski" --var "aic"
python processing.py "outdir_aic_jankowski" --var "param_estimates"

python fit.py --outdir "outdir_aic_jankowski_without" --aic --aic_no_corr --jan_set --model "simple_power_law;broken_power_law;log_parabolic_spectrum;low_frequency_turn_over_power_law;high_frequency_cut_off_power_law"
python processing.py "outdir_aic_jankowski_without" --var "aic"
python processing.py "outdir_aic_jankowski_without" --var "param_estimates"
```

## Interactivity
Interactive plots are supported in some plots with the help of `mplcursors`. To enable them, change the line `INTERACTIVE = False` to `INTERACTIVE = True` in the code.
