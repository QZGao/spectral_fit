from argparse import Namespace

from catalogue import Catalogue


class Env:
    def __init__(self, args: Namespace, model_dict: dict, catalogue: Catalogue):
        self.args = args
        self.model_dict = model_dict
        self.catalogue = catalogue

    @property
    def outdir(self):
        return self.args.outdir


def fit_only_env(outdir: str = '', model_dict: dict = None, catalogue: Catalogue = None, **kwargs) -> Env:
    args = Namespace(
            outdir=outdir,
            no_requirements=False,
            fixed_freq_prior=False,
            gaussian=True,
            gaussian_patch=False,
            outliers_rm=False,
            outliers_min=None,
            outliers_min_plus=None,
            outliers_all=None,
            equad=False,
            efac=False,
            efac_qbound=0.5,
            err_all=False,
            err_thresh=None,
            aic=False,
            aic_no_corr=False,
            override=True,
            no_checkpoint=False,
            no_plot=False,
            corner=False,
            plot_hide_model_name=False,
            plot_hide_legend=False,
            plot_legend_top=False,
            plot_bigger_font=False,
            pdf=False,
            print_lit=False
        )
    args.__dict__.update(kwargs)
    return Env(
        args=args,
        model_dict=model_dict,
        catalogue=catalogue,
    )
