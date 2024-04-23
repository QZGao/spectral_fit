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


def fit_only_env(outdir: str, model_dict: dict, catalogue: Catalogue, **kwargs) -> Env:
    args = Namespace(
            outdir=outdir,
            no_requirements=False,
            fixed_freq_prior=False,
            gaussian=False,
            outliers_rm=False,
            outliers_min=None,
            outliers_all=None,
            err_all=False,
            err_thresh=None,
            aic=False,
            aic_no_corr=False,
            override=False,
        )
    args.__dict__.update(kwargs)
    return Env(
        args=args,
        model_dict=model_dict,
        catalogue=catalogue,
    )
