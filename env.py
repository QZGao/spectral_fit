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
