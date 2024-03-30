from catalogue import Catalogue


class Env:
    def __init__(self, args: dict, model_dict: dict, catalogue: Catalogue, outdir: str):
        self.args = args
        self.model_dict = model_dict
        self.catalogue = catalogue
        self.outdir = outdir
