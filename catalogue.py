import pickle

import numpy as np


class Dataset:
    def __init__(self, jname, X, Y, YERR, REF, v0: float):
        self.jname = jname
        self.X = X
        self.Y = Y
        self.YERR = YERR
        self.REF = REF
        self.v0 = v0

    def to_dict(self) -> dict:
        return {
            'jname': self.jname,
            'X': self.X.tolist(),
            'Y': self.Y.tolist(),
            'YERR': self.YERR.tolist(),
            'REF': self.REF.tolist(),
            'v0': self.v0,
        }

    def yerr_y(self, ix=None):
        if ix is None:
            return self.YERR / self.Y
        else:
            return self.YERR[ix] / self.Y[ix]

    def max_yerr_y(self):
        return np.max(self.yerr_y())

    def combined_err(self, sigma: float, ix=None, threshold: float = None):
        if threshold is None:
            if ix is None:
                return np.sqrt(self.YERR ** 2 + sigma ** 2 * self.Y ** 2)
            else:
                return np.sqrt(self.YERR[ix] ** 2 + sigma ** 2 * self.Y[ix] ** 2)
        else:
            if ix is None:
                return np.where(self.yerr_y() < threshold,
                                np.sqrt(self.YERR ** 2 + sigma ** 2 * self.Y ** 2),
                                self.YERR)
            else:
                return np.where(self.yerr_y(ix) < threshold,
                                np.sqrt(self.YERR[ix] ** 2 + sigma ** 2 * self.Y[ix] ** 2),
                                self.YERR[ix])


class Catalogue:
    def __init__(self, cat_dict, citation_dict):
        self.cat_dict = cat_dict
        self.citation_dict = citation_dict

    def get_pulsar(self, jname: str, nparams: int, args) -> Dataset | None:
        if jname not in self.cat_dict:
            return

        X = np.array(self.cat_dict[jname]['X'])
        Y = np.array(self.cat_dict[jname]['Y'])
        YERR = np.array(self.cat_dict[jname]['YERR'])
        REF = np.array(self.cat_dict[jname]['REF'])

        # Remove outlier YERRs
        if args.outliers_rm:
            YERR_Y = YERR / Y
            while np.median(YERR_Y) / np.min(YERR_Y) >= 10.:
                ix = np.where(YERR_Y == np.min(YERR_Y))
                X = np.delete(X, ix)
                Y = np.delete(Y, ix)
                YERR = np.delete(YERR, ix)
                REF = np.delete(REF, ix)
                YERR_Y = YERR / Y
        else:
            # Set minimum YERR / Y
            if args.outliers_set_min:
                YERR = np.where(YERR / Y < args.outliers_set_min, args.outliers_set_min * Y, YERR)

        # Don't believe in any of the YERRs
        if args.outliers_set_all:
            YERR = np.ones_like(Y) * args.outliers_set_all * Y

        # If there are less unique Xs than the number of parameters, skip
        if len(np.unique(X)) < nparams:
            return

        # If Xs do not span at least a factor of 2, skip
        if X.max() / X.min() < 2:
            return

        v0 = 10 ** ((np.log10(X.max()) + np.log10(X.min())) / 2)  # central frequency
        return Dataset(jname, X, Y, YERR, REF, v0)


def get_catalogue(args):
    with open('catalogue_fluxes_0202_good.pkl', 'rb') as f:
        cat_dict = pickle.load(f)
    with open('catalogue_fluxes_0202_citation_dict.pkl', 'rb') as f:
        citation_dict = pickle.load(f)

    return Catalogue(cat_dict, citation_dict)
