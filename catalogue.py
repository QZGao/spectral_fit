import os
import pickle
import re
from argparse import Namespace

import numpy as np
import pandas as pd
import psrqpy
from pulsar_spectra.catalogue import collect_catalogue_fluxes


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

    def dof(self, n_params: int) -> int:
        return len(self.Y) - n_params

    def yerr_y(self, ix=None):
        if ix is None:
            return self.YERR / self.Y
        else:
            return self.YERR[ix] / self.Y[ix]

    def max_yerr_y(self) -> float:
        return float(np.max(self.yerr_y()))

    def combined_err(self, sigma: float, ix=None, thresh: float = None):
        if thresh is None:
            if ix is None:
                return np.sqrt(self.YERR ** 2 + sigma ** 2 * self.Y ** 2)
            else:
                return np.sqrt(self.YERR[ix] ** 2 + sigma ** 2 * self.Y[ix] ** 2)
        else:
            if ix is None:
                return np.where(self.yerr_y() < thresh,
                                np.sqrt(self.YERR ** 2 + sigma ** 2 * self.Y ** 2),
                                self.YERR)
            else:
                return np.where(self.yerr_y(ix) < thresh,
                                np.sqrt(self.YERR[ix] ** 2 + sigma ** 2 * self.Y[ix] ** 2),
                                self.YERR[ix])


class Catalogue:
    def __init__(self, cat_dict: dict, citation_dict: dict):
        self.cat_dict = cat_dict
        self.citation_dict = citation_dict
        self.embeded_info = None

    def get_pulsar(self, jname: str, args: Namespace = None) -> Dataset | None:
        if jname not in self.cat_dict:
            return

        X = np.array(self.cat_dict[jname]['X'])
        Y = np.array(self.cat_dict[jname]['Y'])
        YERR = np.array(self.cat_dict[jname]['YERR'])
        REF = np.array(self.cat_dict[jname]['REF'])

        if args is not None:
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
                if args.outliers_min:
                    YERR = np.where(YERR / Y < args.outliers_min, args.outliers_min * Y, YERR)

            # Don't believe in any of the YERRs
            if args.outliers_all:
                YERR = np.ones_like(Y) * args.outliers_all * Y

        # Extreme case: no data, or after removing outliers, no data left
        if len(X) <= 0:
            return

        v0 = 10 ** ((np.log10(X.max()) + np.log10(X.min())) / 2)  # central frequency
        return Dataset(jname, X, Y, YERR, REF, v0)

    def filter(self, jname_list: list) -> 'Catalogue':
        cat_dict = {jname: self.cat_dict[jname] for jname in jname_list}
        return Catalogue(cat_dict, self.citation_dict)

    def at_least_n_points(self, n: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if len(data['X']) >= n]

    def __add__(self, other):
        cat_dict = self.cat_dict.copy()
        for jname, data in other.cat_dict.items():
            if jname in cat_dict:
                cat_dict[jname]['X'].extend(data['X'])
                cat_dict[jname]['Y'].extend(data['Y'])
                cat_dict[jname]['YERR'].extend(data['YERR'])
                cat_dict[jname]['REF'].extend(data['REF'])
            else:
                cat_dict[jname] = data

        citation_dict = self.citation_dict.copy()
        citation_dict.update(other.citation_dict)

        return Catalogue(cat_dict, citation_dict)

    def extend(self, jname: str, X: list, Y: list, YERR: list, REF: list):
        if jname in self.cat_dict:
            self.cat_dict[jname]['X'] = np.concatenate((self.cat_dict[jname]['X'], X))
            self.cat_dict[jname]['Y'] = np.concatenate((self.cat_dict[jname]['Y'], Y))
            self.cat_dict[jname]['YERR'] = np.concatenate((self.cat_dict[jname]['YERR'], YERR))
            self.cat_dict[jname]['REF'] = np.concatenate((self.cat_dict[jname]['REF'], REF))
        else:
            self.cat_dict[jname] = {
                'X': X,
                'Y': Y,
                'YERR': YERR,
                'REF': REF
            }

    def __len__(self):
        return len(self.cat_dict)


def collect_catalogue_from_ATNF(jname_list: list = None, atnf_ver: str = '1.54') -> Catalogue:
    print('Collecting data from ATNF Pulsar Catalogue via psrqpy...', end='')
    query_obj = psrqpy.QueryATNF(psrs=jname_list, version=atnf_ver)
    query = query_obj.pandas
    freqs = [key for key in query.keys() if re.fullmatch(r'S\d+G?', key) is not None]
    if jname_list is None:
        jname_list = list(query['PSRJ'])

    cat_dict = {}
    for jname in jname_list:
        if jname not in cat_dict:
            cat_dict[jname] = {
                'X': [],
                'Y': [],
                'YERR': [],
                'REF': []
            }

        query_id = list(query['PSRJ']).index(jname)
        for x_str in freqs:
            ref = query[f'{x_str}_REF'][query_id]
            y = query[x_str][query_id]
            if np.isnan(y):
                continue
            if f'{x_str}_ERR' not in query.keys():
                continue
            yerr = query[f'{x_str}_ERR'][query_id]
            if np.isnan(yerr) or yerr <= 0.:
                yerr = 0.5 * y  # Assume 50% error if not provided
            if x_str.endswith('G'):
                x = float(x_str[1:-1]) * 1e3  # GHz to MHz
            else:
                x = float(x_str[1:])

            cat_dict[jname]['X'].append(x)
            cat_dict[jname]['Y'].append(y)
            cat_dict[jname]['YERR'].append(yerr)
            cat_dict[jname]['REF'].append(ref)

    citation_dict = get_refs_from_ATNF(atnf_ver=atnf_ver)
    print(f' Done. (Version: {query_obj.get_version})')
    return Catalogue(cat_dict, citation_dict)


def get_refs_from_ATNF(atnf_ver: str = '1.54') -> dict:
    ref_dict = psrqpy.get_references(version=atnf_ver)  # Following version used by Jankowski et al. (2018)
    citation_dict = {}

    for ref_code, ref_str in ref_dict.items():
        ref_re = re.fullmatch(r'(.+)[.,] ?([1-2]{1}[0-9]{3})([a-z]?)\..*', ref_str)
        if ref_re is not None:
            ref_groups = ref_re.groups()
            ref_names = ref_groups[0].replace('&', ',')

            # Fix some names
            ref_names = re.sub('G"ou g"uc s', 'Göʇüş', ref_names)
            ref_names = re.sub('Ro.Zko', 'Rożko', ref_names)
            ref_names = re.sub('Tepedelenliov glu', 'Tepedelenlıoǧlu', ref_names)
            ref_names = re.sub('Tepedelenliv olu', 'Tepedelenlıoǧlu', ref_names)
            ref_names = re.sub(r"'([aeiou])", r'\1' + u'\u0301', ref_names)
            ref_names = re.sub(r'"([aeiouAEIOU])', r'\1' + u'\u0308', ref_names)

            # Deal with multiple authors
            ref_names = ref_names.split(',')
            ref_year = ref_groups[1] + ref_groups[2]
            if len(ref_names) < 4 or re.search(r'thesis', ref_groups[0], re.IGNORECASE) is not None:
                ref_name = f'{ref_names[0].strip()} ({ref_year})'
            elif 3 < len(ref_names) < 5:
                ref_name = f'{ref_names[0].strip()} & {ref_names[2].strip()} ({ref_year})'
            elif re.search(r'et al', ref_names[0]) is None:
                ref_name = f'{ref_names[0].strip()} et al. ({ref_year})'
            else:
                ref_name = f'{ref_names[0].strip()} ({ref_year})'

            # ATNF citation prefix
            citation_dict[ref_code] = 'ATNF: ' + ref_name
        else:
            citation_dict[ref_code] = 'ATNF: ' + ref_code

    return citation_dict


def collect_catalogue(custom_lit: list = None, jname_list: list = None) -> Catalogue:
    if custom_lit is None:
        custom_lit = DEFAULT_LITERATURE_SET

    print(f'Collecting data from {len(custom_lit)} literature sources via pulsar_spectra...', end='')
    pusp_cat_dict = collect_catalogue_fluxes(only_use=custom_lit, use_atnf=False)

    # Delete pulsars not in jname_list
    if jname_list is not None:
        pusp_cat_dict = {jname: data for jname, data in pusp_cat_dict.items() if jname in jname_list}

    # Get all references
    pusp_lit = []
    for _, v in pusp_cat_dict.items():
        pusp_lit.extend(v[4])
    pusp_lit = np.unique(pusp_lit)  # List of all references

    # Apply formatting
    cat_dict = {}
    for jname, (X, _, Y, YERR, REF) in pusp_cat_dict.items():
        # Check correctness of the data
        if (any([np.isnan(x) or x <= 0. for x in X]) or
                any([np.isnan(y) or y < 0. for y in Y]) or
                any([np.isnan(y) or y < 0. for y in YERR])):
            raise ValueError(f'Incorrect data for {jname}: X = {X}, Y = {Y}, YERR = {YERR}, REF = {REF}.')

        cat_dict[jname] = {
            'X': X,
            'Y': Y,
            'YERR': YERR,
            'REF': REF,
        }

    # Citations
    citation_dict = DEFAULT_LITERATURE_CITATIONS
    # Deal with literature not in citation_dict
    for ref in pusp_lit:
        if ref not in citation_dict:
            re_groups = re.fullmatch(r'(.+)_([0-9]{4})[a-z]?', ref)
            if re_groups is not None:
                author = re_groups.group(1).replace('_', ' ')
                year = re_groups.group(2)
                citation_dict[ref] = f'{author} et al. ({year})'
            else:  # Not in the standard format
                citation_dict[ref] = ref

    print(' Done.')
    cat = Catalogue(cat_dict, citation_dict)

    # The following are patches to the catalogue in pulsar_spectra v2.0.4:

    # Add data from Posselt et al. (2023)
    if 'Posselt_2023' in custom_lit:
        posselt_df = pd.read_csv('catalogue/Posselt_2023_data.csv')
        for _, row in posselt_df.iterrows():
            jname = row['PSRJ']
            x, y, yerr, ref = [], [], [], []
            for i in range(1, 9):
                if not pd.isna(row[f'ch{i}freq']):
                    x.append(float(row[f'ch{i}freq']))
                    y.append(float(row[f'ch{i}flux']))
                    yerr.append(float(row[f'ch{i}errflux']))
                    ref.append('Posselt_2023')
            yerr = [0.5 * y[i] if yerr[i] == 0. else yerr[i] for i in range(len(y))]
            cat.extend(jname, x, y, yerr, ref)
        print('Posselt_2023: Manually added measurements of 8 frequency channels from Posselt et al. (2023).')

    # Add data from Spiewak et al. (2022)
    # pulsar_spectra uses the data from Table 1 of the paper, but we can use the supporting data published by the author
    # https://github.com/NickSwainston/pulsar_spectra/pull/56/commits/824371ddcb3190ff1ddfd7bd22e07d0d39db3544
    if 'Spiewak_2022' in custom_lit:
        spiewak_df = pd.read_csv('catalogue/Spiewak_2022_data.csv')
        for _, row in spiewak_df.iterrows():
            jname = row['PSRJ']
            if jname in cat.cat_dict and 'Spiewak_2022' in cat.cat_dict[jname]['REF']:
                # Remove the existing mean flux data (which only has one mean value for each pulsar)
                idx = np.where(np.array(cat.cat_dict[jname]['REF']) == 'Spiewak_2022')
                cat.cat_dict[jname]['X'] = np.delete(cat.cat_dict[jname]['X'], idx)
                cat.cat_dict[jname]['Y'] = np.delete(cat.cat_dict[jname]['Y'], idx)
                cat.cat_dict[jname]['YERR'] = np.delete(cat.cat_dict[jname]['YERR'], idx)
                cat.cat_dict[jname]['REF'] = np.delete(cat.cat_dict[jname]['REF'], idx)
            x, y, yerr, ref = [], [], [], []
            for i in range(0, 8):
                if not pd.isna(row[f'FLUX_{i}_FRQ']):
                    x.append(float(row[f'FLUX_{i}_FRQ']))
                    y.append(float(row[f'FLUX_{i}_VAL']))
                    yerr.append(float(row[f'FLUX_{i}_ERR']))
                    ref.append('Spiewak_2022')
            yerr = [0.5 * y[i] if yerr[i] == 0. else yerr[i] for i in range(len(y))]
            cat.extend(jname, x, y, yerr, ref)
        print('Spiewak_2022: Manually added measurements of 8 frequency channels from Spiewak et al. (2022).')

    # Remove incorrect data from Johnston & Kerr (2018)
    # https://github.com/NickSwainston/pulsar_spectra/issues/93
    if 'J1842-0359' in cat.cat_dict and 'Johnston_2018' in cat.cat_dict['J1842-0359']['REF']:
        ix = np.where(np.array(cat.cat_dict['J1842-0359']['REF']) == 'Johnston_2018')
        cat.cat_dict['J1842-0359']['X'] = np.delete(cat.cat_dict['J1842-0359']['X'], ix)
        cat.cat_dict['J1842-0359']['Y'] = np.delete(cat.cat_dict['J1842-0359']['Y'], ix)
        cat.cat_dict['J1842-0359']['YERR'] = np.delete(cat.cat_dict['J1842-0359']['YERR'], ix)
        cat.cat_dict['J1842-0359']['REF'] = np.delete(cat.cat_dict['J1842-0359']['REF'], ix)
        print('Johnston_2018: Removed incorrect data for J1842-0359 (0.01 mJy, class "N") from Johnston & Kerr (2018).')

    return cat


def get_catalogue(args: Namespace = None) -> Catalogue:
    if args is None:
        args = Namespace(jan_set=False, lit_set=None, atnf=False, atnf_ver='1.54')

    # Reproduce Jankowski et al. (2018)'s dataset
    if args.jan_set:
        # Load catalogue from pickle
        if os.path.exists('catalogue/catalogue_jan.pkl'):
            with open('catalogue/catalogue_jan.pkl', 'rb') as f:
                return pickle.load(f)

        with open('catalogue/Jankowski_2018_data.csv', 'r', encoding='utf-8-sig') as f:
            jan_df = pd.read_csv(f)
        jan_jnames = jan_df['PSRJ'].values.tolist()
        cat = (collect_catalogue_from_ATNF(jname_list=jan_jnames, atnf_ver=args.atnf_ver) +
               collect_catalogue(custom_lit=JANKOWSKI_LITERATURE_SET, jname_list=jan_jnames))

        # Save to pickle
        with open('catalogue/catalogue_jan.pkl', 'wb') as f:
            pickle.dump(cat, f)
        return cat

    # Load catalogue from pickle
    if os.path.exists('catalogue/catalogue.pkl'):
        with open('catalogue/catalogue.pkl', 'rb') as f:
            cat = pickle.load(f)
        if args.lit_set == cat.embeded_info['lit_set'] and args.atnf == cat.embeded_info['atnf']:
            return cat

    if args.lit_set is not None:  # Custom literature set
        catalogue = collect_catalogue(custom_lit=args.lit_set.split(';'))
    else:
        catalogue = collect_catalogue()

    if args.atnf:  # ATNF catalogue
        cat = collect_catalogue_from_ATNF(atnf_ver=args.atnf_ver)
        catalogue = cat if catalogue is None else catalogue + cat

    catalogue.embeded_info = {
        'lit_set': args.lit_set,
        'atnf': args.atnf
    }

    # Save to pickle
    with open('catalogue/catalogue.pkl', 'wb') as f:
        pickle.dump(catalogue, f)
    return catalogue


DEFAULT_LITERATURE_SET = [
    'McLean_1973',
    'Bartel_1978',
    'Manchester_1978a',
    'Izvekova_1981',
    'Dewey_1985',
    'Stokes_1985',
    'Stokes_1986',
    'Wolszczan_1992',
    'Malofeev_1993',
    'Camilo_1995',
    'Lorimer_1995b',
    'Manchester_1995',
    'Qiao_1995',
    'Seiradakis_1995',
    'Biggs_1996',
    'Manchester_1996',
    'Hoensbroech_1997',
    'Kramer_1997',
    'van_Ommen_1997',
    'Kijak_1998',
    'Kramer_1998',
    'Shrauner_1998',
    'Toscano_1998',
    'Han_1999',
    'Kramer_1999',
    'Stairs_1999',
    'Weisberg_1999',
    'Lommen_2000',
    'Malofeev_2000',
    'Giacani_2001',
    'Kuzmin_2001',
    'Manchester_2001',
    'McGary_2001',
    'Morris_2002',
    'Esamdin_2004',
    'Hobbs_2004a',
    'Champion_2005a',
    'Champion_2005b',
    'Karastergiou_2005',
    'Lorimer_2005',
    'Johnston_2006',
    'Crawford_2007',
    'Kijak_2007',
    'Deller_2009',
    'Joshi_2009',
    'Keith_2011',
    'Kijak_2011',
    'Lynch_2012',
    'Boyles_2013',
    'Demorest_2013',
    'Dowell_2013',
    'Manchester_2013',
    'Zakharenko_2013',
    'Dembska_2014',
    'Stovall_2014',
    'Dai_2015',
    'Dembska_2015',
    'Lazarus_2015',
    'Ng_2015',
    'Basu_2016',
    'Bell_2016',
    'Bhattacharyya_2016',
    'Bilous_2016',
    'Frail_2016',
    'Han_2016',
    'Kondratiev_2016',
    'Kijak_2017',
    'Mignani_2017',
    'Murphy_2017',
    'Xue_2017',
    'Zhao_2017',
    'Basu_2018',
    'Brinkman_2018',
    'Gentile_2018',
    'Jankowski_2018',
    'Johnston_2018',
    'RoZko_2018',
    'Jankowski_2019',
    'Kaur_2019',
    'Sanidas_2019',
    'Surnis_2019',
    'Xie_2019',
    'Zhang_2019',
    'Zhao_2019',
    'Bilous_2020',
    'Bondonneau_2020',
    'Crowter_2020',
    'Curylo_2020',
    'Michilli_2020',
    'Tan_2020',
    'Alam_2021',
    'Bondonneau_2021',
    'Han_2021',
    'Johnston_2021',
    'Shapiro_Albert_2021',
    'Lee_2022',
    'Spiewak_2022',
    'Bhat_2023',
    'Gitika_2023',
    'Posselt_2023'
]
DEFAULT_LITERATURE_CITATIONS = {
    'McLean_1973': 'McLean (1973)',
    'Bartel_1978': 'Bartel et al. (1978)',
    'Manchester_1978a': 'Manchester et al. (1978)',
    'Izvekova_1981': 'Izvekova et al. (1981)',
    'Dewey_1985': 'Dewey et al. (1985)',
    'Stokes_1985': 'Stokes et al. (1985)',
    'Stokes_1986': 'Stokes et al. (1986)',
    'McConnell_1991': 'McConnell et al. (1991)',
    'Johnston_1992': 'Johnston et al. (1992)',
    'Wolszczan_1992': 'Wolszczan & Frail (1992)',
    'Johnston_1993': 'Johnston et al. (1993)',
    'Malofeev_1993': 'Malofeev (1993)',
    'Manchester_1993': 'Manchester et al. (1993)',
    'Camilo_1995': 'Camilo & Nice (1995)',
    'Lorimer_1995': 'Lorimer et al. (1995)',
    'Lorimer_1995b': 'Lorimer et al. (1995)',
    'Lundgren_1995': 'Lundgren et al. (1995)',
    'Manchester_1995': 'Manchester & Johnston (1995)',
    'Nicastro_1995': 'Nicastro et al. (1995)',
    'Qiao_1995': 'Qiao et al. (1995)',
    'Robinson_1995': 'Robinson et al. (1995)',
    'Seiradakis_1995': 'Seiradakis et al. (1995)',
    'Biggs_1996': 'Biggs & Lyne (1996)',
    'Lorimer_1996': 'Lorimer et al. (1996)',
    'Manchester_1996': 'Manchester et al. (1996)',
    'Zepka_1996': 'Zepka et al. (1996)',
    'Bailes_1997': 'Bailes et al. (1997)',
    'Hoensbroech_1997': 'von Hoensbroech & Xilouris (1997)',
    'Kaspi_1997': 'Kaspi et al. (1997)',
    'Kramer_1997': 'Kramer et al. (1997)',
    'Sayer_1997': 'Sayer et al. (1997)',
    'van_Ommen_1997': 'van Ommen et al. (1997)',
    'Kijak_1998': 'Kijak et al. (1998)',
    'Kramer_1998': 'Kramer et al. (1998)',
    'Shrauner_1998': 'Shrauner et al. (1998)',
    'Toscano_1998': 'Toscano et al. (1998)',
    'Han_1999': 'Han & Tian (1999)',
    'Kramer_1999': 'Kramer et al. (1999)',
    'Stairs_1999': 'Stairs et al. (1999)',
    'Weisberg_1999': 'Weisberg et al. (1999)',
    'Kouwenhoven_2000': 'Kouwenhoven (2000)',
    'Lommen_2000': 'Lommen et al. (2000)',
    'Malofeev_2000': 'Malofeev et al. (2000)',
    'Crawford_2001': 'Crawford et al. (2001)',
    'Giacani_2001': 'Giacani et al. (2001)',
    'Kuzmin_2001': 'Kuzmin & Losovsky (2001)',
    'Manchester_2001': 'Manchester et al. (2001)',
    'McGary_2001': 'McGary et al. (2001)',
    'Morris_2002': 'Morris et al. (2002)',
    'Kramer_2003a': 'Kramer et al. (2003)',
    'Esamdin_2004': 'Esamdin et al. (2004)',
    'Hobbs_2004a': 'Hobbs et al. (2004)',
    'Lewandowski_2004': 'Lewandowski et al. (2004)',
    'Champion_2005a': 'Champion et al. (2005)',
    'Champion_2005b': 'Champion (2005)',
    'Karastergiou_2005': 'Karastergiou et al. (2005)',
    'Lorimer_2005': 'Lorimer et al. (2005)',
    'Johnston_2006': 'Johnston et al. (2006)',
    'Lorimer_2006': 'Lorimer et al. (2006)',
    'Crawford_2007': 'Crawford & Tiffany (2007)',
    'Freire_2007': 'Freire et al. (2007)',
    'Kijak_2007': 'Kijak et al. (2007)',
    'Lorimer_2007': 'Lorimer et al. (2007)',
    'Stappers_2008': 'Stappers et al. (2008)',
    'Deller_2009': 'Deller et al. (2009)',
    'Janssen_2009': 'Janssen et al. (2009)',
    'Joshi_2009': 'Joshi et al. (2009)',
    'Bates_2011': 'Bates et al. (2011)',
    'Keith_2011': 'Keith et al. (2011)',
    'Kijak_2011': 'Kijak et al. (2011)',
    'Lynch_2012': 'Lynch et al. (2012)',
    'Mickaliger_2012': 'Mickaliger et al. (2012)',
    'Boyles_2013': 'Boyles et al. (2013)',
    'Demorest_2013': 'Demorest et al. (2013)',
    'Dowell_2013': 'Dowell et al. (2013)',
    'Lynch_2013': 'Lynch et al. (2013)',
    'Manchester_2013': 'Manchester et al. (2013)',
    'Zakharenko_2013': 'Zakharenko et al. (2013)',
    'Dembska_2014': 'Dembska et al. (2014)',
    'Stovall_2014': 'Stovall et al. (2014)',
    'Dai_2015': 'Dai et al. (2015)',
    'Dembska_2015': 'Dembska et al. (2015)',
    'Kuniyoshi_2015': 'Kuniyoshi et al. (2015)',
    'Lazarus_2015': 'Lazarus et al. (2015)',
    'Ng_2015': 'Ng et al. (2015)',
    'Stovall_2015': 'Stovall et al. (2015)',
    'Basu_2016': 'Basu et al. (2016)',
    'Bell_2016': 'Bell et al. (2016)',
    'Bhattacharyya_2016': 'Bhattacharyya et al. (2016)',
    'Bilous_2016': 'Bilous et al. (2016)',
    'Frail_2016': 'Frail et al. (2016)',
    'Han_2016': 'Han et al. (2016)',
    'Kondratiev_2016': 'Kondratiev et al. (2016)',
    'Mikhailov_2016': 'Mikhailov & van Leeuwen (2016)',
    'Kijak_2017': 'Kijak et al. (2017)',
    'Mignani_2017': 'Mignani et al. (2017)',
    'Murphy_2017': 'Murphy et al. (2017)',
    'Xue_2017': 'Xue et al. (2017)',
    'Zhao_2017': 'Zhao et al. (2017)',
    'Basu_2018': 'Basu et al. (2018)',
    'Brinkman_2018': 'Brinkman et al. (2018)',
    'Gentile_2018': 'Brook et al. (2018)',
    'Jankowski_2018': 'Jankowski et al. (2018)',
    'Johnston_2018': 'Johnston & Kerr (2018)',
    'RoZko_2018': 'Rożko et al. (2018)',
    'Aloisi_2019': 'Aloisi et al. (2019)',
    'Jankowski_2019': 'Jankowski et al. (2019)',
    'Kaur_2019': 'Kaur et al. (2019)',
    'Sanidas_2019': 'Sanidas et al. (2019)',
    'Surnis_2019': 'Surnis et al. (2019)',
    'Titus_2019': 'Titus et al. (2019)',
    'Xie_2019': 'Xie et al. (2019)',
    'Zhang_2019': 'Zhang et al. (2019)',
    'Zhao_2019': 'Zhao et al. (2019)',
    'Bilous_2020': 'Bilous et al. (2020)',
    'Bondonneau_2020': 'Bondonneau et al. (2020)',
    'Crowter_2020': 'Crowter et al. (2020)',
    'Curylo_2020': 'Curyło et al. (2020)',
    'McEwen_2020': 'McEwen et al. (2020)',
    'Michilli_2020': 'Michilli et al. (2020)',
    'Tan_2020': 'Tan et al. (2020)',
    'Alam_2021': 'Alam et al. (2021)',
    'Bondonneau_2021': 'Bondonneau et al. (2021)',
    'Han_2021': 'Han et al. (2021)',
    'Johnston_2021': 'Johnston et al. (2021)',
    'Shapiro_Albert_2021': 'Shapiro-Albert et al. (2021)',
    'Kravtsov_2022': 'Kravtsov et al. (2022)',
    'Lee_2022': 'Lee et al. (2022)',
    'Spiewak_2022': 'Spiewak et al. (2022)',
    'Bhat_2023': 'Bhat et al. (2023)',
    'Gitika_2023': 'Gitika et al. (2023)',
    'Posselt_2023': 'Posselt et al. (2023)'
}
JANKOWSKI_LITERATURE_SET = [  # Part of literature used by Jankowski et al. (2018)
    'Bartel_1978',
    'Izvekova_1981',
    'Lorimer_1995b',
    'van_Ommen_1997',
    'Malofeev_2000',
    'Karastergiou_2005',
    'Johnston_2006',
    'Kijak_2007',
    'Bates_2011',
    'Keith_2011',
    'Zakharenko_2013',
    'Dai_2015',
    'Basu_2016',
    'Bell_2016',
    'Bilous_2016',
    'Han_2016',
    'Kijak_2017',
    'Murphy_2017',
    'Jankowski_2018'
]
