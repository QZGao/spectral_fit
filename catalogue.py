import os
import pickle
import re
from argparse import Namespace
from pathlib import Path

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

    def __copy__(self):
        return Dataset(self.jname, self.X.copy(), self.Y.copy(), self.YERR.copy(), self.REF.copy(), self.v0)

    def copy(self):
        return self.__copy__()

    @property
    def len(self):
        return len(self.X)

    def __len__(self):
        return self.len

    def __str__(self):
        return (f'{self.jname}: {self.len} measurements\n'
                f'X = [{", ".join(map(str, self.X))}] MHz\n'
                f'Y = [{", ".join(map(str, self.Y))}] mJy\n'
                f'YERR = [{", ".join(map(str, self.YERR))}] mJy\n'
                f'REF = [{", ".join(self.REF)}]\n'
                f'v0 = {self.v0} MHz')

    def dof(self, n_params: int) -> int:
        return self.len - n_params

    def yerr_y(self, ix=None):
        if ix is None:
            return self.YERR / self.Y
        else:
            return self.YERR[ix] / self.Y[ix]

    @property
    def max_yerr_y(self) -> float:
        return float(np.max(self.yerr_y()))

    @property
    def min_yerr_y(self) -> float:
        return float(np.min(self.yerr_y()))

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

    @property
    def peak_frequency(self) -> float:
        return self.X[np.argmax(self.Y)]


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
                    if args.outliers_min_plus:
                        YERR = np.where(YERR / Y < args.outliers_min, np.sqrt(Y ** 2 * args.outliers_min_plus ** 2 + YERR ** 2), YERR)
                    else:
                        YERR = np.where(YERR / Y < args.outliers_min, args.outliers_min * Y, YERR)

            # Don't believe in any of the YERRs
            if args.outliers_all:
                YERR = np.ones_like(Y) * args.outliers_all * Y

        # Extreme case: no data, or after removing outliers, no data left
        if len(X) <= 0:
            return

        v0 = 10 ** ((np.log10(X.max()) + np.log10(X.min())) / 2)  # central frequency
        return Dataset(jname, X, Y, YERR, REF, v0)

    def __getitem__(self, item):
        return self.get_pulsar(item)

    def by_jname_ref(self):
        cat_dict_mod = {}
        for jname, data in self.cat_dict.items():
            cat_dict_mod[jname] = {}
            for i in range(len(data['X'])):
                x, y, yerr, ref = data['X'][i], data['Y'][i], data['YERR'][i], data['REF'][i]
                if ref not in cat_dict_mod[jname]:
                    cat_dict_mod[jname][ref] = {
                        'X': [],
                        'Y': [],
                        'YERR': []
                    }
                cat_dict_mod[jname][ref]['X'].append(x)
                cat_dict_mod[jname][ref]['Y'].append(y)
                cat_dict_mod[jname][ref]['YERR'].append(yerr)
        return cat_dict_mod

    def by_ref_jname(self):
        cat_dict_mod = {}
        for jname, data in self.cat_dict.items():
            for i in range(len(data['X'])):
                x, y, yerr, ref = data['X'][i], data['Y'][i], data['YERR'][i], data['REF'][i]
                if ref not in cat_dict_mod:
                    cat_dict_mod[ref] = {}
                if jname not in cat_dict_mod[ref]:
                    cat_dict_mod[ref][jname] = {
                        'X': [],
                        'Y': [],
                        'YERR': []
                    }
                cat_dict_mod[ref][jname]['X'].append(x)
                cat_dict_mod[ref][jname]['Y'].append(y)
                cat_dict_mod[ref][jname]['YERR'].append(yerr)
        return cat_dict_mod

    def is_MSP(self, jname: str) -> bool:
        if jname not in self.cat_dict:
            return False
        if 'P' not in self.cat_dict[jname]:
            return False
        # P < 70 ms and P_DOT < 1e-17
        return self.cat_dict[jname]['P'] < 0.07 and self.cat_dict[jname]['P_DOT'] < 1e-17

    @property
    def MSPs(self) -> list[str]:
        return [jname for jname in self.cat_dict if self.is_MSP(jname)]

    def filter(self, jname_list: list) -> 'Catalogue':
        cat_dict = {jname: self.cat_dict[jname] for jname in jname_list if jname in self.cat_dict}
        return Catalogue(cat_dict, self.citation_dict)

    def more_than_n_points(self, n: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if len(data['X']) > n]

    def less_than_n_points(self, n: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if len(data['X']) < n]

    def exactly_n_points(self, n: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if len(data['X']) == n]

    def at_least_n_points(self, n: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if len(data['X']) >= n]

    def at_most_n_points(self, n: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if len(data['X']) <= n]

    def range_points(self, i: int, j: int) -> list[str]:
        return [jname for jname, data in self.cat_dict.items() if i <= len(data['X']) <= j]

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

    def extend(self, jname: str, X: float | list, Y: float | list, YERR: float | list, REF: str | list):
        if isinstance(X, float):
            X = [X]
        if isinstance(Y, float):
            Y = [Y]
        if isinstance(YERR, float):
            YERR = [YERR]
        if isinstance(REF, str):
            REF = [REF]
        if len(X) != len(Y) or len(Y) != len(YERR) or len(YERR) != len(REF):
            raise ValueError('Lengths of X, Y, YERR, and REF must be the same.')
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

    def remove_refs(self, refs: str | list):
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            for jname, data in self.cat_dict.items():
                X, Y, YERR, REF = [], [], [], []
                for i in range(len(data['X'])):
                    if data['REF'][i] != ref:
                        X.append(data['X'][i])
                        Y.append(data['Y'][i])
                        YERR.append(data['YERR'][i])
                        REF.append(data['REF'][i])
                self.cat_dict[jname] = {
                    'X': X,
                    'Y': Y,
                    'YERR': YERR,
                    'REF': REF
                }

    def remove_refs_from_pulsar(self, jname: str, refs: str | list):
        if isinstance(refs, str):
            refs = [refs]
        if jname in self.cat_dict:
            for ref in refs:
                data = self.cat_dict[jname]
                X, Y, YERR, REF = [], [], [], []
                for i in range(len(data['X'])):
                    if data['REF'][i] != ref:
                        X.append(data['X'][i])
                        Y.append(data['Y'][i])
                        YERR.append(data['YERR'][i])
                        REF.append(data['REF'][i])
                self.cat_dict[jname] = {
                    'X': X,
                    'Y': Y,
                    'YERR': YERR,
                    'REF': REF
                }

    def add_citations(self, citation_dict: dict):
        self.citation_dict.update(citation_dict)

    @property
    def len(self):
        return len(self.cat_dict)

    def __len__(self):
        return self.len

    def cleanup(self):
        """Remove pulsars with no data."""
        cat_dict = {}
        for jname, data in self.cat_dict.items():
            if len(data['X']) > 0:
                cat_dict[jname] = data
        self.cat_dict = cat_dict

    def print_lit(self, outdir: str):
        print(f'Pulsar count: {len(self.cat_dict)}')

        # name, pulsar_count, frequency_range
        lit_dict = {}
        for jname, data in self.cat_dict.items():
            ref_in_pulsar = []
            for (x, y, yerr, ref) in zip(data['X'], data['Y'], data['YERR'], data['REF']):
                if ref not in self.citation_dict or self.citation_dict[ref].startswith('ATNF'):
                    ref = 'ATNF'
                    citation = '1000'
                else:
                    citation = self.citation_dict[ref]
                if ref not in lit_dict:
                    lit_dict[ref] = {
                        'Citation': citation,
                        'Pulsar count': 0,
                        'Frequency range (MHz)': [np.inf, 0.]
                    }
                ref_in_pulsar.append(ref)
                lit_dict[ref]['Frequency range (MHz)'][0] = min(lit_dict[ref]['Frequency range (MHz)'][0], x)
                lit_dict[ref]['Frequency range (MHz)'][1] = max(lit_dict[ref]['Frequency range (MHz)'][1], x)

            ref_in_pulsar = np.unique(ref_in_pulsar)
            for ref in ref_in_pulsar:
                lit_dict[ref]['Pulsar count'] += 1

        lit_df = pd.DataFrame(lit_dict).T
        lit_df['Frequency range (MHz)'] = lit_df['Frequency range (MHz)'].apply(lambda x: f'{x[0]:.0f}-{x[1]:.0f}')

        # Sort by year
        lit_df['Year'] = lit_df['Citation'].apply(lambda ref: int(re.search(r'([1-2]{1}[0-9]{3})', ref).group(0)))
        lit_df = lit_df.sort_values('Year').drop(columns='Year')
        lit_df['Citation'] = lit_df['Citation'].apply(lambda x: 'The ATNF Pulsar Catalogue' if x == '1000' else x)

        # Save to CSV
        lit_df.to_csv(f'{outdir}/literature.csv', index_label='Citekey')
        print(f'Literature list saved to {outdir}/literature.csv.')

    def apply_efrac_to_ref(self, ref: str, efrac: float):
        for jname, data in self.cat_dict.items():
            for i in range(len(data['X'])):
                if data['REF'][i] == ref:
                    data['YERR'][i] = np.sqrt(data['Y'][i] ** 2 * efrac ** 2 + data['YERR'][i] ** 2)

    def apply_fixes(self, lit_set: list):
        """Patches to the catalogue in pulsar_spectra v2.0.4"""

        # Add data from Sieber (1973)
        # https://github.com/NickSwainston/pulsar_spectra/commit/1793a21607eed71a4556701517ccd14f6193b6f4
        # Extracted from Table 1 of the paper, flux densities are calculated by Swainston
        if 'Sieber_1973' in lit_set:
            sieber_df = pd.read_csv('catalogue/Sieber_1973_data.csv')
            for _, row in sieber_df.iterrows():
                self.extend(row['PSRJ'], float(row['FREQ']), float(row['FLUX']), float(row['FLUX_ERR']), 'Sieber_1973')
            print('Sieber_1973: Manually added measurements from Sieber (1973).')

        # Add data from Maron et al. (2000)
        # https://web.archive.org/web/20100106132709/http://astro.ia.uz.zgora.pl/olaf/paper1/
        # Extracted from a PostScript file named "table2.ps" in the "tables.tar.gz".
        if 'Maron_2000' in lit_set:
            maron_df = pd.read_csv('catalogue/Maron_2000_data.csv')
            for _, row in maron_df.iterrows():
                self.extend(
                    row['PSRJ'], X=float(row['FREQ']), Y=float(row['FLUX']), REF='Maron_2000',
                    YERR=float(row['FLUX_ERR']) if float(row['FLUX_ERR']) > 0. else 0.5 * float(row['FLUX'])
                )
            print('Maron_2000: Manually added measurements from Maron et al. (2000).')

        # Kouwenhoven et al. (2000) data is already in the catalogue, but the values are incorrect
        # We replace the values with the correct ones from Table 2-4 of the paper
        if 'Kouwenhoven_2000' in lit_set:
            kouwenhoven_df = pd.read_csv('catalogue/Kouwenhoven_2000_data.csv')
            self.remove_refs('Kouwenhoven_2000')
            for _, row in kouwenhoven_df.iterrows():
                self.extend(row['PSRJ'], 325., float(row['S325']), np.sqrt(float(row['E1_S325'])**2 + float(row['E2_S325'])**2), 'Kouwenhoven_2000')
            print('Kouwenhoven_2000: Corrected measurements from Kouwenhoven et al. (2000).')

        # Remove incorrect data from Johnston & Kerr (2018)
        # https://github.com/NickSwainston/pulsar_spectra/issues/93
        if 'J1842-0359' in self.cat_dict and 'Johnston_2018' in self.cat_dict['J1842-0359']['REF']:
            self.remove_refs_from_pulsar('J1842-0359', 'Johnston_2018')
            print('Johnston_2018: Removed incorrect data for J1842-0359 (0.01 mJy, class "N") from Johnston & Kerr (2018).')

        # Add data from Spiewak et al. (2022)
        # pulsar_spectra uses the data from Table 1 of the paper, but we can use the supporting data published by the author
        # https://github.com/NickSwainston/pulsar_spectra/pull/56/commits/824371ddcb3190ff1ddfd7bd22e07d0d39db3544
        if 'Spiewak_2022' in lit_set:
            spiewak_df = pd.read_csv('catalogue/Spiewak_2022_data.csv')
            self.remove_refs('Spiewak_2022')
            for _, row in spiewak_df.iterrows():
                jname = row['PSRJ']
                x, y, yerr, ref = [], [], [], []
                for i in range(0, 8):
                    if not pd.isna(row[f'FLUX_{i}_FRQ']):
                        x.append(float(row[f'FLUX_{i}_FRQ']))
                        y.append(float(row[f'FLUX_{i}_VAL']))
                        yerr.append(float(row[f'FLUX_{i}_ERR']))
                        ref.append('Spiewak_2022')
                yerr = [0.5 * y[i] if yerr[i] == 0. else yerr[i] for i in range(len(y))]
                self.extend(jname, x, y, yerr, ref)
            print('Spiewak_2022: Manually added measurements of 8 frequency channels from Spiewak et al. (2022).')

        # Add data from Posselt et al. (2023)
        # Extracted from the supporting data published by the author
        if 'Posselt_2023' in lit_set:
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
                self.extend(jname, x, y, yerr, ref)
            print('Posselt_2023: Manually added measurements of 8 frequency channels from Posselt et al. (2023).')

        # Add data from Anumarlapudi et al. (2023) and Gordon et al. (2021)
        # Extracted from Table B1 of the former paper
        if 'Anumarlapudi_2023' in lit_set:
            anumarlapudi_df = pd.read_csv('catalogue/Anumarlapudi_2023_data.csv')
            for _, row in anumarlapudi_df.iterrows():
                jname = row['Name']
                x = [888.]
                y = [float(row['S888'])]
                yerr = [float(row['e_S888'])]
                ref = ['Anumarlapudi_2023']
                if not np.isnan(row['S3000']):
                    x.append(3000.)
                    y.append(float(row['S3000']))
                    yerr.append(float(row['e_S3000']))
                    ref.append('Gordon_2021')
                self.extend(jname, x, y, yerr, ref)
            print('Anumarlapudi_2023 & Gordon_2021: Manually added measurements of 8 frequency channels from Anumarlapudi et al. (2023) and Gordon et al. (2021).')

        # Apply added fractional error based on the declarement of the authors
        # efrac_df = pd.read_csv('catalogue/efrac.csv')
        # for _, row in efrac_df.iterrows():
        #     if not np.isnan(float(row['EFRAC'])):
        #         self.apply_efrac_to_ref(row['REF'], float(row['EFRAC']))
            # print(f'{row["REF"]}: Applied additional fractional error of {row["EFRAC"]} to the flux densities.')


    def keys(self):
        return self.cat_dict.keys()


    def items(self):
        return self.cat_dict.items()


def collect_catalogue_from_ATNF(jname_list: list = None, atnf_ver: str = '1.54', p_only: bool = False) -> Catalogue:
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
                'REF': [],
            }

        query_id = list(query['PSRJ']).index(jname)
        P = query['P0'][query_id]
        P_dot = query['P1'][query_id]
        if not np.isnan(P) and not np.isnan(P_dot):
            cat_dict[jname]['P'] = P
            cat_dict[jname]['P_DOT'] = P_dot

        if p_only:
            continue
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

        if len(cat_dict[jname]['X']) == 0:
            # No flux density measurements available in the ATNF Pulsar Catalogue
            del cat_dict[jname]

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
    cat.apply_fixes(lit_set=custom_lit)

    if jname_list is not None:  # Filter by pulsar names
        for jname in jname_list:
            if jname not in cat.cat_dict:
                print(f'Warning: {jname} not found in the catalogue.')
        cat = cat.filter(jname_list)

    return cat


def load_catalogue(outdir: str = '') -> Catalogue:
    if os.getcwd().endswith('notebooks'):
        outdir = f'../output/{outdir}'
    if not os.path.exists(f'{outdir}/catalogue.pkl'):
        raise FileNotFoundError(f'{outdir}/catalogue.pkl')
    with open(f'{outdir}/catalogue.pkl', 'rb') as f:
        cat = pickle.load(f)
    return cat


def get_catalogue(outdir: str = None, refresh: bool = False, lit_set: str | list = None,
                  atnf: bool = False, atnf_ver: str = '1.54', jan_set: bool = False) -> Catalogue:
    if os.getcwd().endswith('notebooks'):
        os.chdir('..')

    # Load catalogue from pickle
    if (outdir is not None) and (not refresh) and os.path.exists(f'{outdir}/catalogue.pkl'):
        with open(f'{outdir}/catalogue.pkl', 'rb') as f:
            cat = pickle.load(f)
        if lit_set == cat.embeded_info['lit_set'] and atnf == cat.embeded_info['atnf'] and \
                atnf_ver == cat.embeded_info['atnf_ver'] and jan_set == cat.embeded_info['jan_set']:
            return cat

    # Reproduce Jankowski et al. (2018)'s dataset
    if jan_set:
        with open('catalogue/Jankowski_2018_data.csv', 'r', encoding='utf-8-sig') as f:
            jan_df = pd.read_csv(f)
        jan_jnames = jan_df['PSRJ'].values.tolist()
        catalogue = (collect_catalogue_from_ATNF(jname_list=jan_jnames, atnf_ver=atnf_ver) +
               collect_catalogue(custom_lit=JANKOWSKI_LITERATURE_SET, jname_list=jan_jnames))

    else:
        catalogue = collect_catalogue_from_ATNF(atnf_ver=atnf_ver, p_only=not atnf)
        if lit_set is not None:  # Custom literature set
            if lit_set == 'IMAGING_SURVEYS':
                catalogue += collect_catalogue(custom_lit=IMAGING_SURVEYS_LITERATURE_SET)
            else:
                if isinstance(lit_set, str):
                    lit_set = lit_set.split(';')
                catalogue += collect_catalogue(custom_lit=lit_set)
        else:
            catalogue += collect_catalogue()

    catalogue.cleanup()
    catalogue.embeded_info = {
        'lit_set': lit_set,
        'atnf': atnf, 'atnf_ver': atnf_ver,
        'jan_set': jan_set
    }

    # Save to pickle
    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with open(f'{outdir}/catalogue.pkl', 'wb') as f:
            pickle.dump(catalogue, f)
    return catalogue


def get_catalogue_from_args(args: Namespace) -> Catalogue:
    return get_catalogue(
        outdir=args.outdir, refresh=args.refresh,
        lit_set=args.lit_set,
        atnf=args.atnf, atnf_ver=args.atnf_ver,
        jan_set=args.jan_set
    )


DEFAULT_LITERATURE_SET = [
    'McLean_1973',
    'Sieber_1973',  # Added
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
    'Maron_2000',  # Added
    'Malofeev_2000',
    'Kouwenhoven_2000',  # Replaced
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
    'Johnston_2018',  # Modified
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
    'Gordon_2021',  # Added
    'Lee_2022',
    'Spiewak_2022',  # Replaced
    'Bhat_2023',
    'Gitika_2023',
    'Posselt_2023',  # Added
    'Anumarlapudi_2023'  # Added
]
IMAGING_SURVEYS_LITERATURE_SET = [
    'Kouwenhoven_2000',
    'McGary_2001',
    'Bilous_2016',
    'Kondratiev_2016',
    'Frail_2016',
    'Murphy_2017',
    'Gordon_2021',
    'Anumarlapudi_2023'
]
DEFAULT_LITERATURE_CITATIONS = {
    'Sieber_1973': 'Sieber (1973)',
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
    'Maron_2000': 'Maron et al. (2000)',
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
    'Gordon_2021': 'Gordon et al. (2021)',
    'Kravtsov_2022': 'Kravtsov et al. (2022)',
    'Lee_2022': 'Lee et al. (2022)',
    'Spiewak_2022': 'Spiewak et al. (2022)',
    'Bhat_2023': 'Bhat et al. (2023)',
    'Gitika_2023': 'Gitika et al. (2023)',
    'Posselt_2023': 'Posselt et al. (2023)',
    'Anumarlapudi_2023': 'Anumarlapudi et al. (2023)'
}
JANKOWSKI_LITERATURE_SET = [  # Part of literature used by Jankowski et al. (2018)
    'Jankowski_2018',
    'Sieber_1973',
    'Bartel_1978',
    'Izvekova_1981',
    'Lorimer_1995b',
    'van_Ommen_1997',
    'Maron_2000',
    'Malofeev_2000',
    'Karastergiou_2005',
    'Johnston_2006',
    'Kijak_2007',
    'Keith_2011',
    'Bates_2011',
    'Kijak_2011',
    'Zakharenko_2013',
    'Bilous_2016',
    'Dai_2015',
    'Bell_2016',
    'Basu_2016',
    'Han_2016',
    'Murphy_2017',
    'Kijak_2017'
]
