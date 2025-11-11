# Pulsar spectrum fitting

This repository contains code to fit pulsar radio spectra using Bayesian methods.

The dataset is based on a compilation of literature data, including the latest [`pulsar-spectra`](https://github.com/NickSwainston/pulsar_spectra) catalogue (version 2.1.0) and a set of additional recent measurements.

The fitting methods include various spectral models and options for handling outliers and systematic uncertainties. Although most of the options are not used in the paper, they present our attempts to improve the fitting process and account for data issues.

## Installation

1. Prepare a Python environment between versions 3.9 and 3.10. A virtual environment is recommended.

2. Install the required packages (`pandas` has to be downgraded to version < 2.0 due to compatibility issues with `pulsar-spectra`):
```bash
pip install -r requirements.txt
```

3. Then, upgrade `pandas` back to the latest version:
```bash
pip install --upgrade pandas
```

## Usage

### Data compilation and fitting

Default Bayesian fitting with Cauchy likelihood:
```bash
python fit.py
```

Method used in the paper: Bayesian fitting with Gaussian likelihood, and a dynamic $e_{\text{fac}}$ that is calculated based on a 50% $e_{\text{quad}}$:
```bash
python fit.py --gaussian --efac_qbound .5
```

Catalogue options:
* `--jname <name>`: Specific pulsar name (string). Can be a single name or multiple names separated by `;`.
* `--lit_set <file>`: Customize literature list (string, default: None).
* `--atnf`: Include ATNF pulsar catalogue (flag, default: False).
* `--atnf_ver <version>`: ATNF pulsar catalogue version (string, default: `1.54`).
* `--jan_set`: Use Jankowski et al. (2018)'s (reproduced) dataset (flag, default: False). This is an incomplete dataset reproduced from their literature list, to our best effort.
* `--refresh`: Refresh the catalogue (flag, default: False).

Fitting behaviour and priors:
* `--model <models>`: Specific model name(s) (string). Default: `simple_power_law;broken_power_law;log_parabolic_spectrum;high_frequency_cut_off_power_law;low_frequency_turn_over_power_law;double_turn_over_spectrum` (multiple models can be given separated by `;`).
* `--no_requirements`: Do not require the dataset to have at least 4 points and a frequency range of at least 2 (flag, default: False).
* `--fixed_freq_prior`: Use fixed frequency prior (flag, default: False).

Outlier handling / likelihood options:
* `--gaussian`: Use Gaussian likelihood (default is Cauchy likelihood when omitted) (flag, default: False).
* `--gaussian_patch`: Use simplified Gaussian distribution that works better for a small dataset (flag, default: False).
* `--outliers_rm`: Remove outliers from the dataset (flag, default: False).
* `--outliers_min <value>`: Set minimum YERR / Y (when `--outliers_rm` is not set) (float).
* `--outliers_min_plus <value>`: Add a systematic error to the uncertainty instead of replacing it (float).
* `--outliers_all <value>`: Ignore reported YERRs and set all YERR/Y to this value (float).

Systematic error / extra uncertainty parameters:
* `--equad <value>`: Add an additional systematic error (float).
* `--efac <value>`: Multiply reported uncertainties by this systematic error factor (float).
* `--efac_qbound <value>`: Determine $e_{\text{fac}}$ dynamically based on an $e_{\text{quad}}$ bound (float). Example used in the paper: `--efac_qbound .5`.

AIC / Jankowski et al. (2018) method:
* `--aic`: Use Jankowski et al. (2018)'s AIC-based method instead of the Bayesian fit (flag, default: False). Code is adapted from `pulsar-spectra`.
* `--aic_no_corr`: Do not apply the small-sample correction term in the AIC calculation (flag, default: False).
* Note: none of the following parameters apply to the AIC method: `--fixed_freq_prior`, `--gaussian`, `--gaussian_patch`, `--outliers_rm`, `--outliers_min`, `--outliers_min_plus`, `--outliers_all`, `--equad`, `--efac`, `--efac_qbound`, `--no_checkpoint`, `--corner`.

Output, multiprocessing and plotting:
* `--label <label>`: Output directory label (used when `--outdir` is not set).
* `--outdir <dir>`: Output directory (if not set, a timestamped `output/outdir_YYYY-MM-DD_HH-MM-SS` is created).
* `--override`: Override finished jobs (flag, default: False).
* `--nproc <int>`: Number of parallel processes to use (int). Default: `cpu_count() - 1` or 1.
* `--no_checkpoint`: Do not save intermediate pickle dump files (flag, default: False).
* `--no_plot`: Do not create result plots (flag, default: False).
* `--corner`: Generate a corner plot (flag, default: False).
* `--pdf`: Save plots as PDF instead of PNG (flag, default: False).
* `--print_lit`: Print literature list and save it to the output directory (flag, default: False). Exits the program after printing without performing any fitting.

### Processing fitted results

Extract variables from the output files:
```bash
python processing.py <output_dir>              # default variable is 'log_evidence'
python processing.py <output_dir> --var "aic"  # extract AIC values (if AIC method was used)
```

Extract estimated parameter values (including errors) from the output files:
```bash
python processing.py <output_dir> --var "param_estimates"
```

Extract frequency-flux plots from the output files (`<filter>` can be a list of pulsar names separated by `;`, or an expression regarding the number of measurements in a pulsar e.g. `>=15`):
```bash
python processing.py <output_dir> --plot <filter>           # default format to be extracted is "png"
python processing.py <output_dir> --plot <filter> --plot_format "pdf" # specify format
```

## Dataset

The literature dataset used in this work is compiled from various sources. The main component is the `pulsar-spectra` catalogue version 2.1.0. In addition, we have added several recent publications and some older literature that were not included in the catalogue.

[^a]: Literature present in the `pulsar-spectra` catalogue version 2.0.4.
[^b]: New literature added in the `pulsar-spectra` catalogue version 2.1.0.
[^c]: Supplementary literature added to the catalogue as part of the effort of the paper.
[^i]: Literature with data obtained through imaging surveys.
[^j]: Literature present in Jankowski et al. (2018)'s dataset.

|       Cite key        |                                                     Citation                                                     | Pulsar count | Frequency range (MHz) | Note         |
|:---------------------:|:----------------------------------------------------------------------------------------------------------------:|:------------:|:---------------------:|:-------------|
|     `McLean_1973`     |                      [McLean (1973)](https://ui.adsabs.harvard.edu/abs/1973MNRAS.165..133M)                      |      18      |        408-408        | [^a]         |
|     `Sieber_1973`     |                      [Sieber (1973)](https://ui.adsabs.harvard.edu/abs/1973A&A....28..237S)                      |      27      |       38-10690        | [^c][^j]     |
|     `Bartel_1978`     |                  [Bartel et al. (1978)](https://ui.adsabs.harvard.edu/abs/1978A&A....68..361B)                   |      18      |      14800-22700      | [^a][^j]     |
|  `Manchester_1978a`   |                [Manchester et al. (1978)](https://ui.adsabs.harvard.edu/abs/1978MNRAS.185..409M)                 |     224      |        40-408         | [^a]         |
|    `Izvekova_1981`    |                 [Izvekova et al. (1981)](https://ui.adsabs.harvard.edu/abs/1981Ap&SS..78...45I)                  |      73      |        39-102         | [^a][^j]     |
|     `Dewey_1985`      |                   [Dewey et al. (1985)](https://ui.adsabs.harvard.edu/abs/1985ApJ...294L..25D)                   |      34      |        390-390        | [^a]         |
|     `Stokes_1985`     |                  [Stokes et al. (1985)](https://ui.adsabs.harvard.edu/abs/1985Natur.317..787S)                   |      20      |        390-390        | [^a]         |
|      `Slee_1986`      |                   [Slee et al. (1986)](https://ui.adsabs.harvard.edu/abs/1986AuJPh..39..103S)                    |      44      |        80-160         | [^b]         |
|     `Stokes_1986`     |                  [Stokes et al. (1986)](https://ui.adsabs.harvard.edu/abs/1986ApJ...311..694S)                   |      5       |        430-430        | [^a]         |
|    `Fruchter_1988`    |                 [Fruchter et al. (1988)](https://ui.adsabs.harvard.edu/abs/1988Natur.333..237F)                  |      1       |        430-430        | [^b]         |
|    `Fruchter_1990`    |                 [Fruchter et al. (1990)](https://ui.adsabs.harvard.edu/abs/1990ApJ...351..642F)                  |      1       |       1490-1490       | [^b]         |
|   `Wolszczan_1992`    |                [Wolszczan & Frail (1992)](https://ui.adsabs.harvard.edu/abs/1992Natur.355..145W)                 |      1       |       430-1400        | [^a]         |
|  `Wielebinski_1993`   |                [Wielebinski et al. (1993)](https://ui.adsabs.harvard.edu/abs/1993A&A...272L..13W)                |      4       |      33900-34800      | [^b]         |
|    `Malofeev_1993`    |                     [Malofeev (1993)](https://ui.adsabs.harvard.edu/abs/1993AstL...19..138M)                     |      33      |        61-102         | [^a]         |
|     `Bailes_1994`     |                  [Bailes et al. (1994)](https://ui.adsabs.harvard.edu/abs/1994ApJ...425L..41B)                   |      3       |        436-436        | [^b]         |
|    `Navarro_1995`     |                  [Navarro et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995ApJ...455L..55N)                  |      1       |       411-1404        | [^b]         |
|   `Seiradakis_1995`   |                [Seiradakis et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A&AS..111..205S)                 |     188      |      1315-10550       | [^a]         |
|    `Lorimer_1995b`    |                         [Lorimer et al. (1995)](https://doi.org/10.1093/mnras/273.2.411)                         |     278      |       408-1606        | [^a][^j]     |
|   `Manchester_1995`   |              [Manchester & Johnston (1995)](https://ui.adsabs.harvard.edu/abs/1995ApJ...441L..65M)               |      2       |       1400-8300       | [^a]         |
|     `Camilo_1995`     |                  [Camilo & Nice (1995)](https://ui.adsabs.harvard.edu/abs/1995ApJ...445..756C)                   |      29      |        430-430        | [^a]         |
|      `Qiao_1995`      |                   [Qiao et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995MNRAS.274..572Q)                    |      61      |       660-1440        | [^a]         |
|   `Manchester_1996`   |                [Manchester et al. (1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.279.1235M)                 |      55      |        436-436        | [^a]         |
|     `Biggs_1996`      |                   [Biggs & Lyne (1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..691B)                   |      4       |        408-408        | [^a]         |
|     `Camilo_1996`     |                  [Camilo et al. (1996)](https://ui.adsabs.harvard.edu/abs/1996ApJ...469..819C)                   |      19      |        430-800        | [^b]         |
|  `Hoensbroech_1997`   |            [von Hoensbroech & Xilouris (1997)](https://ui.adsabs.harvard.edu/abs/1997A&AS..126..121V)            |      27      |      1410-10450       | [^a]         |
|     `Kramer_1997`     |                  [Kramer et al. (1997)](https://ui.adsabs.harvard.edu/abs/1997ApJ...488..364K)                   |      4       |      14600-43000      | [^a]         |
|   `van_Ommen_1997`    |                 [van Ommen et al. (1997)](https://ui.adsabs.harvard.edu/abs/1997MNRAS.287..307V)                 |      82      |        800-960        | [^a][^j]     |
|     `Kijak_1997`      |                   [Kijak et al. (1997)](https://ui.adsabs.harvard.edu/abs/1997A&A...318L..63K)                   |      4       |       4850-4850       | [^b]         |
|     `Kijak_1998`      |                   [Kijak et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998A&AS..127..153K)                   |      83      |       4850-4850       | [^a]         |
|    `Shrauner_1998`    |                 [Shrauner et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998ApJ...509..785S)                  |      20      |         82-82         | [^a]         |
|     `Kramer_1998`     |                  [Kramer et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998ApJ...501..270K)                   |      23      |       1410-1579       | [^a]         |
|    `Toscano_1998`     |                  [Toscano et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998ApJ...506..863T)                  |      19      |       436-1660        | [^a]         |
|     `Stairs_1999`     |                  [Stairs et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999ApJS..123..627S)                   |      19      |       410-1414        | [^a]         |
|    `Weisberg_1999`    |                 [Weisberg et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999ApJS..121..171W)                  |      98      |       1418-1418       | [^a]         |
|     `Kramer_1999`     |                  [Kramer et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999ApJ...526..957K)                   |      15      |       2695-4850       | [^a]         |
|      `Han_1999`       |                    [Han & Tian (1999)](https://ui.adsabs.harvard.edu/abs/1999A&AS..136..571H)                    |     106      |       1435-1435       | [^a]         |
|     `Maron_2000`      |              [Maron et al. (2000)](https://aas.aanda.org/articles/aas/abs/2000/20/h2144/h2144.html)              |     281      |       40-87000        | [^c][^j]     |
|    `Malofeev_2000`    |                 [Malofeev et al. (2000)](https://ui.adsabs.harvard.edu/abs/2000ARep...44..436M)                  |     211      |        102-102        | [^a][^j]     |
|  `Kouwenhoven_2000`   |                   [Kouwenhoven (2000)](https://ui.adsabs.harvard.edu/abs/2000A&AS..145..243K)                    |      68      |        325-325        | [^a][^i]     |
|     `Lommen_2000`     |                  [Lommen et al. (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...545.1007L)                   |      3       |       430-1400        | [^a]         |
|     `McGary_2001`     |                  [McGary et al. (2001)](https://ui.adsabs.harvard.edu/abs/2001AJ....121.1192M)                   |      3       |       1452-1452       | [^a][^i]     |
|    `Giacani_2001`     |                  [Giacani et al. (2001)](https://ui.adsabs.harvard.edu/abs/2001AJ....121.3133G)                  |      2       |       1420-8460       | [^a]         |
|     `Kuzmin_2001`     |                [Kuzmin & Losovsky (2001)](https://ui.adsabs.harvard.edu/abs/2001A&A...368..230K)                 |      30      |        102-111        | [^a]         |
|   `Manchester_2001`   |                [Manchester et al. (2001)](https://ui.adsabs.harvard.edu/abs/2001MNRAS.328...17M)                 |     100      |       1374-1374       | [^a]         |
|     `Morris_2002`     |                  [Morris et al. (2002)](https://ui.adsabs.harvard.edu/abs/2002MNRAS.335..275M)                   |     120      |       1374-1374       | [^a]         |
|     `Maron_2004`      |                   [Maron et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A&A...413L..19M)                   |      3       |       8350-8350       | [^b]         |
|    `Esamdin_2004`     |                  [Esamdin et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A&A...425..949E)                  |      2       |        327-327        | [^a]         |
|     `Hobbs_2004a`     |                   [Hobbs et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004MNRAS.352.1439H)                   |     453      |       1400-1400       | [^a]         |
|  `Karastergiou_2005`  |               [Karastergiou et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005MNRAS.359..481K)                |      48      |       3100-3100       | [^a][^j]     |
|   `Champion_2005b`    |                     [Champion (2005)](https://ui.adsabs.harvard.edu/abs/2005PhDT.......282C)                     |      1       |        327-430        | [^a]         |
|   `Champion_2005a`    |                 [Champion et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005MNRAS.363..929C)                  |      17      |        430-430        | [^a]         |
|    `Lorimer_2005`     |                  [Lorimer et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005MNRAS.359.1524L)                  |      38      |        400-430        | [^a]         |
|    `Johnston_2006`    |                 [Johnston et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.369.1916J)                  |      31      |       8356-8356       | [^a][^j]     |
|    `Crawford_2007`    |                [Crawford & Tiffany (2007)](https://ui.adsabs.harvard.edu/abs/2007AJ....134.1231C)                |      2       |       1384-3100       | [^a]         |
|     `Kijak_2007`      |          [Kijak et al. (2007)](https://www.aanda.org/articles/aa/abs/2007/05/aa6125-06/aa6125-06.html)           |      11      |       325-1060        | [^a][^j]     |
|    `Champion_2008`    |                 [Champion et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008Sci...320.1309C)                  |      1       |       1400-5000       | [^b]         |
|     `Deller_2009`     |                  [Deller et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...701.1243D)                   |      9       |       1650-1650       | [^a]         |
|     `Joshi_2009`      |                   [Joshi et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.398..943J)                   |      3       |       626-1400        | [^a]         |
|     `Bates_2011`      |                   [Bates et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.1575B)                   |      18      |       6591-6591       | [^a][^j]     |
|     `Kijak_2011`      |         [Kijak et al. (2011)](https://www.aanda.org/articles/aa/abs/2011/07/aa14274-10/aa14274-10.html)          |      15      |       610-4850        | [^a][^j]     |
|     `Keith_2011`      |                   [Keith et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.416..346K)                   |      9       |      17000-24000      | [^a][^j]     |
|    `Hessels_2011`     |                  [Hessels et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011AIPC.1357...40H)                  |      12      |        350-350        | [^b]         |
|     `Lynch_2012`      |                   [Lynch et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...745..109L)                   |      12      |       2000-2000       | [^a]         |
|   `Kowalinska_2012`   |                [Kowalińska et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012ASPC..466..101K)                 |      5       |       8350-8350       | [^b]         |
|    `Demorest_2013`    |                 [Demorest et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...762...94D)                  |      17      |       327-2300        | [^a]         |
|     `Dowell_2013`     |                  [Dowell et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..28D)                   |      1       |         41-81         | [^a]         |
|   `Zakharenko_2013`   |                [Zakharenko et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013MNRAS.431.3624Z)                 |      40      |         20-25         | [^a][^j]     |
|   `Manchester_2013`   |                [Manchester et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013PASA...30...17M)                 |      20      |       700-3100        | [^a]         |
|     `Boyles_2013`     |            [Boyles et al. (2013)](https://iopscience.iop.org/article/10.1088/0004-637X/763/2/80/meta)            |      13      |        820-820        | [^a]         |
|    `Stovall_2014`     |                  [Stovall et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJ...791...67S)                  |      67      |        350-820        | [^a]         |
|    `Dembska_2014`     |                          [Dembska et al. (2014)](https://doi.org/10.1093/mnras/stu1905)                          |      19      |       610-8350        | [^a]         |
|       `Ng_2015`       |                    [Ng et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015MNRAS.450.2922N)                     |      57      |       325-1352        | [^a]         |
|    `Dembska_2015`     |                  [Dembska et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.1869D)                  |      6       |        610-610        | [^a]         |
|      `Dai_2015`       |                    [Dai et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3223D)                    |      24      |       730-3100        | [^a][^j]     |
|    `Lazarus_2015`     |                  [Lazarus et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...812...81L)                  |     127      |       1375-1375       | [^a]         |
|      `Han_2016`       |                    [Han et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016RAA....16..159H)                    |     228      |       1274-1523       | [^a][^j]     |
| `Bhattacharyya_2016`  |               [Bhattacharyya et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...817..130B)               |      12      |        322-322        | [^a]         |
|      `Basu_2016`      |                   [Basu et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.2509B)                    |      1       |       325-1280        | [^a][^j]     |
|     `Bilous_2016`     |                  [Bilous et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A&A...591A.134B)                   |     158      |        149-149        | [^a][^j]     |
|      `Bell_2016`      |                           [Bell et al. (2016)](https://doi.org/10.1093/mnras/stw1293)                            |      17      |        154-154        | [^a][^j]     |
|   `Kondratiev_2016`   |                [Kondratiev et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A&A...585A.128K)                 |      48      |        149-149        | [^a][^i]     |
|     `Frail_2016`      |                   [Frail et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...829..119F)                   |     200      |        148-148        | [^a][^i]     |
|     `Levin_2016`      |                   [Levin et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...818..166L)                   |      37      |       1500-1500       | [^b]         |
|     `Kijak_2017`      |                        [Kijak et al. (2017)](https://dx.doi.org/10.3847/1538-4357/aa6ff2)                        |      12      |        325-610        | [^a][^j]     |
|      `Xue_2017`       |                    [Xue et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017PASA...34...70X)                    |      48      |        185-185        | [^a]         |
|    `Mignani_2017`     |                  [Mignani et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..10M)                  |      1       |     97500-343500      | [^a]         |
|      `Zhao_2017`      |                   [Zhao et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...845..156Z)                    |      26      |       8600-8600       | [^a]         |
|     `Murphy_2017`     | [Murphy et al. (2017)](https://www.cambridge.org/core/product/identifier/S1323358017000133/type/journal_article) |      60      |        76-227         | [^a][^i][^j] |
|      `Basu_2018`      |                   [Basu et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.1469B)                    |      6       |       325-1280        | [^a]         |
|   `Jankowski_2018`    |               [Jankowski et al. (2018)](http://academic.oup.com/mnras/article/473/4/4436/4315944)                |     418      |       728-3100        | [^a][^j]     |
|    `Brinkman_2018`    |                         [Brinkman et al. (2018)](https://doi.org/10.1093/mnras/stx2842)                          |      12      |       327-1400        | [^a]         |
|    `Johnston_2018`    |                 [Johnston & Kerr (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.4629J)                  |     585      |       1360-1360       | [^a]         |
|     `RoZko_2018`      |                   [Rożko et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.2193R)                   |      2       |       325-5900        | [^a]         |
|    `Gentile_2018`     |                   [Brook et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...868..122B)                   |      28      |       430-2100        | [^a]         |
|     `Surnis_2019`     |                  [Surnis et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...870....8S)                   |      3       |       325-1170        | [^a]         |
|    `Sanidas_2019`     |                  [Sanidas et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019A&A...626A.104S)                  |     288      |        135-135        | [^a]         |
|     `Zhang_2019`      |                   [Zhang et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...885L..37Z)                   |      3       |       768-3968        | [^a]         |
|   `Jankowski_2019`    |                 [Jankowski et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.3691J)                 |     205      |        843-843        | [^a]         |
|      `Kaur_2019`      |                   [Kaur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...882..133K)                    |      1       |        81-220         | [^a]         |
|      `Xie_2019`       |                    [Xie et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019RAA....19..103X)                    |      32      |       1369-1369       | [^a]         |
|      `Zhao_2019`      |                   [Zhao et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...874...64Z)                    |      71      |       4820-5124       | [^a]         |
|     `Bilous_2020`     |                  [Bilous et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A&A...635A..75B)                   |      43      |         54-64         | [^a]         |
|    `Crowter_2020`     |                  [Crowter et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3052C)                  |      1       |       350-1500        | [^a]         |
|    `Michilli_2020`    |                 [Michilli et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491..725M)                  |      19      |       129-1532        | [^a]         |
|     `Curylo_2020`     |              [Curyło et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020past.conf...69C/abstract)              |      1       |        150-150        | [^a]         |
|      `Tan_2020`       |                    [Tan et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.5878T)                    |      20      |       119-1532        | [^a]         |
|   `Bondonneau_2020`   |                [Bondonneau et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A&A...635A..76B)                 |      64      |         53-65         | [^a]         |
|      `Alam_2021`      |                   [Alam et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJS..252....4A)                    |      47      |       430-2100        | [^a]         |
|     `Gordon_2021`     |                       [Gordon et al. (2021)](https://dx.doi.org/10.3847/1538-4365/ac05c0)                        |      44      |       3000-3000       | [^c][^i]     |
|      `Han_2021`       |                    [Han et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021RAA....21..107H)                    |     201      |       1250-1250       | [^a]         |
|   `Bondonneau_2021`   |                [Bondonneau et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A&A...652A..34B)                 |      12      |         50-50         | [^a]         |
|    `Johnston_2021`    |                 [Johnston et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.1253J)                  |      44      |       1369-1369       | [^a]         |
| `Shapiro_Albert_2021` |              [Shapiro-Albert et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...909..219S)               |      3       |       430-1500        | [^a]         |
|    `Spiewak_2022`     |                  [Spiewak et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022PASA...39...27S)                  |     189      |       945-1623        | [^a]         |
|      `Lee_2022`       |                    [Lee et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022PASA...39...42L)                    |      22      |        70-352         | [^a]         |
|      `Bhat_2023`      |                   [Bhat et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023PASA...40...20B)                    |     120      |        154-154        | [^a]         |
|  `Anumarlapudi_2023`  |                    [Anumarlapudi et al. (2023)](https://dx.doi.org/10.3847/1538-4357/aceb5d)                     |     150      |        888-888        | [^c][^i]     |
|    `Posselt_2023`     |                [Posselt et al. (2023)](https://academic.oup.com/mnras/article/520/3/4582/7049638)                |     1237     |       941-1640        | [^c]         |
|     `Gitika_2023`     |                          [Gitika et al. (2023)](https://doi.org/10.1093/mnras/stad2841)                          |      89      |       944-1625        | [^a]         |
|     `Keith_2024`      |                   [Keith et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.1581K)                   |     597      |       1284-1284       | [^b]         |
|      `Wang_2024`      |                   [Wang et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...961...48W)                    |      10      |       2250-8600       | [^b]         |
|     `Kumar_2025`      |                   [Kumar et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025ApJ...982..132K)                   |      96      |         35-79         | [^b]         |
