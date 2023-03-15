import pandas as pd
import glob
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.matching import match_coordinates_sky as match_coord
from astropy.table import Table


################################################################################

filterid = 1 # 1 for g, 2 for r, 12 for combined g and r


if filterid==1: band='g'
if filterid==2: band='r'
if filterid==12: band='g_and_r'

################################################################################

max_sep = 1.5 * u.arcsec
################################################################################

main_path = '/Volumes/Paula_SSD/ZTF_DR_work/'
input_main_folder = main_path+'classification_DR11_4MOST_extragalactic_2023/classifications_and_features/'
catalog_folder = '../catalogs_others/'
input_catalog = catalog_folder + 'Chen2020_VS_ZTFDR2.parquet' #example for Chen et al. 2020 catalog
output_folder = '../catalogs_others/'
output_file = output_folder+'Chen2020_DR11_classifications_match_'+band+'band.parquet'
output_file_unique = output_folder+'Chen2020_DR11_classifications_match_unique_'+band+'band.parquet'

################################################################################


input_folder = input_main_folder + band + 'band/'

input_files = sorted(glob.glob(input_folder+'*parquet'))


################################################################################
#reading catalog file

catalog_df = pd.read_parquet(input_catalog)
catalog_df = catalog_df[['ID','RAJ2000', 'DEJ2000','Per-g', 'Per-r','Type']] #replace according to catalog format
catalog_df = catalog_df[catalog_df.DEJ2000<=20]
coords_catalog = SkyCoord(ra=catalog_df.RAJ2000.values*u.degree, dec=catalog_df.DEJ2000.values*u.degree)

################################################################################

match_cat = []

for file in input_files:

    print('processing file ',file)
    df = pd.read_parquet(file)
    df['objectid'] = df['objectid'].astype(np.int64)

    coords_DR = SkyCoord(ra=df.objra.values*u.degree, dec=df.objdec.values*u.degree)
    idx, d2d, d3d = match_coord(coords_catalog,coords_DR,nthneighbor=1)
    ang=np.array(d2d)
    n=np.where(ang<max_sep.to(u.deg).value)
    n=n[0]

    print("number of matches",len(n))
    catalog_matches = catalog_df.iloc[n]
    catalog_matches.reset_index(inplace=True)
    catalog_matches.drop(['index'], axis=1, inplace=True)

    DR_matches = df.iloc[idx[n]]
    DR_matches.reset_index(inplace=True)
    DR_matches.drop(['index'], axis=1, inplace=True)



    df_concat = pd.concat([catalog_matches, DR_matches], axis=1)

    match_cat.append(df_concat)

match_cat = pd.concat(match_cat)

match_cat.reset_index(inplace=True)



match_cat.to_parquet(output_file)

b = match_cat.sort_values(by=['nepochs']).drop_duplicates('ID', keep='last')
b.to_parquet(output_file_unique)
