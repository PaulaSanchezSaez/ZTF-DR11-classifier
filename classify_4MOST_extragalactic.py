import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import pickle
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
import pyarrow
from scipy.stats import randint as sp_randint
from sklearn.utils import class_weight
from astropy.table import Table
import glob

#######################################################################################
#set only these parameters

filterid = 2 # 1 for g, 2 for r, 12 for combined g and r

main_path = '/Volumes/Expansion/ZTF_DR_work/'
input_files = sorted(glob.glob(main_path+'DR11_features_4MOST_extragalactic/'+'*parquet'))
output_main_folder = main_path+'classification_DR11_4MOST_extragalactic/classifications_and_features/'

#######################################################################################

if filterid==1: band='g'
if filterid==2: band='r'
if filterid==12: band='g_and_r'

output_folder = output_main_folder + band + 'band/'
output_AGN_folder = output_main_folder + 'AGN_' + band + 'band/'
output_plots_folder = output_main_folder+ 'plots_' + band + 'band/'

features_list_4model = 'models/optimal_hierarchical_features_'+band+'band_zsep'
model_init_layer = 'models/hierarchical_model_init_level_'+band+'band_zsep'
model_variable_layer = 'models/hierarchical_model_var_level_'+band+'band_zsep'
model_periodic_layer = 'models/hierarchical_model_periodic_level_'+band+'band_zsep'
model_stochastic_layer = 'models/hierarchical_model_stochastic_level_'+band+'band_zsep'
model_transient_layer = 'models/hierarchical_model_transient_level_'+band+'band_zsep'


input_files_names = np.core.defchararray.replace(input_files,main_path+'DR11_features_4MOST_extragalactic/','')
#input_files_names = input_files_names[700:]
#######################################################################################

#Loading RF models

rf_model_init = pd.read_pickle(model_init_layer)
rf_model_var = pd.read_pickle(model_variable_layer)
rf_model_periodic = pd.read_pickle(model_periodic_layer)
rf_model_stochastic = pd.read_pickle(model_stochastic_layer)
rf_model_transient = pd.read_pickle(model_transient_layer)

features_list = pd.read_pickle(features_list_4model)

#######################################################################################

for file in input_files_names:
#for i in range(10):
#    file = input_files_names[i]
    print('classifying file ',file)
    df_feat = pd.read_parquet(main_path+'DR11_features_4MOST_extragalactic/'+file)
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat = df_feat[(df_feat.filterid==filterid) & (df_feat.Mean<=20.5)]


    df_feat['gps1-rps1']=df_feat.gmag-df_feat.rmag
    df_feat['rps1-ips1']=df_feat.rmag-df_feat.imag
    df_feat['gps1-W1']=df_feat.gmag-df_feat.w1mpro_pm_2
    df_feat['gps1-W2']=df_feat.gmag-df_feat.w2mpro_pm_2
    df_feat['rps1-W1']=df_feat.rmag-df_feat.w1mpro_pm_2
    df_feat['rps1-W2']=df_feat.rmag-df_feat.w2mpro_pm_2
    df_feat['ips1-W1']=df_feat.imag-df_feat.w1mpro_pm_2
    df_feat['ips1-W2']=df_feat.imag-df_feat.w2mpro_pm_2
    df_feat['W1-W2']=df_feat.w1mpro_pm_2-df_feat.w2mpro_pm_2



    df=df_feat[features_list]
    df = df.astype('float32', copy=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.fillna(-999)


    #classifying with init level

    y_pred_init = rf_model_init.predict(df)
    y_pred_prob_init = rf_model_init.predict_proba(df)

    # separating classes (according to the predicted classification):

    X_var =  df.loc[y_pred_init=='variable',:]

    y_pred_prob_init_var = np.max(y_pred_prob_init[np.where(y_pred_init=='variable')],axis=1)

    #classifying with variable level

    y_pred_var = rf_model_var.predict(X_var)
    y_pred_prob_var = rf_model_var.predict_proba(X_var)

    # separating classes (according to the predicted classification):

    X_periodic =  X_var.loc[y_pred_var=='Periodic',:]
    X_stochastic =  X_var.loc[y_pred_var=='Stochastic',:]
    X_transient =  X_var.loc[y_pred_var=='Transient',:]

    y_pred_prob_init_periodic = y_pred_prob_init_var[np.where(y_pred_var=='Periodic')]
    y_pred_prob_init_stochastic = y_pred_prob_init_var[np.where(y_pred_var=='Stochastic')]
    y_pred_prob_init_transient = y_pred_prob_init_var[np.where(y_pred_var=='Transient')]

    y_pred_prob_var_periodic = np.max(y_pred_prob_init[np.where(y_pred_var=='Periodic')],axis=1)
    y_pred_prob_var_stochastic = np.max(y_pred_prob_init[np.where(y_pred_var=='Stochastic')],axis=1)
    y_pred_prob_var_transient = np.max(y_pred_prob_init[np.where(y_pred_var=='Transient')],axis=1)

    y_pred_periodic = rf_model_periodic.predict(X_periodic)
    y_pred_stochastic = rf_model_stochastic.predict(X_stochastic)
    y_pred_transient = rf_model_transient.predict(X_transient)

    y_pred_prob_periodic = rf_model_periodic.predict_proba(X_periodic)
    y_pred_prob_stochastic = rf_model_stochastic.predict_proba(X_stochastic)
    y_pred_prob_transient = rf_model_transient.predict_proba(X_transient)

    #putting all together

    #non var sources

    X_nonvar = df.loc[y_pred_init!='variable',:]
    X_nonvar.loc[:,'pred_class']=y_pred_init[np.where(y_pred_init!='variable')]
    X_nonvar.loc[:,'pred_class_prob']=np.max(y_pred_prob_init[np.where(y_pred_init!='variable')],axis=1)

    #periodic sources

    X_periodic.loc[:,'pred_class']=y_pred_periodic
    X_periodic.loc[:,'pred_class_prob']=np.max(y_pred_prob_periodic,axis=1)

    X_periodic.loc[:,'pred_init_class']='variable'
    X_periodic.loc[:,'pred_init_class_prob']=y_pred_prob_init_periodic

    X_periodic.loc[:,'pred_var_class']='Periodic'
    X_periodic.loc[:,'pred_var_class_prob']=y_pred_prob_var_periodic


    y_pred_prob_var = rf_model_init.predict_proba(df)

    #stocashtic sources

    X_stochastic.loc[:,'pred_class']=y_pred_stochastic
    X_stochastic.loc[:,'pred_class_prob']=np.max(y_pred_prob_stochastic,axis=1)

    X_stochastic.loc[:,'pred_init_class']='variable'
    X_stochastic.loc[:,'pred_init_class_prob']=y_pred_prob_init_stochastic

    X_stochastic.loc[:,'pred_var_class']='Stochastic'
    X_stochastic.loc[:,'pred_var_class_prob']=y_pred_prob_var_stochastic



    #transient sources

    X_transient.loc[:,'pred_class']=y_pred_transient
    X_transient.loc[:,'pred_class_prob']=np.max(y_pred_prob_transient,axis=1)

    X_transient.loc[:,'pred_init_class']='variable'
    X_transient.loc[:,'pred_init_class_prob']=y_pred_prob_init_transient

    X_transient.loc[:,'pred_var_class']='Transient'
    X_transient.loc[:,'pred_var_class_prob']=y_pred_prob_var_transient

    all_pred = pd.concat([X_nonvar,X_transient,X_stochastic,X_periodic])


    #out_file = df_feat[['objectid','objra','objdec','gal_b','gal_l','nepochs','Mean','gmag','rmag','imag','w1mpro_pm_2','w2mpro_pm_2']].join(all_pred)
    out_file = df_feat.join(all_pred[['pred_init_class','pred_init_class_prob','pred_var_class','pred_var_class_prob','pred_class','pred_class_prob']])

    #print(out_file.columns)
    out_file.to_parquet(output_folder+'classification_'+file)

    out_file_agn = out_file[(out_file.pred_class=="lowz-AGN") | (out_file.pred_class=="midz-AGN")  | (out_file.pred_class=="highz-AGN") | (out_file.pred_class=="Blazar")]
    out_file_agn.to_parquet(output_AGN_folder+'AGN_cands_'+file)

    #plotting results
    class_counts = Counter(out_file.pred_class)
    class_counts.most_common()
    print(class_counts)

    df_hist = pd.DataFrame.from_dict(class_counts, orient='index',columns=['number'])
    df_hist = df_hist.sort_values(by=['number'],ascending=False)
    df_hist.plot(kind='bar',legend=False,figsize=(15,7))
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(output_plots_folder+file+'.png')
    plt.clf()

    print('processed file ',file)
