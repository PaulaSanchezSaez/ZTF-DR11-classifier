# ZTF-DR11-classifier
## Code used for the work presented in Sanchez-Saez et al. 2023

Code to train the g-band model: BRF_indep_g_band_hierarchical_PS1score_WISE_ZTFfeat_SNgrouped_zsep_20230309.ipynb <br />
Code to train the r-band model: BRF_indep_r_band_hierarchical_PS1score_WISE_ZTFfeat_SNgrouped_zsep_20230309.ipynb <br />
Code used to classify the ZTF/4MOST data: classify_4MOST_extragalactic.py <br />
Code used to obtain classifications for an external catalog: matching_cat_ZTF_DR11_classifications.py <br />


### Note: some of the features used by the code published in this repository have a name different than the one presented in the paper:

- Multiband_period -> Period # name inherited from the model presented in Sanchez-Saez et al. 2021a
- gps1-rps1 -> g-r
- rps1-ips1 -> r-i
- gps1-W1 -> g-W1
- gps1-W2 -> g-W2
- rps1-W1 -> r-W1
- rps1-W2 -> r-W2
- ips1-W1 -> i-W1
- ips1-W2 -> i-W2

