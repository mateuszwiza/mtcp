"""
SCRIPT FOR APPLYING SINGLE-TARGET CONFORMAL REGRESSION
WITH A MULTI-TARGET UNDERLYING MODEL
"""

# Authors: Mateusz Wiza

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import RegressorChain
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nonconformist_multi.base import RegressorAdapter
from nonconformist_multi.icp import IcpRegressor
from nonconformist_multi.nc import RegressorNc, AbsErrorErrFunc, RegressorNormalizer
from nonconformist_multi.evaluation import cross_val_score
from nonconformist_multi.evaluation import RegIcpCvHelper
from nonconformist_multi.evaluation import reg_mean_errors, reg_median_size, reg_q3_size, reg_q1_size

from morfist import MixedRandomForest


def apply_cp(features, targets, underlying):
    """Perform normalized conformal prediction.
        Parameters
        ----------
        features : numpy array of shape [n_samples, n_features]
                   Features of each sample
        targets : numpy array of shape [n_samples, n_targets]
                  True output labels of each sample.
        underlying : model implementing fit() and predict()
                     Underlying model for conformal prediction.
        Returns
        -------
        Results cross_val_score() from averaged from the 4 iterations
        of cross validation.
    """
    # Adapt underlying and normalizing models for conformal prediction
    u_model = RegressorAdapter(underlying)
    n_model = RegressorAdapter(RandomForestRegressor(random_state=15))
    # Initialize the normalizing model
    normalizer = RegressorNormalizer(u_model, n_model, AbsErrorErrFunc())
    # Define the nonconformity function
    ncs = RegressorNc(u_model, AbsErrorErrFunc(), normalizer)
    # Initialize inductive conformal regressor
    icp = IcpRegressor(ncs)

    # Perform cross validation
    scores = cross_val_score(RegIcpCvHelper(icp),
                             features,
                             targets,
                             iterations=4,
                             folds=10,
                             scoring_funcs=[reg_mean_errors, reg_median_size, reg_q1_size, reg_q3_size],
                             significance_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Return the results
    return scores.drop(['fold', 'iter'], axis=1)


# -----------------------------------------------------------------------------
# PREPARE DATA
# -----------------------------------------------------------------------------

# Read data from CSV
data = pd.read_csv("data.csv", sep=',', encoding='latin-1')

# Transform categorical variable 'Country' into dummy variables
dummy_country = data['country'].str.get_dummies()
data = pd.concat([data, dummy_country], axis=1)

# Remove outliers
data = data[data['STP_hours'] < 8000]
data = data[data['avg_dur'] < 15]
data = data[data['daily_occ_stp'] > 0]
data = data[data['daily_occ_stp'] < 1]
data = data[data['rev_adj'] > 0]
data = data[data['rev_adj'] < 10000]

# Select features / inputs
data_input = data[
    ['Size', 'ts', 'comp', 'spaces', 'office_osm', 'shop_osm', 'food_osm', 'all_osm', 'edu_osm', 'health_osm',
     'hotel_osm', 'ind_osm', 'office', 'BE', 'DE', 'FR', 'GB', 'IE', 'NL']]

# Select targets / outputs
data_output = data[['STP_hours', 'avg_dur', 'avg_entry', 'daily_occ_stp', 'rev_adj']]

# Define input and output matrix
Y = np.array(data_output)
X = np.array(data_input)

# Apply 0-1 normalization to the dataset
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

# -----------------------------------------------------------------------------
# EXPERIMENT TO DETERMINE BEST PERFORMING MULTI-TARGET MODEL
# -----------------------------------------------------------------------------

# 1. Principal Component Regression (PCR)
# Define PCR pipeline
pcr = make_pipeline(PCA(n_components=1), RandomForestRegressor())
# Perform conformal prediction
scores_pcr = apply_cp(X, Y, pcr)
# Save results as CSV
scores_pcr.to_csv('graph-data/scores_multi_pcr.csv')

# 2. Partial Least Squares (PLS)
# Initialize PLS
pls = PLSRegression(n_components=5)
# Perform conformal prediction
scores_pls = apply_cp(X, Y, pls)
# Save results as CSV
scores_pls.to_csv('graph-data/scores_multi_pls.csv')

# 3. Stacked Single-Target (SST)
# Initialize the initial single-target model
rf = RandomForestRegressor(random_state=22)
# Get cross-validated single-target predictions for each target
temp_predictions = cross_val_predict(rf, X, Y, cv=10)
# Augment the input space with the single-target predictions
X_pls = np.concatenate((X, temp_predictions), axis=1)
# Perform conformal prediction using the augmented input space
scores_sst = apply_cp(X_pls, Y, rf)
# Save results as CSV
scores_sst.to_csv('graph-data/scores_multi_sst.csv')

# 4. Ensemble of Regression Chains (ERC)
# Define number of chains
n_chains = 10
# Perform conformal prediction for each chain
results_list = []
for n in range(n_chains):
    # Define a regression chain with random order
    chain = RegressorChain(base_estimator=rf, order='random')
    # Perform conformal prediction
    scores_chain = apply_cp(X, Y, chain)
    results_list.append(scores_chain)
# Take the average of the results
scores_erc = pd.concat(results_list).groupby(level=0).mean()
# Save results as CSV
scores_erc.to_csv('graph-data/scores_multi_erc.csv')

# 5. Multi-Objective Random Forest (MORF)
# Initialize MORF
morf = MixedRandomForest(
    n_estimators=100,
    min_samples_leaf=1,
    class_targets=[1]
)
# Perform conformal prediction
scores_morf = apply_cp(X, Y, morf)
# Save results as CSV
scores_morf.to_csv('graph-data/scores_multi_morf.csv')

# -----------------------------------------------------------------------------
# PLOT THE RESULTS
# -----------------------------------------------------------------------------

# 1. Plot interval width for one target to identify the best performing model

# Get baseline results for comparison
scores_rev = pd.read_csv("graph-data/scores_rev.csv", sep=',', encoding='latin-1')

# Define significance levels and the 'ideal' width of 10% (for plots)
significance = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ideal = np.ones(len(significance)) * 10

# Get all results for output 4
algs = ['pcr', 'pls', 'sst', 'erc', 'morf']
dfs = {}
for alg in algs:
    filename = 'graph-data/scores_multi_' + str(alg) + '.csv'
    multi = pd.read_csv(filename, sep=',', encoding='latin-1')
    multi_temp = multi[multi['output'] == 4]
    dfs[str(alg)] = multi_temp.groupby(['significance']).mean()

# Generate the plot
plt.scatter(scores_rev.index, scores_rev['reg_median_size'] * 100, color='k', s=15)  # Baseline solution
plt.plot(scores_rev.index, scores_rev['reg_median_size'] * 100, color='k', linewidth=0.8)  # Baseline solution

for df in dfs:
    data = dfs[df]
    plt.scatter(data.index, data['reg_median_size'] * 100, s=15)
    plt.plot(data.index, data['reg_median_size'] * 100, linewidth=0.8)

plt.plot(significance, ideal, '--', color='k', linewidth=0.7)
plt.ylabel('Interval width [%]')
plt.xlabel('Significance level εₜ')
plt.legend(['Baseline', 'PCR', 'PLS', 'SST', 'ERC', 'MORF'])
plt.show()

# 2. Plot results for all targets for the best performing model
complete_results = pd.read_csv("graph-data/scores_multi_sst_new.csv", sep=',', encoding='latin-1')

# Split the complete_results dataframe into target-specific dataframes
# and take the average from the 4 iterations of cross validation
multi_temp = complete_results[complete_results['output'] == 0]
multi_stp = multi_temp.groupby(['significance']).mean()

multi_temp = complete_results[complete_results['output'] == 1]
multi_dur = multi_temp.groupby(['significance']).mean()

multi_temp = complete_results[complete_results['output'] == 2]
multi_entry = multi_temp.groupby(['significance']).mean()

multi_temp = complete_results[complete_results['output'] == 3]
multi_occstp = multi_temp.groupby(['significance']).mean()

multi_temp = complete_results[complete_results['output'] == 4]
multi_rev = multi_temp.groupby(['significance']).mean()

# Plot prediction interval widths (median and IQR)
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

ax = axs[0, 0]
diff_q3 = multi_stp['reg_q3_size'] * 100 - multi_stp['reg_median_size'] * 100
diff_q1 = multi_stp['reg_median_size'] * 100 - multi_stp['reg_q1_size'] * 100
ax.errorbar(multi_stp.index, multi_stp['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(multi_stp.index, multi_stp['reg_median_size'] * 100)
ax.plot(multi_stp.index, ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average daily sold parking hours')
ax.set_ylabel('Interval width [%]')

ax = axs[0, 1]
diff_q3 = multi_dur['reg_q3_size'] * 100 - multi_dur['reg_median_size'] * 100
diff_q1 = multi_dur['reg_median_size'] * 100 - multi_dur['reg_q1_size'] * 100
ax.errorbar(multi_dur.index, multi_dur['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(multi_dur.index, multi_dur['reg_median_size'] * 100)
ax.plot(multi_stp.index, ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average duration of stay')

ax = axs[1, 0]
diff_q3 = multi_entry['reg_q3_size'] * 100 - multi_entry['reg_median_size'] * 100
diff_q1 = multi_entry['reg_median_size'] * 100 - multi_entry['reg_q1_size'] * 100
ax.errorbar(multi_entry.index, multi_entry['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(multi_entry.index, multi_entry['reg_median_size'] * 100)
ax.plot(multi_stp.index, ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average time of entry')
ax.set_ylabel('Interval width [%]')

ax = axs[1, 1]
diff_q3 = multi_occstp['reg_q3_size'] * 100 - multi_occstp['reg_median_size'] * 100
diff_q1 = multi_occstp['reg_median_size'] * 100 - multi_occstp['reg_q1_size'] * 100
ax.errorbar(multi_occstp.index, multi_occstp['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(multi_occstp.index, multi_occstp['reg_median_size'] * 100)
ax.plot(multi_stp.index, ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average daytime occupancy')
ax.set_xlabel('Significance level εₜ')

ax = axs[2, 0]
diff_q3 = multi_rev['reg_q3_size'] * 100 - multi_rev['reg_median_size'] * 100
diff_q1 = multi_rev['reg_median_size'] * 100 - multi_rev['reg_q1_size'] * 100
ax.errorbar(multi_rev.index, multi_rev['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(multi_rev.index, multi_rev['reg_median_size'] * 100)
ax.plot(multi_stp.index, ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average daily revenue')
ax.set_ylabel('Interval width [%]')
ax.set_xlabel('Significance level εₜ')

fig.delaxes(axs[2, 1])
fig.suptitle('Tightness of prediction regions per target')

plt.show()

# Plot empirical validity
fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(10, 10))

ax = axs[0, 0]
ax.scatter(multi_stp.index, multi_stp['reg_mean_errors'])
ax.plot(multi_stp.index, multi_stp['reg_mean_errors'])
ax.plot(multi_stp.index, multi_stp.index, '--', color='k')
ax.set_title('Average daily sold parking hours')
ax.set_ylabel('Error rate')

ax = axs[0, 1]
ax.scatter(multi_dur.index, multi_dur['reg_mean_errors'])
ax.plot(multi_dur.index, multi_dur['reg_mean_errors'])
ax.plot(multi_dur.index, multi_dur.index, '--', color='k')
ax.set_title('Average duration of stay')

ax = axs[1, 0]
ax.scatter(multi_entry.index, multi_entry['reg_mean_errors'])
ax.plot(multi_entry.index, multi_entry['reg_mean_errors'])
ax.plot(multi_entry.index, multi_entry.index, '--', color='k')
ax.set_title('Average time of entry')
ax.set_ylabel('Error rate')

ax = axs[1, 1]
ax.scatter(multi_occstp.index, multi_occstp['reg_mean_errors'])
ax.plot(multi_occstp.index, multi_occstp['reg_mean_errors'])
ax.plot(multi_occstp.index, multi_occstp.index, '--', color='k')
ax.set_title('Average daytime occupancy')
ax.set_xlabel('Significance level εₜ')

ax = axs[2, 0]
ax.scatter(multi_rev.index, multi_rev['reg_mean_errors'])
ax.plot(multi_rev.index, multi_rev['reg_mean_errors'])
ax.plot(multi_rev.index, multi_rev.index, '--', color='k')
ax.set_title('Average daily revenue')
ax.set_ylabel('Error rate')
ax.set_xlabel('Significance level εₜ')

fig.delaxes(axs[2, 1])
fig.suptitle('Empirical validity per target')

plt.show()
