"""
SCRIPT FOR APPLYING SINGLE-TARGET CONFORMAL REGRESSION
WITH A SINGLE-TARGET UNDERLYING MODEL
"""

# Authors: Mateusz Wiza

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc, RegressorNormalizer
from nonconformist.evaluation import cross_val_score
from nonconformist.evaluation import RegIcpCvHelper
from nonconformist.evaluation import reg_mean_errors, reg_median_size, reg_q3_size, reg_q1_size


def apply_cp(features, target, underlying):
    """Perform not-normalized conformal prediction.
        Parameters
        ----------
        features : numpy array of shape [n_samples, n_features]
                   Features of each sample
        target : numpy array of shape [n_samples]
                 True output labels of each sample.
        underlying : model implementing fit() and predict()
                     Underlying model for conformal prediction.
        Returns
        -------
        Results cross_val_score() from averaged from the 4 iterations
        of cross validation.
    """
    # Adapt underlying model for conformal prediction
    u_model = RegressorAdapter(underlying)
    # Define the nonconformity function
    ncs = RegressorNc(u_model, AbsErrorErrFunc())
    # Initialize inductive conformal regressor
    icp = IcpRegressor(ncs)

    # Perform cross validation
    scores = cross_val_score(RegIcpCvHelper(icp),
                             features,
                             target,
                             iterations=4,
                             folds=10,
                             scoring_funcs=[reg_mean_errors, reg_median_size, reg_q1_size, reg_q3_size],
                             significance_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Return results averaged from the 4 iterations of cross validation
    return scores.groupby(['significance']).mean()


def apply_norm_cp(features, target, underlying, normalizing):
    """Perform normalized conformal prediction.
        Parameters
        ----------
        features : numpy array of shape [n_samples, n_features]
                   Features of each sample
        target : numpy array of shape [n_samples]
                 True output labels of each sample.
        underlying : model implementing fit() and predict()
                     Underlying model for conformal prediction.
        normalizing : model implementing fit() and predict()
                      Normalizing model for conformal prediction.
        Returns
        -------
        Results cross_val_score() from averaged from the 4 iterations
        of cross validation.
    """
    # Adapt underlying and normalizing models for conformal prediction
    u_model = RegressorAdapter(underlying)
    n_model = RegressorAdapter(normalizing)
    # Initialize the normalizing model
    normalizer = RegressorNormalizer(u_model, n_model, AbsErrorErrFunc())
    # Define the nonconformity function
    ncs = RegressorNc(u_model, AbsErrorErrFunc(), normalizer)
    # Initialize inductive conformal regressor
    icp = IcpRegressor(ncs)

    # Perform cross validation
    scores = cross_val_score(RegIcpCvHelper(icp),
                             features,
                             target,
                             iterations=4,
                             folds=10,
                             scoring_funcs=[reg_mean_errors, reg_median_size, reg_q1_size, reg_q3_size],
                             significance_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Return results averaged from the 4 iterations of cross validation
    return scores.groupby(['significance']).mean()


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
# EXPERIMENTS USING ONE TARGET TO DETERMINE OPTIMAL SOLUTION
# -----------------------------------------------------------------------------

# Define significance levels and the 'ideal' width of 10% (for plots)
significance = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ideal = np.ones(len(significance)) * 10

# 1. Experiment to determine the best performing underlying model
models = [Lasso(alpha=5.0),
          Ridge(alpha=5.0),
          GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1, loss='ls'),
          KNeighborsRegressor(n_neighbors=5),
          RandomForestRegressor(random_state=22)]

df_exp1 = {}

# Perform conformal prediction using different models
for model in models:
    df_exp1[str(model)[0:2]] = apply_cp(X, Y[:, 4], model)

# Plot the results (Fig. 1 in report)
for df in df_exp1:
    data = df_exp1[df]
    plt.scatter(data.index, data['reg_median_size'] * 100)
    plt.plot(data.index, data['reg_median_size'] * 100)

plt.plot(significance, ideal, '--', color='k', linewidth=0.5)
plt.ylabel('Interval width [%]')
plt.xlabel('Significance level εₜ')
plt.legend(['Lasso', 'Ridge', 'Gradient Boost', 'k-NN', 'Random Forest'])
plt.show()

# 2. Experiment to determine the best performing normalizing model
underlying_model = RandomForestRegressor(random_state=22)
norm_models = [GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1, loss='ls'),
               RandomForestRegressor(random_state=15)]

df_exp2 = {}

# Perform normalized conformal prediction using different normalizing models
for norm_model in norm_models:
    df_exp2[str(norm_model)[0:2]] = apply_norm_cp(X, Y[:, 4], underlying_model, norm_model)

# Plot the results (Fig. 2 in report)
plt.scatter(data.index, df_exp1['Ra']['reg_median_size'] * 100, color='C4')  # Baseline solution
plt.plot(data.index, df_exp1['Ra']['reg_median_size'] * 100, color='C4')  # Baseline solution

for df in df_exp2:
    if df == 'Ra':
        c = 'k'
    else:
        c = 'C8'
    data = df_exp2[df]
    plt.scatter(data.index, data['reg_median_size'] * 100, color=c)
    plt.plot(data.index, data['reg_median_size'] * 100, color=c)

plt.plot(significance, ideal, '--', color='k', linewidth=0.5)
plt.ylabel('Interval width [%]')
plt.xlabel('Significance level εₜ')
plt.legend(['Not normalized', 'Gradient Boost', 'Random Forest'])
plt.show()

# -----------------------------------------------------------------------------
# PLOT RESULTS FOR ALL TARGETS USING OPTIMAL BASELINE SOLUTION
# -----------------------------------------------------------------------------
rerun_experiments = False  # Set to True to re-run experiments instead of loading results from CSVs

# Get conformal prediction results by re-running experiments and save to CSV
if rerun_experiments:
    normalizing_model = RandomForestRegressor(random_state=15)

    scores_stp = apply_norm_cp(X, Y[:, 0], underlying_model, normalizing_model)
    scores_stp.to_csv('graph-data/scores_stp.csv')

    scores_dur = apply_norm_cp(X, Y[:, 1], underlying_model, normalizing_model)
    scores_dur.to_csv('graph-data/scores_dur.csv')

    scores_entry = apply_norm_cp(X, Y[:, 2], underlying_model, normalizing_model)
    scores_entry.to_csv('graph-data/scores_entry.csv')

    scores_occstp = apply_norm_cp(X, Y[:, 3], underlying_model, normalizing_model)
    scores_occstp.to_csv('graph-data/scores_occstp.csv')

    scores_rev = df_exp2['Ra']
    scores_rev.to_csv('graph-data/scores_rev.csv')

# Get conformal prediction results from CSV files
else:
    scores_stp = pd.read_csv("graph-data/scores_stp.csv", sep=',', encoding='latin-1')
    scores_dur = pd.read_csv("graph-data/scores_dur.csv", sep=',', encoding='latin-1')
    scores_entry = pd.read_csv("graph-data/scores_entry.csv", sep=',', encoding='latin-1')
    scores_occstp = pd.read_csv("graph-data/scores_occstp.csv", sep=',', encoding='latin-1')
    scores_rev = pd.read_csv("graph-data/scores_rev.csv", sep=',', encoding='latin-1')

# Plot prediction interval widths (median and IQR)
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

ax = axs[0, 0]
diff_q3 = scores_stp['reg_q3_size'] * 100 - scores_stp['reg_median_size'] * 100
diff_q1 = scores_stp['reg_median_size'] * 100 - scores_stp['reg_q1_size'] * 100
ax.errorbar(scores_stp['significance'], scores_stp['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(scores_stp['significance'], scores_stp['reg_median_size'] * 100)
ax.plot(scores_stp['significance'], ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average daily sold parking hours')
ax.set_ylabel('Interval width [%]')

ax = axs[0, 1]
diff_q3 = scores_dur['reg_q3_size'] * 100 - scores_dur['reg_median_size'] * 100
diff_q1 = scores_dur['reg_median_size'] * 100 - scores_dur['reg_q1_size'] * 100
ax.errorbar(scores_dur['significance'], scores_dur['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(scores_dur['significance'], scores_dur['reg_median_size'] * 100)
ax.plot(scores_stp['significance'], ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average duration of stay')

ax = axs[1, 0]
diff_q3 = scores_entry['reg_q3_size'] * 100 - scores_entry['reg_median_size'] * 100
diff_q1 = scores_entry['reg_median_size'] * 100 - scores_entry['reg_q1_size'] * 100
ax.errorbar(scores_entry['significance'], scores_entry['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(scores_entry['significance'], scores_entry['reg_median_size'] * 100)
ax.plot(scores_stp['significance'], ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average time of entry')
ax.set_ylabel('Interval width [%]')

ax = axs[1, 1]
diff_q3 = scores_occstp['reg_q3_size'] * 100 - scores_occstp['reg_median_size'] * 100
diff_q1 = scores_occstp['reg_median_size'] * 100 - scores_occstp['reg_q1_size'] * 100
ax.errorbar(scores_occstp['significance'], scores_occstp['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(scores_occstp['significance'], scores_occstp['reg_median_size'] * 100)
ax.plot(scores_stp['significance'], ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average daytime occupancy')
ax.set_xlabel('Significance level εₜ')

ax = axs[2, 0]
diff_q3 = scores_rev['reg_q3_size'] * 100 - scores_rev['reg_median_size'] * 100
diff_q1 = scores_rev['reg_median_size'] * 100 - scores_rev['reg_q1_size'] * 100
ax.errorbar(scores_rev.index, scores_rev['reg_median_size'] * 100, yerr=[diff_q1, diff_q3], fmt='--o')
ax.scatter(scores_rev.index, scores_rev['reg_median_size'] * 100)
ax.plot(scores_stp['significance'], ideal, '--', color='k', linewidth=0.5)
ax.set_title('Average daily revenue')
ax.set_ylabel('Interval width [%]')
ax.set_xlabel('Significance level εₜ')

fig.delaxes(axs[2, 1])
fig.suptitle('Tightness of prediction regions per target ')

plt.show()

# Plot empirical validity
fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(10, 10))

ax = axs[0, 0]
ax.scatter(scores_stp['significance'], scores_stp['reg_mean_errors'])
ax.plot(scores_stp['significance'], scores_stp['reg_mean_errors'])
ax.plot(scores_stp['significance'], scores_stp['significance'], '--', color='k')
ax.set_title('Average daily sold parking hours')
ax.set_ylabel('Error rate')

ax = axs[0, 1]
ax.scatter(scores_dur['significance'], scores_dur['reg_mean_errors'])
ax.plot(scores_dur['significance'], scores_dur['reg_mean_errors'])
ax.plot(scores_dur['significance'], scores_dur['significance'], '--', color='k')
ax.set_title('Average duration of stay')

ax = axs[1, 0]
ax.scatter(scores_entry['significance'], scores_entry['reg_mean_errors'])
ax.plot(scores_entry['significance'], scores_entry['reg_mean_errors'])
ax.plot(scores_entry['significance'], scores_entry['significance'], '--', color='k')
ax.set_title('Average time of entry')
ax.set_ylabel('Error rate')

ax = axs[1, 1]
ax.scatter(scores_occstp['significance'], scores_occstp['reg_mean_errors'])
ax.plot(scores_occstp['significance'], scores_occstp['reg_mean_errors'])
ax.plot(scores_occstp['significance'], scores_occstp['significance'], '--', color='k')
ax.set_title('Average daytime occupancy')
ax.set_xlabel('Significance level εₜ')

ax = axs[2, 0]
ax.scatter(scores_rev.index, scores_rev['reg_mean_errors'])
ax.plot(scores_rev.index, scores_rev['reg_mean_errors'])
ax.plot(scores_rev.index, scores_rev.index, '--', color='k')
ax.set_title('Average daily revenue')
ax.set_ylabel('Error rate')
ax.set_xlabel('Significance level εₜ')

fig.delaxes(axs[2, 1])
fig.suptitle('Empirical validity per target')

plt.show()
