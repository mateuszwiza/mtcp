"""
SCRIPT FOR PLOTTING THE RESULTS
OF THE COPULA-BASED CONFORMAL PREDICTION
"""

# Authors: Mateusz Wiza

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def read_json_hyper(copula, version):
    """Get data on hyper-rectangle volume and validity
       from copula-based conformal prediction.
        Parameters
        ----------
        copula : string e.g. 'independent', 'gumbel, or 'empirical'
                 Name of copula.
        version : string e.g. 'rf' or 'sst'
                  Abbreviated name of model.
        Returns
        -------
        DataFrame with results loaded from JSON
    """
    # Create path
    combination = 'results_' + copula + '_' + version
    filename1 = 'graph-data/copulas/' + combination + '_hyper_interval.json'
    filename2 = 'graph-data/copulas/' + combination + '_hyper_levels.json'

    # Open and load the file
    f_v = open(filename1, )
    f_l = open(filename2, )
    volumes = json.load(f_v)
    levels = json.load(f_l)

    # Define GLOBAL significance levels
    significances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data_list = []
    for sign in significances:
        # Get results from JSON file for given target and significance
        hyp_median_size = volumes['AVG'][str(sign)]['MEDIAN']['hypercube']
        hyp_q1_size = volumes['AVG'][str(sign)]['Q1']['hypercube']
        hyp_q3_size = volumes['AVG'][str(sign)]['Q3']['hypercube']
        hyp_validity = 100 - levels['AVG'][str(sign)]['hypercube']
        # Append results to a list
        data_list.append([cop, sign, hyp_validity, hyp_median_size, hyp_q1_size, hyp_q3_size])

    # Convert the list to DataFrame
    return pd.DataFrame(data_list, columns=['copula', 'significance', 'hyp_validity', 'hyp_median_size', 'hyp_q1_size',
                                            'hyp_q3_size'])


def read_json_target(copula, version):
    """Get data on target-wise interval width and validity
       from copula-based conformal prediction.
        Parameters
        ----------
        copula : string e.g. 'independent', 'gumbel, or 'empirical'
                 Name of copula.
        version : string e.g. 'rf' or 'sst'
                  Abbreviated name of model.
        Returns
        -------
        DataFrame with results loaded from JSON
    """
    # Create paths
    combination = 'results_' + copula + '_' + version
    filename1 = 'graph-data/copulas/' + combination + '_target_interval.json'
    filename2 = 'graph-data/copulas/' + combination + '_target_levels.json'

    # Open and load the files
    f_i = open(filename1, )
    f_l = open(filename2, )
    intervals = json.load(f_i)
    levels = json.load(f_l)

    # Define GLOBAL significance levels
    significances = [0.2262190625, 0.40951, 0.67232, 0.83193, 0.92224, 0.96875, 0.98976, 0.99757, 0.99968]
    # Get number of targets
    targets = len(levels['AVG']['0.40951']) - 1

    data_list = []
    for t in range(targets):
        for sign in significances:
            # Convert global significance to target-wise significance
            significance = 1 - pow(1 - sign, 1 / 5)
            # Get results from JSON file for given target and significance
            reg_mean_errors = levels['AVG'][str(sign)][str(t)]
            reg_median_size = intervals['AVG'][str(sign)]['MEDIAN'][str(t)]
            reg_q1_size = intervals['AVG'][str(sign)]['Q1'][str(t)]
            reg_q3_size = intervals['AVG'][str(sign)]['Q3'][str(t)]
            # Append results to a list
            data_list.append([t, significance, reg_mean_errors, reg_median_size, reg_q1_size, reg_q3_size])

    # Convert the list to DataFrame
    return pd.DataFrame(data_list,
                        columns=['output', 'significance', 'reg_mean_errors', 'reg_median_size', 'reg_q1_size',
                                 'reg_q3_size'])


# -----------------------------------------------------------------------------
# PLOT THE RESULTS
# -----------------------------------------------------------------------------

# 1. Volume of hyper-rectangle for all combinations

# Define copulas and models
copulas_list = ['independent', 'empirical', 'gumbel']
versions = ['rf', 'sst']

dfs_hyper = {}
for v in versions:
    for cop in copulas_list:
        # Format name to e.g. 'RF Gumbel'
        name = v.upper() + ' ' + cop.capitalize()
        # Read data from JSON and append DataFrame
        dfs_hyper[name] = read_json_hyper(cop, v)

# Create the plot
legend = []
for df in dfs_hyper:
    # Define colors
    if 'SST' in df:
        color = 'g'
    else:
        color = 'k'

    # Define line styles
    if 'Empirical' in df:
        style = '--'
    elif 'Gumbel' in df:
        style = '-.'
    else:
        style = '-'

    data = dfs_hyper[df]
    plt.scatter(data['significance'], data['hyp_median_size'] * 100, s=15, c=color)
    plt.plot(data['significance'], data['hyp_median_size'] * 100, style, c=color)
    legend.append(df)

plt.ylabel('Hyper Rectangle Volume (log)')
plt.xlabel('Global significance level εₕ')
plt.yscale("log")
plt.legend(legend)
plt.show()

# 2. Validity of hyper-rectangle for all combinations

# Define global significance levels and the 'ideal' width of 10% (for plots)
significance_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ideal = np.ones(10) * 10

# Generate 2 graphs, 1 per underlying model
for v in versions:
    legend = ['Calibration line']
    # Plot calibration line
    plt.plot(significance_levels, significance_levels, '--', color='r')

    for cop in copulas_list:
        # Define line styles
        if 'empirical' in cop:
            style = '--'
        elif 'gumbel' in cop:
            style = '-.'
        else:
            style = '-'

        # Get subset of data for current version and copula
        df_name = v.upper() + ' ' + cop.capitalize()
        data_temp = dfs_hyper[df_name]

        plt.scatter(data_temp['significance'], data_temp['hyp_validity'] / 100, color='k')
        plt.plot(data_temp['significance'], data_temp['hyp_validity'] / 100, style, color='k')

        name = cop.capitalize()
        legend.append(name)

    plt.ylabel('Error rate')
    plt.xlabel('Global significance level εₕ')
    plt.legend(legend)
    plt.show()

# 3. Target-wise interval width and validity for all targets
# Re-define list of copulas
copulas_list2 = ['empirical', 'gumbel']

for v in versions:
    for cop in copulas_list2:
        # Get the results from JSON
        results = read_json_target(v, cop)

        # Split results by target
        results_occstp = results[results['output'] == 0]
        results_stp = results[results['output'] == 1]
        results_rev = results[results['output'] == 2]
        results_dur = results[results['output'] == 3]
        results_entry = results[results['output'] == 4]

        # Plot prediction interval widths (median and IQR)
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

        ax = axs[0, 0]
        diff_q3 = results_stp['reg_q3_size'] * 100 - results_stp['reg_median_size'] * 100
        diff_q1 = results_stp['reg_median_size'] * 100 - results_stp['reg_q1_size'] * 100
        ax.errorbar(results_stp['significance'], results_stp['reg_median_size'] * 100, yerr=[diff_q1, diff_q3],
                    fmt='--o')
        ax.scatter(results_stp['significance'], results_stp['reg_median_size'] * 100)
        ax.plot(results_stp['significance'], ideal, '--', color='k', linewidth=0.5)
        ax.set_title('Average daily sold parking hours')
        ax.set_ylabel('Interval width [%]')

        ax = axs[0, 1]
        diff_q3 = results_dur['reg_q3_size'] * 100 - results_dur['reg_median_size'] * 100
        diff_q1 = results_dur['reg_median_size'] * 100 - results_dur['reg_q1_size'] * 100
        ax.errorbar(results_dur['significance'], results_dur['reg_median_size'] * 100, yerr=[diff_q1, diff_q3],
                    fmt='--o')
        ax.scatter(results_dur['significance'], results_dur['reg_median_size'] * 100)
        ax.plot(results_stp['significance'], ideal, '--', color='k', linewidth=0.5)
        ax.set_title('Average duration of stay')

        ax = axs[1, 0]
        diff_q3 = results_entry['reg_q3_size'] * 100 - results_entry['reg_median_size'] * 100
        diff_q1 = results_entry['reg_median_size'] * 100 - results_entry['reg_q1_size'] * 100
        ax.errorbar(results_entry['significance'], results_entry['reg_median_size'] * 100, yerr=[diff_q1, diff_q3],
                    fmt='--o')
        ax.scatter(results_entry['significance'], results_entry['reg_median_size'] * 100)
        ax.plot(results_stp['significance'], ideal, '--', color='k', linewidth=0.5)
        ax.set_title('Average time of entry')
        ax.set_ylabel('Interval width [%]')

        ax = axs[1, 1]
        diff_q3 = results_occstp['reg_q3_size'] * 100 - results_occstp['reg_median_size'] * 100
        diff_q1 = results_occstp['reg_median_size'] * 100 - results_occstp['reg_q1_size'] * 100
        ax.errorbar(results_occstp['significance'], results_occstp['reg_median_size'] * 100, yerr=[diff_q1, diff_q3],
                    fmt='--o')
        ax.scatter(results_occstp['significance'], results_occstp['reg_median_size'] * 100)
        ax.plot(results_stp['significance'], ideal, '--', color='k', linewidth=0.5)
        ax.set_title('Average daytime occupancy')
        ax.set_xlabel('Significance level εₜ')

        ax = axs[2, 0]
        diff_q3 = results_rev['reg_q3_size'] * 100 - results_rev['reg_median_size'] * 100
        diff_q1 = results_rev['reg_median_size'] * 100 - results_rev['reg_q1_size'] * 100
        ax.errorbar(results_rev['significance'], results_rev['reg_median_size'] * 100, yerr=[diff_q1, diff_q3],
                    fmt='--o')
        ax.scatter(results_rev['significance'], results_rev['reg_median_size'] * 100)
        ax.plot(results_stp['significance'], ideal, '--', color='k', linewidth=0.5)
        ax.set_title('Average daily revenue')
        ax.set_ylabel('Interval width [%]')
        ax.set_xlabel('Significance level εₜ')

        fig.delaxes(axs[2, 1])
        fig.suptitle('Tightness of prediction regions per target')

        plt.show()

        # Plot empirical validity
        fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(10, 10))

        ax = axs[0, 0]
        ax.scatter(results_stp['significance'], 1 - results_stp['reg_mean_errors'] / 100)
        ax.plot(results_stp['significance'], 1 - results_stp['reg_mean_errors'] / 100)
        ax.plot(results_stp['significance'], results['significance'], '--', color='k')
        ax.set_title('Average daily sold parking hours')
        ax.set_ylabel('Error rate')

        ax = axs[0, 1]
        ax.scatter(results_dur['significance'], 1 - results_dur['reg_mean_errors'] / 100)
        ax.plot(results_dur['significance'], 1 - results_dur['reg_mean_errors'] / 100)
        ax.plot(results_dur['significance'], results['significance'], '--', color='k')
        ax.set_title('Average duration of stay')

        ax = axs[1, 0]
        ax.scatter(results_entry['significance'], 1 - results_entry['reg_mean_errors'] / 100)
        ax.plot(results_entry['significance'], 1 - results_entry['reg_mean_errors'] / 100)
        ax.plot(results_entry['significance'], results['significance'], '--', color='k')
        ax.set_title('Average time of entry')
        ax.set_ylabel('Error rate')

        ax = axs[1, 1]
        ax.scatter(results_occstp['significance'], 1 - results_occstp['reg_mean_errors'] / 100)
        ax.plot(results_occstp['significance'], 1 - results_occstp['reg_mean_errors'] / 100)
        ax.plot(results_occstp['significance'], results['significance'], '--', color='k')
        ax.set_title('Average daytime occupancy')
        ax.set_xlabel('Significance level εₜ')

        ax = axs[2, 0]
        ax.scatter(results_rev['significance'], 1 - results_rev['reg_mean_errors'] / 100)
        ax.plot(results_rev['significance'], 1 - results_rev['reg_mean_errors'] / 100)
        ax.plot(results_rev['significance'], results['significance'], '--', color='k')
        ax.set_title('Average daily revenue')
        ax.set_ylabel('Error rate')
        ax.set_xlabel('Significance level εₜ')

        fig.delaxes(axs[2, 1])
        fig.suptitle('Empirical validity per target')

        plt.show()
