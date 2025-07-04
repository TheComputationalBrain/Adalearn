"""
This script evaluate pupil effects on learning rate and the mediating influence of pupil responses on uncertainty-associated effects on learning rates for participants undertaking the Pupil task in the adalearn project. It computes the coefficients for the mediating baseline-corrected and non-baseline-corrected pupil responses at a participant level, before evaluating the coefficients at a group level. The results are saved in .csv files for further analysis. The following script need to be run before this script to obtain the relevant .csv files:
- adalearn_pupil_group_level_participant_demographics.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_behav_task_performance.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_model_parameters.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_learning_rate_analysis.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_assessment.py
- adalearn_pupil/adalearn_pupil_group_level_analysis_pupil_peristimulus_regression.py

"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.metrics import mean_squared_error
import pingouin
import statsmodels.api as sm

import adalearn_analysis_group_level_functions_data_utilities as group_data_utils
import adalearn_analysis_group_level_functions_pupil_analysis as group_pupil

from adalearn_pupil.adalearn_pupil_participant_info_and_parameters import base_dir, condition_str, exclude_zero



dataframes = group_data_utils.load_sample_csvs(
    directory_path=base_dir, 
    substrings=[
        'exclude_participants_behav_cor_df',
        'exclude_participants_pupil_analysis_df', 
        'computational_var_df', 
        'change_point_indices_df', 
        'outcome_epochs_descriptive_stats_df', 
        'outcome_epoched_pupil_df',
        'outcome_epochs_columns_df', 
        'outcome_epochs_to_remove_df',
        'outcome_epochs_descriptive_stats_nobc_df', 
        'outcome_epoched_pupil_nobc_df', 
        'cluster_permutation_analysis', 
        'cluster_permutation_analysis_nobc', 
        'uncertainty_params_ttest_df'],
    prefix='sample_level_analysis-ses_01-task_Adalearn_Pupil-')

participant_list = group_data_utils.get_participant_numbers(directory=Path(f'{base_dir}/data'))

excluded_participants = list(set(
    list(dataframes['exclude_participants_behav_cor_df']['participant']) +
    list(dataframes['exclude_participants_pupil_analysis_df']['exclude_participant'])
))

dataframe_list = []
pupil_bc_effect = []
pupil_nobc_effect =[]
pupil_bc_around_cp_list = []
pupil_nobc_around_cp_list = []
pupil_v_lr_regress_coef_df_list = []
mediation_results_list = []

# Setting the variables for the mediation analysis
uncertainty_on_lr_var = dataframes['uncertainty_params_ttest_df']
excl_last = uncertainty_on_lr_var.iloc[:-1]
uncert_on_lr_sig_vars = excl_last[excl_last['p_value'] < 0.05]
significant_var_names = uncert_on_lr_sig_vars['variable'].tolist()
mapped_names = []
for var in significant_var_names:
    if var == 'conf_zscore':
        mapped_names.append('prior_conf')
    elif var == 'cpp_zscore':
        mapped_names.append('cpp')
 

# Running for all participants then excluding those identified due to excessive interpolation 
for participant in participant_list:
    if participant not in excluded_participants:
        participant_str = str(participant).zfill(2)
        print(f"\n\n\nProcessing participant {participant_str}...")
        directory_path = Path(f'{base_dir}/data/sub-{participant_str}')
    else:
        continue

    # Getting individual participant data
    comp_df = dataframes['computational_var_df'][dataframes['computational_var_df']['participant']==participant].copy()
    comp_df = comp_df.reset_index()

    change_point_indices = dataframes['change_point_indices_df'][dataframes['change_point_indices_df']['participant']==participant].copy()

    epoch_data = dataframes['outcome_epoched_pupil_df'][dataframes['outcome_epoched_pupil_df']['participant'] == participant]

    epoch_nobc_data = dataframes['outcome_epoched_pupil_nobc_df'][dataframes['outcome_epoched_pupil_nobc_df']['participant'] == participant]

    epoch_descriptive_stats = dataframes['outcome_epochs_descriptive_stats_df'][dataframes['outcome_epochs_descriptive_stats_df']['participant']==participant]

    epoch_nobc_descriptive_stats = dataframes['outcome_epochs_descriptive_stats_nobc_df'][dataframes['outcome_epochs_descriptive_stats_nobc_df']['participant']==participant]

    analysed_epochs = dataframes['outcome_epochs_columns_df'][dataframes['outcome_epochs_columns_df']['participant']==participant]

    epochs_to_remove = dataframes['outcome_epochs_to_remove_df'][dataframes['outcome_epochs_to_remove_df']['participant']==participant]
    
    # Processing epoched dataframes
    column_titles = list(analysed_epochs['columns'].values.astype(int))

    if len(column_titles)+len(epochs_to_remove) == len(comp_df):
        print(f'Correct number of epochs imported for participant {participant_str}')
    else:
        print(f'Incorrect number of epochs imported for participant {participant_str}')

    epoch_data = epoch_data.drop(columns=['participant'])
    epoch_data = epoch_data.T
    epoch_data.columns = column_titles
    epoch_data['time'] = epoch_descriptive_stats['time'].values.astype(float)

    epoch_nobc_data = epoch_nobc_data.drop(columns=['participant'])
    epoch_nobc_data = epoch_nobc_data.T
    epoch_nobc_data.columns = column_titles
    epoch_nobc_data['time'] = epoch_nobc_descriptive_stats['time'].values.astype(float)

    # Extracting cluster times to set window for regression and mediation analyses
    baseline_cor_start_time, baseline_cor_end_time = group_pupil.extract_cluster_times(
        dataframes['cluster_permutation_analysis'],
        epoch_data['time']
    )

    nobc_start_time, nobc_end_time = group_pupil.extract_cluster_times(
        dataframes['cluster_permutation_analysis_nobc'],
        epoch_nobc_data['time']
    )

    # Convert to list (if not already)
    epochs_to_remove_list = epochs_to_remove['epochs'].tolist()

    # Remove rows from the computational variable dataframe where the row has been taken out of the pupil data due to excessive interpolation
    data_comp_filtered = comp_df[~comp_df.index.isin(epochs_to_remove_list)]

    conf_zscore = stats.zscore(data_comp_filtered['prior_conf'], nan_policy='omit')
    cpp_zscore = stats.zscore(data_comp_filtered['cpp'], nan_policy='omit')
    lr_zscore = stats.zscore(data_comp_filtered['cleaned_learning_rate_estimate'], nan_policy='omit')

    # Preparing the pupil data
    epoch_data = epoch_data.set_index('time')

    # Baseline corrected response refers to the average of the pupil signal over the window of a significant effect from the peristimulus regression
    filtered_pupil_data_bc = epoch_data.loc[(epoch_data.index >=  baseline_cor_start_time) & (epoch_data.index <=  baseline_cor_end_time)]
    pupil_variable_bc = np.array(filtered_pupil_data_bc.mean())
    pupil_bc_zscore = stats.zscore(pupil_variable_bc, nan_policy='omit')

    avg_pupil_bc_effect = pd.DataFrame({
        'participant': [participant_str],
        'avg_pupil_effect': [pupil_variable_bc.mean()]
    })

    pupil_bc_effect.append(avg_pupil_bc_effect)

    epoch_nobc_data = epoch_nobc_data.set_index('time')

    # The non-baseline corrected response refers to the average of the pupil signal from the beginning of the observed peristimulus effect to outcome presentation
    filtered_pupil_data_nobc = epoch_nobc_data.loc[(epoch_nobc_data.index >= nobc_start_time) & (epoch_nobc_data.index <= 0)]
    pupil_variable_nobc = np.array(filtered_pupil_data_nobc.mean())
    pupil_nobc_zscore = stats.zscore(pupil_variable_nobc, nan_policy='omit')

    avg_pupil_nobc_effect = pd.DataFrame({
        'participant': [participant_str],
        'avg_pupil_effect': [pupil_variable_nobc.mean()]
    })

    pupil_nobc_effect.append(avg_pupil_nobc_effect)

    # Collating a regression variable dataframe
    print('\nCreating regression variable dataframe...')

    regression_df = pd.DataFrame({
        'conf': data_comp_filtered['prior_conf'].values,
        'conf_zscore': conf_zscore,
        'cpp': data_comp_filtered['cpp'].values,
        'cpp_zscore': cpp_zscore,
        'lr': data_comp_filtered['cleaned_learning_rate_estimate'].values,
        'lr_zscore': lr_zscore,
        'pupil_bc': pupil_variable_bc,
        'pupil_bc_zscore': pupil_bc_zscore,
        'pupil_nobc': pupil_variable_nobc,
        'pupil_nobc_zscore': pupil_nobc_zscore,
        'participant': participant_str
    })

    # Removing rows associated with null values in the cleaning of learning rate
    regression_df_filtered = regression_df.dropna(subset=['lr_zscore'])

    # Excluding zero update trials (optional)
    if exclude_zero:
        before_zero_filter = len(regression_df_filtered)
        regression_df_filtered = regression_df_filtered[regression_df_filtered['lr'] != 0]
        removed_zero = before_zero_filter - len(regression_df_filtered)
        print(f"Removed {removed_zero} rows where learning rate update = 0")
        # Re z-scoring the data after zero updates removed
        zscore_cols = ['conf', 'cpp', 'lr', 'pupil_bc', 'pupil_nobc']
        for col in zscore_cols:
            regression_df_filtered[f'{col}_zscore'] = stats.zscore(regression_df_filtered[col], nan_policy='omit')

    dataframe_list.append(regression_df_filtered)

    # Examining pupil response around change points
    print(f'\nExamining pupil responses around change points...')
    pupil_bc_effect_around_cp_df = group_pupil.pupil_response_around_change_point(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=epoch_data, change_point_indices=change_point_indices, baseline_correct_window=[baseline_cor_start_time, baseline_cor_end_time], nobc_window=[nobc_start_time, nobc_end_time], before=5, after=5, estimate_type='baseline_corrected_effect', plot_pupil_around_change_point=False)

    pupil_bc_effect_around_cp_df['participant'] = participant_str

    pupil_nobc_effect_around_cp_df = group_pupil.pupil_response_around_change_point(directory_path=directory_path, participant_str=participant_str, condition_str=condition_str, outcome_epoched_pupil_df=epoch_nobc_data, change_point_indices=change_point_indices, baseline_correct_window=[baseline_cor_start_time, baseline_cor_end_time], nobc_window=[nobc_start_time, nobc_end_time], before=5, after=5, estimate_type='nonbaseline_corrected_effect', plot_pupil_around_change_point=False)

    pupil_nobc_effect_around_cp_df['participant'] = participant_str

    pupil_bc_around_cp_list.append(pupil_bc_effect_around_cp_df)
    pupil_nobc_around_cp_list.append(pupil_nobc_effect_around_cp_df)


    # Running regression analysis for pupil responses on learning rates
    print(f'\nRunning regression analysis for pupil responses on learning rate...')
    X = regression_df_filtered[['pupil_bc_zscore', 'pupil_nobc_zscore']]
    y = regression_df_filtered['lr_zscore']

    # Add intercept term
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Extract coefficients and model fit statistics
    coefficients = model.params.iloc[1:].values  
    intercept = model.params.iloc[0]  
    r_squared = model.rsquared
    p_values = model.pvalues.values
    std_err_mean = model.bse.values  
    conf_int = model.conf_int().values  

    # Calculate MSE and RMSE
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Create a DataFrame for regression and correlation results
    pupil_v_lr_df = pd.DataFrame({
        "variable": ["Intercept"] + X.columns[1:].tolist(),  # Include predictor names
        "coefficient": [intercept] + coefficients.tolist(),
        "p_value": p_values.tolist(),
        "std_err": std_err_mean.tolist(),
        "ci_lower": conf_int[:, 0].tolist(),
        "ci_upper": conf_int[:, 1].tolist(),
        "participant": participant_str
    })

    pupil_v_lr_regress_coef_df_list.append(pupil_v_lr_df)

    # Running mediation analysis
    print(f'Running mediation analysis...')
    mediation_analysis = pingouin.mediation_analysis(data=regression_df_filtered, x='conf_zscore', m=['pupil_nobc_zscore'], y='lr_zscore', covar=None, alpha=0.05, n_boot=500, seed=None, return_dist=False, logreg_kwargs=None)

    mediation_analysis['participant'] = participant_str

    mediation_results_list.append(mediation_analysis)

group_dfs = pd.concat(dataframe_list, ignore_index=True)
group_pupil_bc_avg_effect_signal_df = pd.concat(pupil_bc_effect, ignore_index=True)
group_pupil_nobc_avg_effect_signal_df = pd.concat(pupil_nobc_effect, ignore_index=True)
group_pupil_bc_around_cp_df = pd.concat(pupil_bc_around_cp_list, ignore_index=True)
group_pupil_nobc_around_cp_df = pd.concat(pupil_nobc_around_cp_list, ignore_index=True)
group_pupil_v_lr_dfs = pd.concat(pupil_v_lr_regress_coef_df_list, ignore_index=True)
group_mediation_df = pd.concat(mediation_results_list, ignore_index=True)

# Running t-tests on pupil on learning rate regression 
pupil_v_lr_bc_coef = group_pupil_v_lr_dfs[group_pupil_v_lr_dfs['variable'] == 'pupil_bc_zscore'].sort_values('participant').copy()

pupil_v_lr_nobc_coef = group_pupil_v_lr_dfs[group_pupil_v_lr_dfs['variable'] == 'pupil_nobc_zscore'].sort_values('participant').copy()

pupil_v_lr_bc_coef_mean = pupil_v_lr_bc_coef['coefficient'].mean()
pupil_v_lr_nobc_coef_mean = pupil_v_lr_nobc_coef['coefficient'].mean()
pupil_v_lr_bc_coef_sem = stats.sem(pupil_v_lr_bc_coef['coefficient'])
behav_v_lr_nobc_coef_sem = stats.sem(pupil_v_lr_nobc_coef['coefficient'])

# t-tests against zero and between coefficients
t_bc_pupil = stats.ttest_1samp(pupil_v_lr_bc_coef['coefficient'], 0)
t_nobc_pupil = stats.ttest_1samp(pupil_v_lr_nobc_coef['coefficient'], 0)
t_between_coef = stats.ttest_rel(pupil_v_lr_bc_coef['coefficient'], pupil_v_lr_nobc_coef['coefficient'], alternative='two-sided')

pupil_v_lr_ttest_df = pd.DataFrame({
    'variable': ['pupil_bc', 'pupil_nobc', 'between_pupil_coef'],
    'regression_coef': [pupil_v_lr_bc_coef_mean, pupil_v_lr_nobc_coef_mean, None],
    'regress_coef_sem': [pupil_v_lr_bc_coef_sem, behav_v_lr_nobc_coef_sem, None],
    't_statistic': [t_bc_pupil.statistic, t_nobc_pupil.statistic, t_between_coef.statistic],
    'p_value': [t_bc_pupil.pvalue, t_nobc_pupil.pvalue, t_between_coef.pvalue]
})

print(pupil_v_lr_ttest_df)

# Collating the mediation analysis
group_mediation_analysis_df = pd.DataFrame()

n_boot = 5000 

for path in group_mediation_df['path'].unique():
    data = group_mediation_df[group_mediation_df['path'] == path]
    coefs = data['coef'].values
    mean = data['coef'].mean()
    path_label = data['path'].values[0]

    # Bootstrap sampling
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(coefs, size=len(coefs), replace=True)
        boot_means.append(np.mean(sample))

    # Get 95% CI from bootstrap distribution
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    boot_mean = np.mean(boot_means)
    boot_sem = np.std(boot_means, ddof=1)

    df_mediation = pd.DataFrame({
        'path': [path_label],
        'sample_mean': [mean],
        'boot_mean': [boot_mean],
        'sem': [boot_sem],
        'ci_lower': [ci_lower],
        'ci_upper': [ci_upper]
    })

    group_mediation_analysis_df = pd.concat([group_mediation_analysis_df, df_mediation], ignore_index=True)

print(group_mediation_analysis_df)

# Saving dataframes
group_data_utils.save_sample_dataframes_to_csv(dataframes={
    'average_bc_pupil_effect_signal': group_pupil_bc_avg_effect_signal_df,
    'average_nobc_pupil_effect_signal': group_pupil_nobc_avg_effect_signal_df,
    'pupil_v_lr_regress_coef_df': group_pupil_v_lr_dfs,
    'pupil_v_lr_coef_ttest_df': pupil_v_lr_ttest_df,
    'pupil_bc_around_cp': group_pupil_bc_around_cp_df,
    'pupil_nobc_around_cp': group_pupil_nobc_around_cp_df,
    'pupil_lr_uncert_mediation': group_mediation_analysis_df
},
directory_path=base_dir,
condition_str=condition_str,
task='Pupil')