import random
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns
import road_data_manipulation_v2 as rdm
import datetime
import aux_functions as aux
import helper
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Feature scaling
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import numpy as np
from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin
from sklearn.ensemble import RandomForestRegressor
from functools import partial
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import warnings
from IPython import embed
import sage
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_data(destination, dates):

    merged = pd.DataFrame()

    for date in dates:
        filename = f'{destination}/{date}_to_{date}_merged.h5'
        merged_date = pd.read_hdf(filename)
        merged = merged.append(merged_date)

    # Removing pings while reversing
    merged_forward = merged[merged.Forward > 0].copy()

    # Quantity is stored per trip and not per trip type. Hence we reset quantity to 0 for all type 0 and 2 to 0
    index = merged_forward.Type.isin([0, 2])
    merged_forward.loc[index, 'Quantity'] = 0

    return merged_forward


def get_X_feat():
    X_feat_all = ['LengthDistance',  # 'LengthTime',
                  'Quantity', 'Type',
                  'AltitudeGain', 'AltitudeLoss', 'AltitudeDeltaEndStart', 'AltitudeChange',
                  'AltitudeLossPart', 'AltitudeGainPart',
                  'AltitudeDeltaMaxMin',
                  #   'MAX_speedMean_altitudeLossPart', 'MAX_speedMean_altitudeGainPart',
                  'Sum_RotationXAltitudeDiffPos', 'Sum_RotationXAltitudeDiffNeg',
                  'UpInclination', 'DownInclination',
                  'DownInclinationPart', 'UpInclinationPart',
                  # 'Fuel',
                  'AccTime',
                  'SpeedMean',  # 'SpeedVariance',
                  #'SpeedDiffPositiveSum', 'SpeedDiffNegativeSum',
                  #'IdlingTime', 'SumRotation',
                  'SumRotation', 'TypeTripLogId', 'DistanceFullyRoute', 'ControlStartTime', 'ControlStartTimeClock', 'outliers'
                  ]
    # X_feat = [
    #     'Quantity',
    #     # 'AltitudeChange',  # 'AltitudeLoss',
    #     'AltitudeDeltaEndStart',  'AltitudeGain',
    #     # 'AltitudeDeltaMaxMin',
    #     # 'UpInclination',  # 'DownInclination',
    #     'AltitudeGainPart',  # 'AltitudeLossPart',
    #     # 'Fuel',
    #     'AccTime',
    #     'SpeedMean',  # 'LengthDistance', # 'SpeedVariance',
    #     # 'SumRotation'
    # ]

    X_feat = [
        'Quantity',
        # 'AltitudeChange',  # 'AltitudeLoss',
        'AltitudeDeltaEndStart',  'AltitudeGain',
        # 'AltitudeDeltaMaxMin',
        'UpInclination',  # 'DownInclination',
        # 'AltitudeGainPart',  # 'AltitudeLossPart',
        # 'Fuel',
        'AccTime',
        'SpeedMean',  # 'LengthDistance',
        # 'SumRotation'
    ]

    return X_feat_all, X_feat


def get_train_test_val_set(statdata, destination_val, dates_val, psedo_random=True, re_use_data=True, sub_name='half_unique'):
    # Selecting some control values, for control of bad predictions

    X_feat_all, X_feat = get_X_feat()

    y_feat = ['Fuel']
    X_train_fact, X_test_fact, y_train, y_test = train_test_split(
        statdata[X_feat_all], statdata[y_feat], test_size=0.20, random_state=42)

    # The actual features the model will take as input

    X_train = X_train_fact[X_feat]
    X_test = X_test_fact[X_feat]

    # Feature scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train_scaled = pd.DataFrame(
        sc_X.fit_transform(X_train.values), columns=X_feat)
    y_train_scaled = pd.DataFrame(sc_y.fit_transform(
        y_train.values), columns=y_feat).values
    X_test_scaled = pd.DataFrame(sc_X.transform(X_test.values), columns=X_feat)
    y_test_scaled = pd.DataFrame(sc_y.transform(y_test.values), columns=y_feat)
    y_train = y_train.values

    index = y_test_scaled.sort_values('Fuel').index
    X_test_scaled_sorted = X_test_scaled.iloc[index]

    y_test_scaled_sorted = y_test_scaled.iloc[index].values.flatten()

    # TODO: unscaled variants - may delete
    y_test_sorted = y_test.iloc[index].values.flatten()
    X_test_sorted = X_test.iloc[index]
    X_test_fact_sorted = X_test_fact.iloc[index]

    ################################################################
    ######################### Validation ###########################
    ################################################################

    merged_ = pd.DataFrame()

    for date in dates_val:
        filename = f'{destination_val}/{date}_to_{date}_merged.h5'
        merged_date = pd.read_hdf(filename)
        merged_ = merged_.append(merged_date)

    # Copying the same as above
    merged_forward_ = merged_[merged_.Forward > 0].copy()
    index_q = merged_forward_.Type.isin([0, 2])
    merged_forward_.loc[index_q, 'Quantity'] = 0

    # Check distribution of trips with only forward motion:
    merged_forward_['TypeTripLogId'] = merged_forward_[
        'TripLogId'] + '_' + merged_forward_['subtrack'].astype('int').astype('str')

    # Run the sampling and select some trips to plot
    if not re_use_data:
        TripTypes = [0, 1, 2]
        TypeTripLogIds = None
        statdata_val = helper.sample_trips(merged_forward_, trip_types=TripTypes,
                                           minlength=30,
                                           maxlength=30,
                                           Nsamples=100,
                                           TypeTripLogIds=TypeTripLogIds,
                                           idling_threshold=1,
                                           acceleration_threshold=0.01,
                                           altitude_threshold=.01,
                                           Nplots=0,
                                           half_unique=True,
                                           fully_unique=False,
                                           psedo_random=psedo_random)
    else:
        statdata_val, _ = helper.get_df_of_statistics(dates_val, sub_name)

    X_feat_all, X_feat = get_X_feat()

    X_val_fact = statdata_val[X_feat_all]
    X_val = X_val_fact[X_feat]
    y_val = statdata_val[y_feat]
    index_1 = y_val.sort_values('Fuel').index

    X_val_scaled = pd.DataFrame(sc_X.transform(X_val.values), columns=X_feat)
    y_val_scaled = pd.DataFrame(sc_y.transform(y_val.values), columns=y_feat)

    X_val_scaled_sorted = X_val_scaled.iloc[index_1]
    y_val_scaled_sorted = y_val_scaled.iloc[index_1].values.flatten()

    y_val_sorted = y_val.iloc[index_1].values.flatten()
    X_val_sorted = X_val.iloc[index_1]
    X_val_fact_sorted = X_val_fact.iloc[index_1]

    return X_train, X_test_sorted, X_val_sorted, X_val_fact_sorted, y_train, y_test_sorted, y_val_sorted


def make_dirs(destination):

    path1 = destination + '/error_vs_iteration/'
    path2 = destination + '/prediction_plot_validation/'
    path3 = destination + '/models/'
    path4 = destination + '/sage/'
    path5 = destination + '/importance_values/'

    # Check whether the specified path exists or not
    isExist = os.path.exists(destination)
    isExist1 = os.path.exists(path1)
    isExist2 = os.path.exists(path2)
    isExist3 = os.path.exists(path3)
    isExist4 = os.path.exists(path4)
    isExist5 = os.path.exists(path5)

    if not isExist:
        os.makedirs(destination_save)
    if not isExist1:
        os.makedirs(path1)
    if not isExist2:
        os.makedirs(path2)
    if not isExist3:
        os.makedirs(path3)
    if not isExist4:
        os.makedirs(path4)
    if not isExist5:
        os.makedirs(path5)

    create_file = open(f'{destination}/information.txt', 'a+')
    create_file.close()

    _, X_feat = get_X_feat()

    feature_information = open(f'{destination}/features.txt', 'w')
    for feat in X_feat:
        feature_information.write(feat + '\n')

    feature_information.close()


def train_XGB_model(X_train, X_test_sorted, X_val_sorted, X_val_fact_sorted, y_train, y_test_sorted, y_val_sorted, destination, val_date):

    seed = 7

    def objective_xgb(params, train_X, val_X, train_y, val_y):
        est = int(params['n_estimators'])
        md = int(params['max_depth'])
        learning = params['learning_rate']

        model = xgb.XGBRegressor(
            n_estimators=est, max_depth=md, learning_rate=learning, random_state=42, seed=42)
        model.fit(train_X, train_y)
        pred = model.predict(val_X)

        score = mean_squared_error(val_y, pred)
        return score

    def optimize_xgb(trial, train_X, val_X, train_y, val_y):
        params = {'n_estimators': hp.uniform('n_estimators', 2, 500),
                  'max_depth': hp.uniform('max_depth', 2, 20),
                  'learning_rate': hp.uniform('learning_rate', 0.001, 0.1)}

        fmin_objective = partial(objective_xgb, train_X=train_X, val_X=val_X,
                                 train_y=train_y, val_y=val_y)

        rstate = np.random.default_rng(42)
        best = fmin(fn=fmin_objective,
                    space=params,
                    algo=tpe.suggest,
                    trials=trial,
                    max_evals=20,
                    rstate=rstate
                    )
        return best

    trial = Trials()
    best_xgb = optimize_xgb(trial, X_train, X_test_sorted,
                            y_train.ravel(), y_test_sorted.ravel())

    # TODO: hyperparameters used?? - maybe nothing to do with
    eval_set = [(X_train, y_train), (X_test_sorted, y_test_sorted)]

    model_xgb = xgb.XGBRegressor(
        n_estimators=int(best_xgb['n_estimators']),
        max_depth=int(best_xgb['max_depth']),
        learning_rate=best_xgb['learning_rate'],
        early_stopping_rounds=0,
        random_state=42,
        seed=42
    )

    # model_xgb = xgb.XGBRegressor(random_state = 0)
    # fit the regressor with x and y data

    model_xgb = model_xgb.fit(X_train, y_train.ravel(), eval_set=eval_set,
                              verbose=False)

    results = model_xgb.evals_result()

    y_pred_xgb_test = model_xgb.predict(X_test_sorted)
    y_pred_xgb_train = model_xgb.predict(X_train)
    y_pred_xgb_val = model_xgb.predict(X_val_sorted)

    plt.plot(results['validation_0']['rmse'], label="training")
    plt.plot(results['validation_1']['rmse'], label="validation")
    plt.legend()
    plt.savefig(destination + '/error_vs_iteration/' + val_date + '.png')
    plt.close()

    filename = destination + '/models/' + 'without_' + date + '.pkl'
    pickle.dump(model_xgb, open(filename, "wb"))

    # max_observations = min(len(y_pred_xgb_val), 70)
    index_included = range(len(y_pred_xgb_val))
    # index_included = np.sort(np.random.choice(
    #     range(len(y_pred_xgb_val)), max_observations, replace=False))

    outliers = X_val_fact_sorted.outliers[index_included]
    index_of_outliers = np.array(
        list(set(np.where(outliers)[0]).intersection(index_included)))
    index_of_outliers = []

    if len(index_of_outliers) > 0:
        plt.scatter(index_of_outliers, y_pred_xgb_val[index_included]
                    [index_of_outliers], marker='x', color='red', label='outliers')

    plt.plot(y_pred_xgb_val[index_included], linestyle=None, label="Predicted")
    plt.plot(y_val_sorted[index_included], linestyle=None, label="actual")
    plt.legend()
    plt.axhline(0)
    plt.savefig(destination + '/prediction_plot_validation/' +
                val_date + '.png')
    plt.close()

    mse_xgb_test = mean_squared_error(y_pred_xgb_test, y_test_sorted)
    mse_xgb_train = mean_squared_error(y_pred_xgb_train, y_train.ravel())
    mse_xgb_val = mean_squared_error(y_pred_xgb_val, y_val_sorted)

    with open(f'{destination}/information.txt', 'r') as f:
        contents = f.readlines()

    infile = open(f'{destination}/information.txt', 'r')
    insert_index = 0
    replacing = False

    insert_text = date + \
        f' -- Test: {mse_xgb_val: .5f}, Val: {mse_xgb_test: .5f}, Train: {mse_xgb_train: .5f} \n'
    for i, line in enumerate(infile):

        if len(line.split()) == 0:
            break

        file_date_str = line.split()[0]
        file_date_time = datetime.datetime(
            *[int(t) for t in file_date_str.split('-')])
        date_time = datetime.datetime(*[int(t) for t in date.split('-')])

        if date_time < file_date_time:
            break

        if file_date_str == date:
            replacing = True
            break

        insert_index += 1

    contents.insert(insert_index, insert_text)
    if replacing:
        contents.pop(insert_index + 1)
    infile.close()

    with open(f'{destination}/information.txt', 'w') as f:
        contents = "".join(contents)
        f.write(contents)

    _, X_feat = get_X_feat()
    # Set up an imputer to handle missing features
    imputer = sage.MarginalImputer(model_xgb, X_test_sorted.values)

    # Set up an estimator
    estimator = sage.PermutationEstimator(imputer, 'mse')

    # Calculate SAGE values
    sage_values = estimator(X_test_sorted.values, y_test_sorted)
    sage_values.plot(X_feat)
    plt.savefig(destination + '/sage/' + date + '_sage_test.png')
    plt.close()

    # WRITE IMPORTANCE VALUES TO FILES
    infile = open(
        destination + '/importance_values/sage_test_' + date + '.txt', 'w')
    for feat, val in zip(X_feat, sage_values.values):
        infile.write(feat + f' {val : .6f} \n')
    infile.close()

    # Set up an imputer to handle missing features
    imputer = sage.MarginalImputer(model_xgb, X_test_sorted.values)

    # Set up an estimator
    estimator = sage.PermutationEstimator(imputer, 'mse')
    # Calculate SAGE values
    sage_values = estimator(X_val_sorted.values, y_val_sorted)
    sage_values.plot(X_feat)

    plt.savefig(destination + '/sage/' + date + '_sage_val.png')
    plt.close()

    # WRITE IMPORTANCE VALUES TO FILES
    infile = open(destination + '/importance_values/sage_val_' +
                  date + '.txt', 'w')
    for feat, val in zip(X_feat, sage_values.values):
        infile.write(feat + f' {val : .6f} \n')
    infile.close()

    infile = open(
        destination + '/importance_values/feature_importances_' + date + '.txt', 'w')
    for feat, val in zip(X_feat, model_xgb.feature_importances_):
        infile.write(feat + f' {val : .6f} \n')
    infile.close()


if __name__ == '__main__':
    print(5.6)
    destination_of_merged_data = helper.PATH_DATA + "raw_2809"
    np.random.seed(42)
    random.seed(42)

    dates = [
        '2021-09-28',
        '2021-09-29',
        '2021-10-13',
        '2021-10-14',
        '2021-10-15',
        '2021-10-18',
        '2021-10-19',
        '2021-10-20',
        # '2021-10-21',
    ]

    settings = {'num': 23,
                're_use_data': True,
                'half_unique': False,
                'fully_unique': True,
                'sub_name': 'fully_unique'}

    # helper.set_outliers(sub_name = settings['sub_name'])
    destination_save = helper.PATH_QUICK_RUN + \
        f'models_and_results{settings["num"]}'

    make_dirs(destination_save)
    dates_copy = dates.copy()
    for i, date in enumerate(tqdm(dates)):

        pop_date = dates_copy.pop(i)

        if not settings['re_use_data']:
            merged_forward = get_data(destination_of_merged_data, dates_copy)
            statdata = helper.sample_trips(merged_forward, trip_types=[0, 1, 2],
                                           minlength=30,
                                           maxlength=30,
                                           Nsamples=1000,
                                           TypeTripLogIds=None,
                                           idling_threshold=1,
                                           acceleration_threshold=0.01,
                                           altitude_threshold=.01,
                                           Nplots=0,
                                           half_unique=settings['half_unique'],
                                           fully_unique=settings['fully_unique'],
                                           psedo_random=True)
        else:
            statdata, _ = helper.get_df_of_statistics(
                dates_copy, sub_name=settings['sub_name'])

        validation_dates = [pop_date]
        X_train, X_test_sorted, X_val_sorted, X_val_fact_sorted, y_train, y_test_sorted, y_val_sorted = get_train_test_val_set(statdata,
                                                                                                                               destination_of_merged_data, validation_dates, re_use_data=True, sub_name='fully_unique')

        train_XGB_model(X_train, X_test_sorted, X_val_sorted, X_val_fact_sorted,
                        y_train, y_test_sorted, y_val_sorted, destination_save, pop_date)

        dates_copy.insert(i, pop_date)

    helper.rank_features(settings['num'])
