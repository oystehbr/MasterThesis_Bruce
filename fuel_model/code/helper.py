from sklearn.metrics import mean_squared_error
import seaborn as sns
import pickle
import sklearn.cluster as cluster
from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATES = [
    '2021-09-28',
    '2021-09-29',
    '2021-10-13',
    '2021-10-14',
    '2021-10-15',
    '2021-10-18',
    '2021-10-19',
    '2021-10-20',
    '2021-10-21',
    '2021-11-03',
    '2021-11-04',
    '2021-11-15',
]


PATH_CURR_DATA = "/Users/oysteinbruce/Documents/GitHub/SINTEF/fuel_model/data/raw_2809"
PATH_DATA = '/Users/oysteinbruce/Documents/Github/SINTEF/fuel_model/data/'
PATH_QUICK_RUN = '/Users/oysteinbruce/Documents/Github/SINTEF/fuel_model/quick_run_all/'
PATH_FIGURES = '/Users/oysteinbruce/Documents/Github/SINTEF/fuel_model/figures/'
PATH_PROJECT = '/Users/oysteinbruce/Documents/Github/SINTEF/fuel_model/'

X_FEAT_ALL = ['LengthDistance',  # 'LengthTime',
              'Quantity', 'Type',
              'AltitudeGain', 'AltitudeLoss', 'AltitudeDeltaEndStart', 'AltitudeChange',
              'AltitudeLossPart', 'AltitudeGainPart',
              'AltitudeDeltaMaxMin',
              'Sum_RotationXAltitudeDiffPos', 'Sum_RotationXAltitudeDiffNeg',
              'DownInclination', 'UpInclination',
              'DownInclinationPart', 'UpInclinationPart',
              # 'Fuel',
              # 'AccTime',
              'SpeedMean',  # 'SpeedVariance',
              # 'SpeedDiffPositiveSum', 'SpeedDiffNegativeSum',
              # 'IdlingTime', 'SumRotation',
              'SumRotation', 'TypeTripLogId', 'DistanceFullyRoute', 'ControlStartTime', 'ControlStartTimeClock'
              ]


def clean_engine_on(df, startstopdata):
    """
    Remove all data points when the engine is not on. The dataframes are assumed to cover
    the same time interval and the same machines (otherwise it would be more efficient to
    filter on those first). Also recommended to drop duplicates beforehand...
    """

    clean_df = pd.DataFrame()
    df['time_step'] = np.nan
    df['delta_consumption'] = np.nan
#    df['consumption_rate'] = np.nan

    for i in range(len(startstopdata)):
        index_serialNumber = df.serialNumber == startstopdata.iloc[i].serialNumber
        index_time = (df[index_serialNumber].moduleTime >= startstopdata.iloc[i].start) & \
                     (df[index_serialNumber].moduleTime <=
                      startstopdata.iloc[i].stop)
        if len(df[index_time]) > 0:
            df.loc[index_time, 'time_step'] = df.loc[index_time,
                                                     'moduleTime'].diff().dt.total_seconds()/60
            df.loc[index_time, 'delta_consumption'] = df.loc[index_time,
                                                             'consumption.value.liter'].diff()
            clean_df = pd.concat([clean_df, df[index_time]])

    empty_flag = 1
    if len(clean_df) > 0:
        if len(clean_df[clean_df.time_step.isna()]) < len(clean_df):
            empty_flag = 0

    return clean_df, empty_flag


def get_diff_quantile(y_pred, y, quantile):
    return np.quantile(np.abs(y_pred - y), quantile)


def check_sample_trip(merged_forward, _id_, start, show_all=True, print_info=True):

    trip = merged_forward[merged_forward.TypeTripLogId == _id_].copy()

    distance = trip['DistanceCumsum'].max() - trip['DistanceCumsum'].min()
    AltitudeEndStart = trip['Altitude'][-1] - trip['Altitude'][0]

    print(f"Distance: {distance}")
    print(f"Fuel: {trip['FuelCumsum'].max() - trip['FuelCumsum'].min()}")
    print(f"Time: {trip['TimeDelta'].max() - trip['TimeDelta'].min()}")
    print(f"AltitudeEndStart: {AltitudeEndStart} \n\n")

    time_arr = trip.TimeDelta - trip.TimeDelta[0]
    trip.FuelCumsum = trip.FuelCumsum.replace(np.nan, 0)

    fuel_arr = (trip.FuelCumsum - trip.FuelCumsum[0])
    altitude_arr = trip.Altitude - trip.Altitude[0]
    distance_arr = trip.DistanceCumsum - trip.DistanceCumsum[0]

    time_slot = [i for i in range(start, start + 31)]

    # Fuel over time
    if show_all:
        time_slot = [i for i in range(0, len(time_arr))]

    plt.plot(time_arr[time_slot], fuel_arr[time_slot], label="Fuel over time")
    if show_all:
        plt.plot([start]*100, np.linspace(fuel_arr[time_slot].min(),
                                          fuel_arr[time_slot].max(), 100))
        plt.plot([start + 30]*100, np.linspace(fuel_arr[time_slot].min(),
                                               fuel_arr[time_slot].max(), 100))
    plt.legend()
    plt.show()

    # Altitude over time
    plt.plot(time_arr[time_slot], altitude_arr[time_slot],
             label="Altitude over time")
    if show_all:
        plt.plot([start]*100, np.linspace(altitude_arr[time_slot].min(),
                                          altitude_arr[time_slot].max(), 100))
        plt.plot([start + 30]*100, np.linspace(altitude_arr[time_slot].min(),
                                               altitude_arr[time_slot].max(), 100))

    plt.legend()
    plt.show()

    # Distance over time
    plt.plot(time_arr[time_slot], distance_arr[time_slot],
             label="Distance over time")
    if show_all:
        plt.plot([start]*100, np.linspace(distance_arr[time_slot].min(),
                 distance_arr[time_slot].max(), 100))
        plt.plot([start + 30]*100, np.linspace(distance_arr[time_slot].min(),
                 distance_arr[time_slot].max(), 100))

    plt.legend()
    plt.show()


def estimate_fuel_consumption_of_a_route(fully_route, XGBmodel_name, X_feat, group_size=30, altitude_threshold=0.01):
    """
    Sample trip pieces from df.
    :param fully_route: Formatted dataframe with Ditio and Fuel data for one route
    :param XGBmodel_name: path to the file including the XGG model
    # TODO: can make a json file to include this information, given a xgb_model
    :param X_feat: the features the model is using
    :param group_size: the amount of datapoints that should create one statistics point and be sent into the model
    :altitude_threshold:The granularity in height difference in meters per bin used to define uphill/downhill movement

    :return: the fuel prediction of the given route
    """

    parts_of_route = [fully_route.iloc[i:i+group_size]
                      for i in range(0, len(fully_route), group_size)]

    # TODO: maybe do something if one part is 1 second or something
    # length_last_part = len(parts_of_route[-1])

    # compute statistics and predict fuel values for the route
    statdata = pd.DataFrame()
    for i in range(len(parts_of_route)):
        trip = parts_of_route[i]

        statdata_row = pd.Series(dtype='object')
        statdata_row['LengthDistance'] = trip['DistanceCumsum'].max(
        ) - trip['DistanceCumsum'].min()
        statdata_row['Quantity'] = trip.iloc[0].Quantity
        # Assign the type that is most frequent
        statdata_row['Type'] = trip.Type.mode().values[0]

        trip['AltitudeDiff'] = trip['Altitude'].diff().fillna(0)
        index_uphill = trip['AltitudeDiff'] > altitude_threshold
        index_downhill = trip['AltitudeDiff'] < -altitude_threshold
        statdata_row['AltitudeLoss'] = trip.loc[index_downhill,
                                                'AltitudeDiff'].sum()
        statdata_row['AltitudeGain'] = trip.loc[index_uphill,
                                                'AltitudeDiff'].sum()

        statdata_row['AltitudeDeltaEndStart'] = trip['Altitude'].iloc[-1] - \
            trip['Altitude'].iloc[0]

        statdata_row['AltitudeChange'] = statdata_row['AltitudeGain'] + \
            abs(statdata_row['AltitudeLoss'])

        statdata_row['AltitudeDeltaMaxMin'] = trip['Altitude'].max() - \
            trip['Altitude'].min()

        # TODO: this will be a litt bit wrong with same model?? - less time
        statdata_row['SpeedMean'] = statdata_row['LengthDistance'] / \
            len(trip)

        # Average inclinations
        average_downInclination_part = [0]
        average_upInclination_part = [0]

        for part_i in range(int(len(trip)/2)):
            part_incline = (
                trip['AltitudeDiff']/trip['DistanceCumsumDiff'])[part_i: (15 + part_i)]
            average_downInclination_part.append(
                part_incline[part_incline < 0].mean())
            average_upInclination_part.append(
                part_incline[part_incline > 0].mean())

        statdata_row['DownInclination'] = 0
        statdata_row['DownInclinationPart'] = 0
        if np.sum(index_downhill) > 0:
            statdata_row['DownInclination'] = statdata_row['AltitudeLoss'] / \
                trip.loc[index_downhill, 'DistanceCumsumDiff'].sum()
            statdata_row['DownInclinationPart'] = min(
                average_downInclination_part)

        statdata_row['UpInclination'] = 0
        statdata_row['UpInclinationPart'] = 0
        if np.sum(index_uphill) > 0:
            statdata_row['UpInclination'] = statdata_row['AltitudeGain'] / \
                trip.loc[index_uphill, 'DistanceCumsumDiff'].sum()
            statdata_row['UpInclinationPart'] = max(average_upInclination_part)

        trip['DiffCourse'] = trip.Course.diff().abs()
        statdata_row['SumRotation'] = trip['DiffCourse'][trip['DiffCourse'] < 200].sum()

        statdata_row['Sum_RotationXAltitudeDiffPos'] = (trip['DiffCourse'][(trip['DiffCourse'] < 200) & (index_uphill)]
                                                        * trip['AltitudeDiff'][(trip['DiffCourse'] < 200) & (index_uphill)]).sum()
        statdata_row['Sum_RotationXAltitudeDiffNeg'] = (trip['DiffCourse'][(trip['DiffCourse'] < 200) & (index_downhill)]
                                                        * trip['AltitudeDiff'][(trip['DiffCourse'] < 200) & (index_downhill)]).sum()

        altitudeLossPart = [0]
        altitudeGainPart = [0]
        for part_i in range(int(len(trip)/2)):
            part_altitude_change = trip['AltitudeDiff'][part_i: (15 + part_i)]
            altitudeLossPart.append(
                part_altitude_change[part_altitude_change < 0].sum())
            altitudeGainPart.append(
                part_altitude_change[part_altitude_change > 0].sum())

        statdata_row['AltitudeLossPart'] = min(altitudeLossPart)
        statdata_row['AltitudeGainPart'] = max(altitudeGainPart)

        statdata_row = statdata_row[X_feat]

        statdata = statdata.append(statdata_row, ignore_index=True)

    try:
        model_xgb_loaded = pickle.load(open(XGBmodel_name,  'rb'))
    except Exception as e:
        print(f"#### ERROR1 #### {e}")
        try:
            print(os.getcwd() + '/' + XGBmodel_name)
            model_xgb_loaded = pickle.load(
                open(os.getcwd() + '/' + XGBmodel_name,  'rb'))
        except Exception as e:
            print(f"#### ERROR2 #### {e}")
            exit()

    predicted_fuel_parts = model_xgb_loaded.predict(statdata.values)

    # scale last prediction value according to its length (in sec) - if it isn't perfectly divided
    if (len(fully_route) % group_size) != 0:
        predicted_fuel_parts[-1] = predicted_fuel_parts[-1] * \
            ((len(fully_route) % group_size)/group_size)

    predicted_fuel_fully_route = sum(predicted_fuel_parts)

    return predicted_fuel_fully_route


def sample_trips(merged,
                 trip_types=[1],
                 minlength=30,
                 maxlength=30,
                 TypeTripLogIds=None,
                 Nsamples=300,
                 idling_threshold=1,
                 acceleration_threshold=1,
                 altitude_threshold=1,
                 Nplots=False,
                 half_unique=True,
                 fully_unique=False,
                 psedo_random=False):
    """
    Sample trip pieces from df.
    :param merged: Formatted dataframe with Ditio and Fuel data.
    :param trip_types: List of trip types to be considered. Trips are split between trip types
    :param minlength: Minimum length of samples i seconds
    :param maxlength: Maximum length of samples i seconds
    :param TripLogId: None for random sampling. Else a list of strings specifying the TypeTripLogId (TripLogId_Type) to compute stats and plot for.
    :param Nsamples: Numper of samples
    :idling_threshold: The threshold to distinguish between idling and non-idling in m/s. If zero, the machine is never idling (unless there are negative values) which is rare due to uncertainty. 1 m/s = 3.6 km/h
    :acceleration_threshold: The granularity in speed difference in m/s between bins to define acceleration/deceleration
    :altitude_threshold: The granularity in height difference in meters per bin used to define uphill/downhill movement
    :Nplots: Number of desired plots. Has to be smaller than Nsamples. Negative if no plots are desired. OBS will plot all samples from same triplogid
    :half_unique: avoids completely same trip (lag of 1 second is ok)
    :fully_unique: no overlap in our trips
    :return: dataframe with statistics from the samples
    """

    statdata = pd.DataFrame()

    # Removing pings while reversing
    merged = merged[merged.Forward > 0].copy()
    # Creating new triplogid that splits at subtrack
    merged['TypeTripLogId'] = merged['TripLogId'] + '_' + \
        merged['subtrack'].astype('int').astype('str')

    # Sample with weights according to length of track
    probabilities = merged.groupby('TypeTripLogId').size()
    probabilities = probabilities/probabilities.sum()

    if TypeTripLogIds == None:
        # List of Trips in df to sample from
        TypeTripLogIds_unique = merged.TypeTripLogId.unique()

        if psedo_random:
            np.random.seed(0)

        TypeTripLogIds = np.random.choice(
            TypeTripLogIds_unique, p=probabilities.values, size=Nsamples)

    Nplots = min(len(TypeTripLogIds), Nplots)
    plotTypeTripLogIds = np.random.choice(TypeTripLogIds, size=Nplots)

    # If the same id is in the Ids more than once, it will not be plotted.
    already_plotted = []

    first_skip = True
    for TypeTripLogId in TypeTripLogIds:
        trip = merged[merged.TypeTripLogId == TypeTripLogId].copy()
        trip['TimeDiff'] = (
            trip.Timestamp - trip.Timestamp.min()).dt.total_seconds().values

        # Check if trip longer than minimum required sample length
        if trip.TimeDiff.max() >= minlength:
            # For a fixed minimum length, the sample subset cannot starter later than max(time) minus minimum length
            # max_feasible_start = np.max(np.min([trip.TimeDiff.max()-minlength, maxlength]), 0)    # TODO: delete
            max_feasible_start = np.max(trip.TimeDiff.max()-minlength, 0)
            start = int(np.random.uniform(low=0, high=max_feasible_start + 1))

            max_feasible_length = np.min(
                [maxlength, trip.TimeDiff.max()-start])

            # length = int(np.random.uniform(low=minlength, high=max_feasible_length + 1))  # TODO: delete
            # Can't have a longer length than the max feasible length
            length = np.min([max_feasible_length, int(
                np.random.uniform(low=minlength, high=max_feasible_length + 1))])

            subset_index = (trip.TimeDiff >= start) & (
                trip.TimeDiff < start + length)

            subset = trip[subset_index].copy()

            subset['TimeDelta'] = subset.TimeDiff.diff().fillna(
                0)  # Timestep length
            subset['SpeedDiff'] = (
                subset['Speed'].diff()/subset['TimeDelta']).fillna(0)

            # Compute statistics
            statdata_row = pd.Series(dtype='object')
            statdata_row['TripLogId'] = subset.iloc[0].TripLogId
            statdata_row['TypeTripLogId'] = subset.iloc[0].TypeTripLogId
            statdata_row['DistanceFullyRoute'] = trip['DistanceCumsum'].max(
            ) - trip['DistanceCumsum'].min()
            statdata_row['ControlStartTime'] = start
            statdata_row['ControlStartTimeClock'] = str(subset.index[0])
            # Assign the type that is most frequent
            statdata_row['Type'] = subset.Type.mode().values[0]
            statdata_row['StartTime'] = subset.iloc[0].Timestamp
            statdata_row['Month'] = subset.iloc[0].Timestamp.month
            statdata_row['LengthTime'] = length
            statdata_row['LengthDistance'] = subset['DistanceCumsum'].max(
            ) - subset['DistanceCumsum'].min()

            # statdata_row['SpeedMean'] = subset['Speed'].mean() TODO: delete? - more accurate with distance/ time?
            statdata_row['SpeedMean'] = statdata_row['LengthDistance'] / \
                statdata_row['LengthTime']
            statdata_row['SpeedVariance'] = subset['Speed'].var()
            statdata_row['Quantity'] = subset.iloc[0].Quantity
            statdata_row['id_uniq_stat'] = statdata_row['TypeTripLogId'] + \
                '_' + str(start)

            # If we want partly unique samples, or fully unique samples
            if half_unique or fully_unique:
                if first_skip:
                    first_skip = False
                else:
                    same_trip_id = statdata[statdata['TypeTripLogId']
                                            == TypeTripLogId]
                    overlap = np.sum((same_trip_id.ControlStartTime > start - 30)
                                     & (same_trip_id.ControlStartTime < start + 30)) > 0

                    if (len(same_trip_id) > 0):
                        # if we have the exact same trip
                        if half_unique and (np.sum(same_trip_id.ControlStartTime == start) > 0):
                            continue
                        # if we have some overlap in our trip
                        elif fully_unique and overlap:
                            continue

            # Integral over increases in velocity (=acceleration)
            index_positive = subset['SpeedDiff'] > acceleration_threshold
            # Sum of positive acceleration
            statdata_row['SpeedDiffPositiveSum'] = (subset.loc[index_positive, 'SpeedDiff'] *
                                                    subset['TimeDelta'].loc[index_positive]).sum()
            # Fuel used while accelerating
            statdata_row['AccFuel'] = (subset.loc[index_positive, 'FuelCumsumDiff'] *
                                       subset['TimeDelta'].loc[index_positive]).sum()
            # Time while accelerating
            statdata_row['AccTime'] = (
                subset['TimeDelta'].loc[index_positive]).sum()

            # Integral over decreases in velocity (=decceleration)
            index_negative = subset['SpeedDiff'] < acceleration_threshold
            # Sum of negative acceleration
            statdata_row['SpeedDiffNegativeSum'] = (subset.loc[index_negative, 'SpeedDiff'] *
                                                    subset['TimeDelta'].loc[index_negative]).sum()

            # Sum of positive/negative altitude differences
            subset['AltitudeDiff'] = subset['Altitude'].diff().fillna(0)
            index_uphill = subset['AltitudeDiff'] > altitude_threshold
            index_downhill = subset['AltitudeDiff'] < -altitude_threshold
            statdata_row['AltitudeLoss'] = subset.loc[index_downhill,
                                                      'AltitudeDiff'].sum()
            statdata_row['AltitudeGain'] = subset.loc[index_uphill,
                                                      'AltitudeDiff'].sum()

            altitudeLossPart = [0]
            altitudeGainPart = [0]
            speedMean_altitudeGain = [0]
            speedMean_altitudeLoss = [0]
            for part_i in range(int(length/2)):
                part_altitude_change = subset['AltitudeDiff'][part_i: (
                    15 + part_i)]
                distanceCumsumPart = subset.DistanceCumsum[part_i: (
                    int(length/2) + part_i)]
                part_speedMean = (max(distanceCumsumPart) -
                                  min(distanceCumsumPart))/int(length/2)

                altitudeLossPart.append(
                    part_altitude_change[part_altitude_change < 0].sum())
                altitudeGainPart.append(
                    part_altitude_change[part_altitude_change > 0].sum())
                speedMean_altitudeLoss.append(
                    part_speedMean * part_altitude_change[part_altitude_change < 0].sum())
                speedMean_altitudeGain.append(
                    part_speedMean * part_altitude_change[part_altitude_change > 0].sum())

            statdata_row['AltitudeLossPart'] = min(altitudeLossPart)
            statdata_row['MAX_speedMean_altitudeLossPart'] = min(
                speedMean_altitudeLoss)

            statdata_row['AltitudeGainPart'] = max(altitudeGainPart)
            statdata_row['MAX_speedMean_altitudeGainPart'] = max(
                speedMean_altitudeGain)

            statdata_row['AltitudeChange'] = statdata_row['AltitudeGain'] + \
                abs(statdata_row['AltitudeLoss'])

            statdata_row['AltitudeDeltaEndStart'] = subset['Altitude'].iloc[-1] - \
                subset['Altitude'].iloc[0]
            statdata_row['AltitudeDeltaMaxMin'] = subset['Altitude'].max(
            ) - subset['Altitude'].min()

            # Average inclinations
            average_downInclination_part = [0]
            average_upInclination_part = [0]

            for part_i in range(int(length/2)):
                part_incline = (
                    subset['AltitudeDiff']/subset['DistanceCumsumDiff'])[part_i: (15 + part_i)]
                average_downInclination_part.append(
                    part_incline[part_incline < 0].mean())
                average_upInclination_part.append(
                    part_incline[part_incline > 0].mean())

            statdata_row['DownInclination'] = 0
            statdata_row['DownInclinationPart'] = 0
            if np.sum(index_downhill) > 0:
                statdata_row['DownInclination'] = statdata_row['AltitudeLoss'] / \
                    subset.loc[index_downhill, 'DistanceCumsumDiff'].sum()
                statdata_row['DownInclinationPart'] = min(
                    average_downInclination_part)

            statdata_row['UpInclination'] = 0
            statdata_row['UpInclinationPart'] = 0
            if np.sum(index_uphill) > 0:
                statdata_row['UpInclination'] = statdata_row['AltitudeGain'] / \
                    subset.loc[index_uphill, 'DistanceCumsumDiff'].sum()
                statdata_row['UpInclinationPart'] = max(
                    average_upInclination_part)

 #           print(statdata_row['AltitudeLoss'], subset.loc[index_downhill, 'DistanceCumsumDiff'].sum(),
 #                 statdata_row['AltitudeGain'], subset.loc[index_uphill, 'DistanceCumsumDiff'].sum(),
 #                 statdata_row['DownInclination'], statdata_row['UpInclination'])

            # Time with idling and fuel during idling
            index_idling = subset['Speed'] < idling_threshold
            statdata_row['IdlingTime'] = subset.loc[index_idling,
                                                    'TimeDelta'].sum()
            statdata_row['IdlingFuel'] = (
                subset.loc[index_idling, 'TimeDelta']*subset.loc[index_idling, 'FuelCumsumDiff']).sum()

            # Sum of changes in direction
            # Data has already been filtered for large changes in angle, so we assume only smallest change is relevant
#            TODO: double check this
            subset['DiffCourse'] = subset.Course.diff().abs()
            subset['DiffCourse360'] = (360 - subset['DiffCourse'])
            subset['DeltaCourse'] = subset[[
                'DiffCourse', 'DiffCourse360']].min(axis=1)

            # statdata_row['SumRotation'] = 1 #subset[['DeltaCourse', 'DeltaCourse180']].min(axis=1).sum()
            statdata_row['SumRotation'] = subset['DiffCourse'][subset['DiffCourse'] < 200].sum(
            )
            statdata_row['Sum_RotationXAltitudeDiffPos'] = (subset['DiffCourse'][(subset['DiffCourse'] < 200) & (index_uphill)]
                                                            * subset['AltitudeDiff'][(subset['DiffCourse'] < 200) & (index_uphill)]).sum()
            statdata_row['Sum_RotationXAltitudeDiffNeg'] = (subset['DiffCourse'][(subset['DiffCourse'] < 200) & (index_downhill)]
                                                            * subset['AltitudeDiff'][(subset['DiffCourse'] < 200) & (index_downhill)]).sum()

            # Fuel
            statdata_row['Fuel'] = subset['FuelCumsum'].max() - \
                subset['FuelCumsum'].min()

            if (statdata_row.Quantity > 0) & (statdata_row['LengthDistance'] > 0):
                statdata_row['FuelPerTonPerMeter'] = statdata_row['Fuel'] / \
                    statdata_row['Quantity']/statdata_row['LengthDistance']

            statdata = statdata.append(statdata_row, ignore_index=True)

            if TypeTripLogId in plotTypeTripLogIds and not TypeTripLogId in already_plotted:
                already_plotted.append(TypeTripLogId)

                N = 6
                fig, axs = plt.subplots(1, N, figsize=(5*N, 6))
                fig.suptitle(
                    f'{TypeTripLogId}, length trip [s]: {int(trip.TimeDiff.max())}, start [s]: {int(start)}, length [s]: {int(length)}, rotation: {statdata_row.SumRotation}', fontsize=16)

                i = 0
                im = axs[i].scatter(trip.x, trip.y, marker='o',
                                    alpha=0.5, c=trip.TimeDiff)
                plt.colorbar(im, ax=axs[i])
                axs[i].scatter(subset.x, subset.y, marker='x', color='black')
                axs[i].ticklabel_format(style='plain')

                i = 1
                xy_cols = ['TimeDiff', 'DistanceCumsum']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5, label='All')
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                x = subset.loc[index_idling, xy_cols[0]]
                y = subset.loc[index_idling, xy_cols[1]]
                axs[i].scatter(x, y, alpha=0.5, label='Idling')
                axs[i].legend()

                i = 2
                xy_cols = ['TimeDiff', 'Speed']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].plot(x, y, alpha=1)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])
                axs[i].axhline(idling_threshold)

                x = subset.loc[index_positive, xy_cols[0]]
                y = (subset.loc[index_positive, 'SpeedDiff'] *
                     subset['TimeDelta'].loc[index_positive]).cumsum() + subset.iloc[0][xy_cols[1]]
                axs[i].scatter(x, y, alpha=0.5,
                               label='SpeedDiffPositiveCumSum')
                x = subset.loc[index_negative, xy_cols[0]]
                y = (subset.loc[index_negative, 'SpeedDiff'] *
                     subset['TimeDelta'].loc[index_negative]).cumsum() + subset.iloc[0][xy_cols[1]]
                axs[i].scatter(x, y, alpha=0.5,
                               label='SpeedDiffNegativeCumSum')

                x = subset.loc[index_idling, xy_cols[0]]
                y = subset.loc[index_idling, 'Speed']
                axs[i].scatter(x, y, alpha=0.5, label='Idling')

                axs[i].legend()

                i = 3
                xy_cols = ['TimeDiff', 'Altitude']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].plot(x, y, alpha=1)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                x = subset.loc[index_uphill, xy_cols[0]]
                y = subset.loc[index_uphill, 'AltitudeDiff'].cumsum(
                ) + subset.iloc[0][xy_cols[1]]
                axs[i].scatter(x, y, alpha=0.5, label='AltitudeGain')
                x = subset.loc[index_downhill, xy_cols[0]]
                y = subset.loc[index_downhill, 'AltitudeDiff'].cumsum(
                ) + subset.iloc[0][xy_cols[1]]
                axs[i].scatter(x, y, alpha=0.5, label='AltitideLoss')
                axs[i].legend()

                i = 4
                xy_cols = ['TimeDiff', 'FuelCumsum']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].plot(x, y, alpha=1)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                x = subset.loc[index_positive, xy_cols[0]]
                y = subset.loc[index_positive, xy_cols[1]]
                axs[i].scatter(x, y, alpha=0.5, label='SpeedDiffPositive')

#                x = subset.loc[index_idling, xy_cols[0]]
#                y = (subset.loc[index_idling, 'TimeDelta']*subset.loc[index_idling, 'FuelCumsumDiff']).cumsum() + subset.iloc[0][xy_cols[1]]
#                axs[i].scatter(x, y, alpha=0.5, label='Fuel idling')

                x = subset.loc[index_positive, xy_cols[0]]
                y = (subset.loc[index_positive, 'FuelCumsumDiff'] *
                     subset.loc[index_positive, 'TimeDelta']).cumsum() + subset.iloc[0][xy_cols[1]]

                axs[i].scatter(x, y, alpha=0.5, label='Fuel accelerating')
                axs[i].legend()

                i = 5
                xy_cols = ['TimeDiff', 'Course']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5, label='Course')
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                xy_cols = ['TimeDiff', 'DiffCourse']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5, label='Diff')

                xy_cols = ['TimeDiff', 'DiffCourse360']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='x', alpha=0.5, label='Diff360')

                xy_cols = ['TimeDiff', 'DeltaCourse']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='x', alpha=0.5, label='Delta')

                axs[i].legend()

                plt.show()
                plt.close()

    return statdata


def check_overall_performance_of_models(weighted=True, data='test'):

    a = num_statistics_per_day('fully_unique', _print=False)
    dict_count_samples = {}
    for date_and_count in a.split('\n')[:-1]:
        date = date_and_count.split(':')[0].strip()
        count = date_and_count.split(':')[1].strip()
        dict_count_samples[date] = count

    i = 0
    extra_path = 'models_and_results' + f'{i}'
    path = PATH_QUICK_RUN + extra_path
    isExist = os.path.exists(path)

    while isExist:

        infile = open(path + '/information.txt')
        test_scores = []
        dates = []
        for line in infile:
            if line.strip() != '':
                date = line.split('--')[0].strip()
                dates.append(date)
                if data == 'test':
                    test_score = float(line.split(
                        ':')[1].split(',')[0].strip())
                elif data == 'val':
                    test_score = float(line.split(
                        ':')[2].split(',')[0].strip())
                test_scores.append(test_score)

        if not weighted:
            mean_of_MSE = np.mean(test_scores)

        else:
            total_samples = 0
            for date in dates:
                total_samples += int(dict_count_samples[date])

            mean_of_MSE = 0
            for date, test_score in zip(dates, test_scores):
                frac = int(dict_count_samples[date])/total_samples
                mean_of_MSE += frac * test_score

        print(extra_path + f':{mean_of_MSE : .5f}')

        i += 1
        extra_path = 'models_and_results' + f'{i}'
        path = PATH_QUICK_RUN + extra_path
        isExist = os.path.exists(path)


def check_errors_given_date_all_models(date):

    i = 0
    path = PATH_QUICK_RUN + '/models_and_results' + f'{i}'
    isExist = os.path.exists(path)

    val_scores = []

    while isExist:
        val_scores.append(100)

        infile = open(path + '/information.txt')
        for line in infile:

            if line.strip() != '':
                if line.split('--')[0].strip() == date:
                    val_scores[i] = float(line.split(
                        ':')[1].split(',')[0].strip())
                    break

        i += 1
        path = PATH_QUICK_RUN + '/models_and_results' + f'{i}'
        isExist = os.path.exists(path)

    print(date + ':')
    for i, score in enumerate(val_scores):
        print(f'folder{i}: {score}')


def rank_features(num, print_info=False, write_file=True):
    """
    Function that will calculate the total rank of a feature 
    based on several criteria, SAGE (test/ validation) and 
    feature importance (in-build XGboost method). 

    The files should be already made. 
    """

    path = PATH_QUICK_RUN + f'models_and_results{num}/importance_values/'

    dict_test = {}
    dict_val = {}
    dict_feat_importance = {}

    for filename in os.listdir(path):

        infile = open(path + filename, 'r')

        if 'test' in filename:
            now_dict = dict_test
        elif 'val' in filename:
            now_dict = dict_val
        elif 'feature' in filename:
            now_dict = dict_feat_importance
        else:
            print('WIERD FILENAME: ' + filename)

        for line in infile:

            feat = line.split()[0]
            val = float(line.split()[1])

            if feat in now_dict:
                now_dict[feat] += val
            else:
                now_dict[feat] = val

        infile.close()

    sorted_dict_val = sorted(
        dict_val.items(), key=lambda x: x[1], reverse=True)
    sorted_dict_test = sorted(
        dict_test.items(), key=lambda x: x[1], reverse=True)
    sorted_dict_feat_importance = sorted(
        dict_feat_importance.items(), key=lambda x: x[1], reverse=True)

    rankings = {}
    for i in range(len(dict_test)):
        rankings[sorted_dict_val[i][0]] = [i]

    for i in range(len(dict_test)):
        rankings[sorted_dict_test[i][0]].append(i)

    for i in range(len(dict_test)):
        rankings[sorted_dict_feat_importance[i][0]].append(i)

    rankings_sum = rankings.copy()
    for key in rankings:
        rankings_sum[key] = np.sum(rankings_sum[key])

    sorted_lol = sorted(rankings_sum.items(),
                        key=lambda x: x[1], reverse=False)
    sorted_rankings_sum = {}
    for i in range(len(sorted_lol)):
        sorted_rankings_sum[sorted_lol[i][0]] = sorted_lol[i][1]

    if print_info:
        print('TOTAL RANK   TEST   VAL   FEAT_IMPORTANCE   (FEATURE)')
        for key in sorted_rankings_sum:
            print(
                f' {sorted_rankings_sum[key] : 5.0f} {rankings[key][0] : 8.0f} {rankings[key][1] : 5.0f} {rankings[key][2]: 11.0f}' + ' '*12 + key)

    if write_file:
        filename = PATH_QUICK_RUN + \
            f'models_and_results{num}/features_ranking.txt'
        outfile = open(filename, 'w')

        outfile.write(
            'TOTAL RANK   TEST   VAL   FEAT_IMPORTANCE   (FEATURE) \n')
        for key in sorted_rankings_sum:
            outfile.write(
                f' {sorted_rankings_sum[key] : 5.0f} {rankings[key][0] : 8.0f} {rankings[key][1] : 5.0f} {rankings[key][2]: 11.0f}' + ' '*12 + key + '\n')

        # sort_keys = [-1]*len(rankings)
        # for key in rankings:
        #     sort_keys[rankings[key][1]] = key

        # outfile.write(
        #     '    VAL           (FEATURE) \n')
        # for key in sort_keys:
        #     outfile.write(
        #         f' {rankings[key][1]: 5.0f}' + ' '*12 + key + '\n')

        outfile.close()


def save_many_statistics(search_num=20,
                         psedo_random=False,
                         fully_unique=False,
                         sub_name='half_unique'):

    merged = pd.DataFrame()
    for date in DATES:
        filename = f'{PATH_CURR_DATA}/{date}_to_{date}_merged.h5'
        merged_date = pd.read_hdf(filename)
        merged = merged.append(merged_date)

    merged_forward = merged[merged.Forward > 0].copy()
    index = merged_forward.Type.isin([0, 2])
    merged_forward.loc[index, 'Quantity'] = 0

    statdata = sample_trips(merged_forward, trip_types=[0, 1, 2],
                            minlength=30,
                            maxlength=30,
                            Nsamples=search_num,
                            TypeTripLogIds=None,
                            idling_threshold=1,
                            acceleration_threshold=0.01,
                            altitude_threshold=.01,
                            Nplots=0,
                            half_unique=not fully_unique,
                            fully_unique=fully_unique,
                            psedo_random=psedo_random)

    filename_save = PATH_DATA + 'statistics_samples/samples_' + sub_name + '.pkl'
    pickle.dump(statdata, open(filename_save, "wb"))


def get_df_of_statistics(dates, sub_name='half_unique'):
    filename_load = PATH_DATA + 'statistics_samples/samples_' + sub_name + '.pkl'
    statdata = pd.read_pickle(filename_load)
    collected_statdata = pd.DataFrame()

    for date in dates:
        statdata_curr_date = statdata[statdata.StartTime.between(
            date + ' 00:00:00', date + ' 23:59:59')]
        collected_statdata = collected_statdata.append(
            statdata_curr_date, ignore_index=True)

    return collected_statdata, filename_load


def num_statistics_per_day(sub_name='half_unique', _print=True):
    output_string = ''
    for date in DATES:
        output_string += date + \
            f' :  {len(get_df_of_statistics([date], sub_name = sub_name)[0])}\n'

    if _print:
        print(output_string)
    return output_string


def find_similar_fuel_values(fuel_num, print_num=3, psedu_random=True, search_num=500,
                             filename_load=PATH_DATA + 'statistics_samples/samples_half_unique.pkl'):

    # find value closest to a number in pandas column
    statdata = pd.read_pickle(filename_load)
    closest = statdata['Fuel'].sub(fuel_num).abs().idxmin()
    statdata['Fuel_diff'] = statdata['Fuel'].sub(fuel_num).abs()

    closest_frame = statdata.sort_values('Fuel_diff')[:print_num]
    for i in range(print_num):
        print('\n\n----------------------------------------------------------------')
        curr_observation = closest_frame.iloc[i]
        for a, b in zip(curr_observation[X_FEAT_ALL + ['Fuel']], X_FEAT_ALL + ['Fuel']):
            print(f'{b} : {a}')


def plots_of_routes(track_data, fuel_data, TripLogId=''):

    N = 3
    fig, axs = plt.subplots(1, N, figsize=(5*N, 6))
    fig.suptitle('TripLogId: ' + TripLogId, fontsize=16)

    i = 0
    im = axs[i].scatter(track_data.x, track_data.y, marker='o', alpha=0.5, c=(
        track_data.index - track_data.index.min()).seconds, label="track")
    plt.colorbar(im, ax=axs[i])
    axs[i].scatter(fuel_data.x, fuel_data.y, marker='x',
                   color='black', label="fuel")
    axs[i].title.set_text(
        f'Start: {(fuel_data.x[0], fuel_data.y[0])}, Stop: {(fuel_data.x[-1], fuel_data.y[-1])}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

    i = 1
    axs[i].plot((track_data.index - track_data.index.min()
                 ).seconds, track_data.Speed, marker='o')
    axs[i].plot((track_data.index - track_data.index.min()
                 ).seconds, track_data.Speed)
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Speed')

    i = 2
    axs[i].plot((fuel_data.index - fuel_data.index.min()).seconds,
                fuel_data.data_Fuel, marker="x", alpha=1)
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Fuel')

    date = track_data.index.date[0].strftime("%Y-%m-%d")
    path = PATH_FIGURES + 'routes/' + date + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    fig.legend()
    plt.savefig(path + TripLogId + '.png')
    plt.close()


def get_X_feat():
    X_feat_all = ['LengthDistance',  # 'LengthTime',
                  'Quantity', 'Type',
                  'AltitudeGain', 'AltitudeLoss', 'AltitudeDeltaEndStart', 'AltitudeChange',
                  'AltitudeLossPart', 'AltitudeGainPart',
                  'AltitudeDeltaMaxMin',
                  'Sum_RotationXAltitudeDiffPos', 'Sum_RotationXAltitudeDiffNeg',
                  'UpInclination', 'DownInclination',
                  'DownInclinationPart', 'UpInclinationPart',
                  # 'Fuel',
                  # 'AccTime',
                  'SpeedMean',  # 'SpeedVariance',
                  # 'SpeedDiffPositiveSum', 'SpeedDiffNegativeSum',
                  # 'IdlingTime', 'SumRotation',
                  'SumRotation', 'TypeTripLogId', 'DistanceFullyRoute', 'ControlStartTime', 'ControlStartTimeClock'
                  ]
    X_feat = [  # 'LengthTime',
        'Quantity',  # 'Type'
        'AltitudeGain', 'AltitudeDeltaEndStart', 'AltitudeLoss',  # 'AltitudeChange',
        # 'AltitudeDeltaMaxMin',
        'Sum_RotationXAltitudeDiffPos', 'Sum_RotationXAltitudeDiffNeg',
        'UpInclination', 'DownInclination',
        'UpInclinationPart', 'DownInclinationPart',
        'AltitudeLossPart', 'AltitudeGainPart',
        # 'Fuel',
        # 'AccTime',
        'SpeedMean', 'LengthDistance',  # 'SpeedVariance',
        # 'SpeedDiffPositiveSum', 'SpeedDiffNegativeSum',
        # 'IdlingTime', 'SumRotation',
        'SumRotation'
    ]

    return X_feat_all, X_feat


def get_cluster_probability(data, algorithm, args, kwds):

    clusterer = algorithm(*args, **kwds).fit(data)
    labels = algorithm(*args, **kwds).fit_predict(data)

    return clusterer.labels_    # dbscan
    return clusterer.outlier_scores_    # hdbscan


def set_outliers(outlier_threshold=0.2, min_cluster_size=5, sub_name='half_unique'):

    df, filename = get_df_of_statistics(DATES, sub_name=sub_name)
    # X_feat = [#'LengthTime',
    #         'AltitudeLoss', 'AltitudeGain',
    #         'Fuel',
    #         'LengthDistance',
    # ]
    X_feat = [  # 'LengthTime',
        'AltitudeDeltaEndStart',
        'Fuel',
        'LengthDistance',
        'AltitudeGain',
        'AltitudeLoss',
        'AltitudeGainPart',
        'Quantity'
    ]

    ########################################
    ########## Cluster HDBSCAN #############
    ########################################
    # outliers = get_cluster_probability(df[X_feat].values, hdbscan.HDBSCAN, (), {'min_cluster_size': min_cluster_size})
    # df['outliers'] = outliers > outlier_threshold

    ########################################
    ########## Cluster DBSCAN ##############
    ########################################
    # USED ON MODEL 20, unique

    eps = 0.5
    adder = 1

    outliers = get_cluster_probability(
        df[X_feat].values, cluster.DBSCAN, (), {'eps': 0.5})
    frac_of_outliers = np.sum(outliers == -1) / len(outliers)
    a, b = [0.1, 0.12]
    last_a = False

    while frac_of_outliers < a or frac_of_outliers > b:
        if frac_of_outliers < a:
            eps -= adder
            if not last_a:
                adder /= 2

            last_a = True
        else:
            eps += adder

            if last_a:
                adder /= 2

            last_a = False

        outliers = get_cluster_probability(
            df[X_feat].values, cluster.DBSCAN, (), {'eps': eps})
        frac_of_outliers = np.sum(outliers == -1) / len(outliers)

    df['outliers'] = (outliers == -1)

    pickle.dump(df, open(filename, "wb"))


def get_statistic_10_min(routes_dict,
                         idling_threshold=1,
                         acceleration_threshold=1,
                         altitude_threshold=1):

    statdata = pd.DataFrame()

    # TODO: Removing pings while reversing
    # merged = merged[merged.Forward > 0].copy()
    # Creating new triplogid that splits at subtrack

    for route_key in routes_dict:
        route = routes_dict[route_key]
        track = route['track_data']

        statdata_row = pd.Series(dtype='object')
        statdata_row['start_time'] = route['start_time']
        statdata_row['stop_time'] = route['stop_time']
        statdata_row['total_seconds'] = (
            route['stop_time'] - route['start_time']).seconds
        statdata_row['length_distance'] = np.sum(
            track.DistanceDriven.diff()[track.DistanceDriven.diff() > 0])   # starts on 0 every new trip
        statdata_row['type_1_frac'] = np.sum(
            track.diff_seconds[track['Type'] == 1]) / statdata_row['total_seconds']
        statdata_row['type_2_frac'] = np.sum(
            track.diff_seconds[track['Type'] == 2]) / statdata_row['total_seconds']
        statdata_row['type_0_frac'] = 1 - \
            (statdata_row['type_1_frac'] + statdata_row['type_2_frac'])

        # statdata_row['SpeedMean'] = statdata_row['LengthDistance'] / \
        #     statdata_row['LengthTime']
        # statdata_row['Quantity'] = subset.iloc[0].Quantity

        track['AltitudeDiff'] = track['Altitude'].diff().fillna(0)
        # TODO: maybe just look at when incline or decline for some time? or nah

        index_uphill = track['AltitudeDiff'] > altitude_threshold
        index_downhill = track['AltitudeDiff'] < -altitude_threshold
        statdata_row['AltitudeLoss'] = track.loc[index_downhill,
                                                 'AltitudeDiff'].sum()
        statdata_row['AltitudeGain'] = track.loc[index_uphill,
                                                 'AltitudeDiff'].sum()

        statdata_row['AltitudeDeltaMaxMin'] = track['Altitude'].max(
        ) - track['Altitude'].min()

        # Time with idling and fuel during idling
        index_idling = track['Speed'] < idling_threshold
        statdata_row['IdlingTime'] = track.loc[index_idling,
                                               'diff_seconds'].sum()

        track['DiffCourse'] = track.Course.diff().abs()
        track['DiffCourse360'] = (360 - track['DiffCourse'])
        track['DeltaCourse'] = track[[
            'DiffCourse', 'DiffCourse360']].min(axis=1)

        # statdata_row['SumRotation'] = 1 #subset[['DeltaCourse', 'DeltaCourse180']].min(axis=1).sum()
        statdata_row['SumRotation'] = track['DiffCourse'][track['DiffCourse'] < 200].sum(
        )
        statdata_row['Sum_RotationXAltitudeDiffPos'] = (track['DiffCourse'][(track['DiffCourse'] < 200) & (index_uphill)]
                                                        * track['AltitudeDiff'][(track['DiffCourse'] < 200) & (index_uphill)]).sum()
        statdata_row['Sum_RotationXAltitudeDiffNeg'] = (track['DiffCourse'][(track['DiffCourse'] < 200) & (index_downhill)]
                                                        * track['AltitudeDiff'][(track['DiffCourse'] < 200) & (index_downhill)]).sum()

        # Fuel
        statdata_row['fuel_consumption_val'] = route['consumption_val']
        statdata_row['fuel_consumption_val_liter'] = route['consumption_val_liter']

        statdata = statdata.append(statdata_row, ignore_index=True)

    return statdata


def get_second_from_time_formatted(time_formatted):
    time_list = time_formatted.split(':')
    seconds = int(time_list[0])*3600 + \
        int(time_list[1])*60 + int(float(time_list[2]))
    return seconds


def get_specific_data_from_fully_unique():
    a, _ = get_df_of_statistics(DATES, 'fully_unique')
    b = a[['StartTime', 'Fuel', 'LengthDistance',
           'AltitudeDeltaEndStart', 'AccTime', 'Quantity']]
    c = b.sort_values(by='Fuel')
    c.index = range(1, len(c) + 1)
    d = c.round({'Fuel': 2, 'LengthDistance': 0, 'AltitudeDeltaEndStart': 1})
    d = d.astype({'LengthDistance': 'int'})
    d['StartTime'] = d['StartTime'].dt.tz_convert(None)

    return d


def quick_comparison_between_fully_or_not_unique_data():

    dates = ['2021-09-28',
             '2021-09-29',
             '2021-10-13',
             '2021-10-14',
             '2021-10-15',
             '2021-10-18',
             '2021-10-19',
             '2021-10-20',
             '2021-10-21',
             ]

    X_feat_all = ['LengthDistance',  # 'LengthTime',
                  'Quantity', 'Type',
                  'AltitudeGain', 'AltitudeLoss', 'AltitudeDeltaEndStart', 'AltitudeChange',
                  'AltitudeLossPart', 'AltitudeGainPart',
                  'AltitudeDeltaMaxMin',
                  'Sum_RotationXAltitudeDiffPos', 'Sum_RotationXAltitudeDiffNeg',
                  'DownInclination', 'UpInclination',
                  'DownInclinationPart', 'UpInclinationPart',
                  # 'Fuel',
                  'AccTime',
                  'SpeedMean',  # 'SpeedVariance',
                  #'SpeedDiffPositiveSum', 'SpeedDiffNegativeSum',
                  #'IdlingTime', 'SumRotation',
                  'SumRotation', 'TypeTripLogId', 'DistanceFullyRoute', 'ControlStartTime', 'ControlStartTimeClock'
                  ]

    print('date: error not_unique, error unique')
    error_not_unique = []
    error_unique = []
    for date in dates:

        statdata, _ = get_df_of_statistics([date], sub_name='fully_unique')

        # Run with not fully unique
        folder = PATH_QUICK_RUN + 'models_and_results0'
        filename = folder + f'/models/without_' + date + '.pkl'
        model_xgb_loaded_not_unique = pickle.load(open(filename,  'rb'))

        # Run with fully unique
        folder = PATH_QUICK_RUN + 'models_and_results11'
        filename = folder + f'/models/without_' + date + '.pkl'
        model_xgb_loaded_unique = pickle.load(open(filename,  'rb'))

        infile = open(folder + '/features.txt', 'r')
        X_feat = []
        for line in infile:
            X_feat.append(line.strip())
        infile.close()

        y_feat = ['Fuel']
        y_val = statdata[y_feat]

        index_1 = y_val.sort_values('Fuel').index
        X_val_fact_sorted = statdata[X_feat_all].iloc[index_1]

        X_val_sorted = X_val_fact_sorted[X_feat]
        y_val_sorted = y_val.iloc[index_1].values.flatten()

        #not unique
        y_pred_xgb_val_not_unique = model_xgb_loaded_not_unique.predict(
            X_val_sorted)
        mse_xgb_val_not_unique = mean_squared_error(
            y_pred_xgb_val_not_unique, y_val_sorted)
        error_not_unique.append(mse_xgb_val_not_unique)

        # unique
        y_pred_xgb_val_unique = model_xgb_loaded_unique.predict(X_val_sorted)
        mse_xgb_val_unique = mean_squared_error(
            y_pred_xgb_val_unique, y_val_sorted)
        error_unique.append(mse_xgb_val_unique)

        print(
            date + f': {mse_xgb_val_not_unique:.4f}, {mse_xgb_val_unique:.4f}')

    a = num_statistics_per_day('fully_unique', _print=False)
    dict_count_samples = {}
    for date_and_count in a.split('\n')[:-1]:
        date = date_and_count.split(':')[0].strip()
        count = date_and_count.split(':')[1].strip()
        dict_count_samples[date] = count

    total_samples = 0
    for date in dates:
        total_samples += int(dict_count_samples[date])

    mean_of_MSE_not_unique = 0
    for date, test_score in zip(dates, error_not_unique):
        frac = int(dict_count_samples[date])/total_samples
        mean_of_MSE_not_unique += frac * test_score

    mean_of_MSE_unique = 0
    for date, test_score in zip(dates, error_unique):
        frac = int(dict_count_samples[date])/total_samples
        mean_of_MSE_unique += frac * test_score

    print(f'TOTAL: {mean_of_MSE_not_unique:.4f}, {mean_of_MSE_unique:.4f}')


if __name__ == '__main__':
    embed()

    # for date in DATES:
    #     print(date + f' :  {len(get_df_of_statistics([date]))}')
    # # find_similar_fuel_values(0.1)

    # get date of pandas timestamp?
