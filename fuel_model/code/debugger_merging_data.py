# conda activate daenv
import datetime
import helper
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns
import road_data_manipulation_v2 as rdm
import aux_functions as aux
from IPython import embed
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def specification(date, folder_name):

    # Ditio specificiations
    # Remember that you cant be on SINTEF-net when calling the Ditio API
    destination = helper.PATH_DATA + folder_name
    eventsFile = f'{destination}/{date}_to_{date}_events_preprocessed.h5'
    trackingFile = f'{destination}/{date}_to_{date}_tracking_events_preprocessed.h5'

    # Mapping dumper ids from motion data to fuel data
    id_mapping = {'605c4727bf26243ac44fd169': 352555108617496}

    return destination, eventsFile, trackingFile, date, id_mapping


def load_data(destination, dates):
    merged = pd.DataFrame()

    for date in dates:
        filename = f'{destination}/{date}_to_{date}_merged.h5'
        merged_date = pd.read_hdf(filename)
        print(date, len(merged_date.TripLogId.unique()))
        merged = merged.append(merged_date)

    return merged


def get_ditio_data(eventsFile, trackingFile, id_mapping, destination):
    events = pd.read_hdf(eventsFile)
    tracking_events = pd.read_hdf(trackingFile)

    data_raw, proj_info = rdm.read_data(events, tracking_events)
    tracking_index_DumperId = data_raw.DumperId_x == list(id_mapping.keys())[0]
    data_raw = data_raw[tracking_index_DumperId]
    data_raw['DumperId'] = data_raw['DumperId_x']
    data_raw.drop(['DumperId_x', 'DumperId_y'], axis=1, inplace=True)
    if len(data_raw) > 1:
        # Preprocess GPS data, removing first/last 30 meters of data (reversing of dumpers)
        data = rdm.trips_processing(data_raw, endpoint_threshold=0, remove_endpoints=False,
                                    variables=list(data_raw.columns) + ['DistanceDriven'])
    else:
        print('No data for the selected parameters')

    return data, proj_info


def get_fuel_data(date, destination):
    filename = f'{destination}/{date}_to_{date}_fuel.h5'
    fuel = pd.read_hdf(filename)
    print(f'Fuel data loaded from {filename}')
    fuel.info()

    return fuel


def add_fuel(tracking_subset: pd.core.frame.DataFrame,
             fuel_subset: pd.core.frame.DataFrame,
             TripLogIds: np.ndarray,
             id_mapping: dict,
             dev_outer_threshold: float,
             dev_points_threshold: float,
             interpolation_cols: list,
             ff_cols: list,
             plotting: bool,
             num_self_call: int = 1,
             threshold_diff_time_track_data: int = 10,
             threshold_diff_time_fuel_data: int = 10,
             threshold_vertical_accuracy: float = 12,
             threshold_fuel=2,
             interpolate_over_one_zero_fuel=True,
             remove_zeros_in_a_row=2,
             threshold_length=30,
             ):
    """
    Adding fuel information to motion dataframe. The fuel data are interpolated to match the motion data. 
    TODO? Try DTW?

    Parameters
    ----------
    tracking_subset : DataFrame 
        DataFrame with tracking data properties (TODO refer to proper class). 
        The function is not optimised for speed, so it's recommended that tracking_subset does not contain pings for trips not listed in TripLogIds.
    fuel_subset : DataFrame 
        DataFrame with fuel properties (TODO refer to proper class) for same time period as tracking_subset.
    TripLogIds : ndarray 
        Array containing TripLogIds.
    id_mapping : dict 
        Dictionary with mappings between dumper id in tracking_subset and fuel.
    dev_outer_threshold : float
       If min/max points are too far away from each other (in meter)
    dev_points_threshold : float
       If subsequent points are too far away from each other (in meter)
    interpolation_cols : list
        List of columns to be interpolated between the two dataframes.
    interpolation_cols : list
        List of columns to be forward filled between the two dataframes.
    num_self_call : int 
        int that is a control for the number of self-calls that the function has done
    threshold_diff_time_track_data : int
        threshold for acceptable gap between data points in the track data (gps data)
    threshold_diff_time_fuel_data : int
        threshold for acceptable gap between data points in the fuel data (gps data)
    threshold_fuel : float
        threshold for saying that the fuel data is too low (wrong), and will either be dropped 
        or will use interpolation from the point before and after
    interpolate_over_one_zero_fuel : bool
        if True, then interpolate the value before and after the zero fuel value (below the threshold_fuel),
        split the that data where we cannot interpolate, cause of for instance diff time > threshold
        if False, no interpolation
    remove_zeros_in_a_row : int
        0: do nothing (no removing)
        1: remove every zero and split the data (below threshold)
        2: remove zero's when there are two in a row (or more and below threshold)

    Returns
    -------
    DataFrame
        tracking_subset DataFrame with fuel information
    """
    new_df = pd.DataFrame()

    for TripLogId in TripLogIds:
        #        for triptype in [0, 1, 2]: # type 0 is very often mis-aligned between fuel and ditio GPS data
        # & (tracking_subset.Type==triptype)]
        track = tracking_subset.loc[(tracking_subset.TripLogId == TripLogId)]
        anglemin, anglemax = 135, 225
        track['AngleChange'] = abs(track['Course'].diff())
        track['TimeDelta'] = (track['Timestamp'] -
                              track['Timestamp'].min()).dt.total_seconds()
        angle_index = (track.AngleChange > anglemin) & (
            track.AngleChange < anglemax)

        if len(track) > 0:

            # Preparing split in forward/backward motion
            track['TurningPoint'] = 0
            track.loc[angle_index, 'TurningPoint'] = 1
            track['subtrack'] = track['TurningPoint'].cumsum()

            starttime = track.Timestamp.min()
            endtime = track.Timestamp.max()

            if num_self_call == 1:
                fuel_index_time = (fuel_subset.data_Time > starttime) & (
                    fuel_subset.data_Time < endtime)
            else:
                fuel_index_time = (fuel_subset.index > starttime) & (
                    fuel_subset.index < endtime)

            fuel_index_id = fuel_subset.unitId == id_mapping[track.iloc[0].DumperId]
            track_fuel = fuel_subset.loc[fuel_index_time &
                                         fuel_index_id].copy()

        # The two data sets have similar time resolution. Before interpolation we remove tracks with very differen
        #length_ratio = np.abs(len(track)-len(track_fuel))/len(track_fuel)
        #    if (length_ratio < threshold_fraction) & (len(track_fuel) > 10):
        # lengths as a simple way of getting rid of problematic data.

        # The euclidian distance between deviation of outer points defining the tracks
            dev_outer = np.sqrt((track.x.min()-track_fuel.x.min())**2 + (track.x.max()-track_fuel.x.max())**2 +
                                (track.y.min()-track_fuel.y.min())**2 + (track.y.max()-track_fuel.y.max())**2)

            # The max distance between to subsequent coordinates in the GPS data
            dev_points = np.sqrt((track.x.diff()**2 + track.y.diff()**2).max())

            if (dev_outer < dev_outer_threshold) & (dev_points < dev_points_threshold) & (len(track_fuel) >= 6):

                if num_self_call == 1:
                    track_fuel.loc[:, 'TimeDiff'] = track_fuel['data_Time'].diff(
                    ).dt.total_seconds()
                    track_fuel.loc[:, 'FuelFromPrevious'] = track_fuel['data_Fuel'] / \
                        3600*track_fuel['TimeDiff']
                    track_fuel.loc[:,
                                   'FuelCumsum'] = track_fuel['FuelFromPrevious'].cumsum()
                    track.loc[:, 'DistanceCumsum'] = track.Distance.cumsum()
                    df1 = track_fuel.set_index('data_Time')
                else:
                    df1 = track_fuel

                df2 = track.set_index('Timestamp')

                #################################################################
                ##################### CONTROL of track data #####################
                #################################################################

                df2['diff_time'] = df2.index.to_series().diff().dt.total_seconds()
                df2['diff_time_gt_threshold'] = df2.diff_time > threshold_diff_time_track_data

                # Track data can also fail if the VerticalAccuracy is over a threshold
                not_satifying_vertical_accuracy = df2.VerticalAccuracy > threshold_vertical_accuracy
                df2['vertical_accuracy_gt_threshold'] = not_satifying_vertical_accuracy | not_satifying_vertical_accuracy.shift(
                    1)
                # df2['vertical_accuracy_gt_threshold'] = not_satifying_vertical_accuracy

                if np.sum(not_satifying_vertical_accuracy) > 0:
                    l = df2[['Distance', 'Altitude', 'VerticalAccuracy',
                             'vertical_accuracy_gt_threshold']]
                    frac_not_ok = np.sum(
                        not_satifying_vertical_accuracy)/len(df2)

                    # if frac_not_ok > 0.1:
                    if df2.Altitude.diff().abs().max() > 5:
                        embed(header='try make sense of it')

                df2['track_data_fails'] = df2['vertical_accuracy_gt_threshold'] | df2['diff_time_gt_threshold']

                # checking when alternating between failing and successful data
                df2['alternating_track_data_fails'] = df2.track_data_fails == df2.track_data_fails.shift(
                    -1)

                # Fixing the end points
                if df2['track_data_fails'][0]:
                    df2['alternating_track_data_fails'][0] = False
                else:
                    if df2['track_data_fails'][1] == False:
                        df2['alternating_track_data_fails'][0] = True

                time_track_data_fails = df2.index.to_series(
                )[df2.alternating_track_data_fails == False]

                if len(time_track_data_fails) % 2 == 1:
                    df2['alternating_track_data_fails'][-1] = not df2['alternating_track_data_fails'][-1]
                time_track_data_fails = df2.index.to_series(
                )[df2.alternating_track_data_fails == False]

                df1['track_data_fails'] = False
                for i in range(0, len(time_track_data_fails), 2):
                    df1['track_data_fails'] = df1['track_data_fails'] | df1.index.to_series(
                    ).between(time_track_data_fails[i], time_track_data_fails[i+1])

                # time_track_data_fails = df2.index.to_series(
                # )[df2.alternating_track_data_fails == False]
                # df1['track_data_fails'] = False
                # for i in range(0, len(time_track_data_fails) - 1, 2):
                #     df1['track_data_fails'] = df1['track_data_fails'] | df1.index.to_series(
                #     ).between(time_track_data_fails[i], time_track_data_fails[i+1])

                #################################################################
                ##################### CONTROL of fuel data ######################
                #################################################################

                # Some dates have several rows for the same timepoint
                df1 = df1.groupby('data_Time').mean()

                df1['diff_time'] = df1.index.to_series().diff().dt.total_seconds()
                df1['diff_xy'] = abs(df1.y.diff()).fillna(
                    0) + abs(df1.x.diff()).fillna(0)

                # If the same fuel number occur 2 times in a row
                df1['stand_still_fuel_1_forback'] = (df1.data_Fuel == df1.data_Fuel.shift(
                    1)) | (df1.data_Fuel == df1.data_Fuel.shift(-1))
                df1['stand_still_fuel_1_forback_below_threshold'] = ((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(
                    1) < threshold_fuel)) | ((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(-1) < threshold_fuel))

                # If the same fuel number occur 3 times in a row (or more)
                df1['stand_still_fuel_2_forback'] = ((df1.data_Fuel == df1.data_Fuel.shift(
                    1)) & (df1.data_Fuel == df1.data_Fuel.shift(-1)))
                df1['stand_still_fuel_2_forback_below_threshold'] = (((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(
                    1) < threshold_fuel)) & ((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(-1) < threshold_fuel)))

                df1['stand_still_fuel_2_forback'] = df1['stand_still_fuel_2_forback'] | ((df1.data_Fuel == df1.data_Fuel.shift(
                    1)) & (df1.data_Fuel == df1.data_Fuel.shift(2)))
                df1['stand_still_fuel_2_forback_below_threshold'] = (((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(
                    1) < threshold_fuel)) & ((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(2) < threshold_fuel)))

                df1['stand_still_fuel_2_forback'] = df1['stand_still_fuel_2_forback'] | ((df1.data_Fuel == df1.data_Fuel.shift(
                    -1)) & (df1.data_Fuel == df1.data_Fuel.shift(-2)))
                df1['stand_still_fuel_2_forback_below_threshold'] = (((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(
                    -1) < threshold_fuel)) & ((df1.data_Fuel < threshold_fuel) & (df1.data_Fuel.shift(-2) < threshold_fuel)))

                # Fuel data fails if the difference in timepoints is to much or fuel data is not moving
                df1['fuel_data_fails'] = (df1.diff_time > threshold_diff_time_fuel_data) | (
                    df1.diff_time.shift(-1) > threshold_diff_time_fuel_data)
                df1['fuel_data_fails'] = df1['fuel_data_fails'] | df1['stand_still_fuel_2_forback']

                # Splitting on all zeros if told
                if remove_zeros_in_a_row == 1:
                    df1['fuel_data_fails'] = df1['fuel_data_fails'] | (
                        df1['data_Fuel'] < threshold_fuel)
                elif remove_zeros_in_a_row == 2:
                    # splitting on zeros (under a fuel threshold) if there are two in a row
                    df1['fuel_data_fails'] = df1['fuel_data_fails'] | df1['stand_still_fuel_1_forback_below_threshold']

                ####################################################################
                ###################### Combining failed data #######################
                ####################################################################

                df1['data_fails'] = df1.fuel_data_fails | df1.track_data_fails

                # Finds the pair of timepoints, where data are successful
                time_slots_start = df1.index[df1['data_fails'].diff().fillna(
                    0).astype('int') == 1]
                time_slots_end = df1.index[df1['data_fails'].diff(
                ).shift(-1).fillna(0).astype('int') == 1]

                ##############################################################
                ####################### Interpolation ########################
                ##############################################################

                # interpolate over one zero (under threshold) fuel value
                if interpolate_over_one_zero_fuel:

                    # TODO: will just interpolate if the addition of the time diffs are under threshold.
                    # TODO: maybe change stand still forback to the new threshold one.
                    index_need_interpolation = ((df1['data_Fuel'] < threshold_fuel) & (df1['stand_still_fuel_1_forback'] == 0) &
                                                (df1['diff_time'] + df1['diff_time'].shift(-1) <= threshold_diff_time_fuel_data))
                    index_with_one_fuel_below_threshold = (df1['data_Fuel'] < threshold_fuel) & (
                        df1['stand_still_fuel_1_forback'] == 0)

                    # will just have interpolation on points that have not failing fuel data before or after
                    index_need_interpolation = index_need_interpolation & (
                        ((df1['data_fails'] | df1['data_fails'].shift(1)) | df1['data_fails'].shift(-1)) == 0)

                    index_need_interpolation[0] = False
                    index_need_interpolation[-1] = False

                    # Taking the average of point before and after and set the "updated" observation
                    interpolation_values = (df1.data_Fuel.shift(-1)[
                                            index_need_interpolation] + df1.data_Fuel.shift(1)[index_need_interpolation])/2
                    df1.data_Fuel[index_need_interpolation] = interpolation_values

                    # the data fails if there are zero fuel value (under threshold) on the left or right of the interval.
                    index_cannot_interpolate = index_with_one_fuel_below_threshold != index_need_interpolation
                    df1['data_fails'] = df1['data_fails'] | index_cannot_interpolate

                #################################################################
                ####################### Merging the data ########################
                #################################################################

                merged = aux.merge_resample_interpolate(df1.drop(['x', 'y'], axis=1), df2,
                                                        interpolation_cols=interpolation_cols,
                                                        ff_cols=ff_cols, frequency='s', plotting=False)
#                Compute differences so the quantities become comparable -> fuel per time unit and distance per time unit -> STATS
                merged.loc[:, 'DistanceCumsumDiff'] = merged.loc[:,
                                                                 'DistanceCumsum'].diff()
                merged.loc[:, 'FuelCumsumDiff'] = merged.loc[:,
                                                             'FuelCumsum'].diff()
                merged.loc[:, 'Timestamp'] = merged.index
                merged.loc[:, 'TimeDelta'] = (
                    merged['Timestamp']-merged['Timestamp'].min()).dt.total_seconds()

                forward_threshold_velocity = 2.5  # Ask Jacob about tuning of these
                forward_threshold_distance = 50
                forward_threshold_time = 30

                #kmax = int(merged.subtrack.max())
                kmax = int(merged.subtrack.max()) + 1
                merged['Forward'] = np.nan

                merged['data_Fuel_v2'] = merged.data_Fuel/3600

                merged['FuelCumsum'] = merged.data_Fuel_v2.cumsum()
                merged['FuelCumsumDiff'] = merged.data_Fuel_v2.diff()

                for k in range(kmax):
                    subtrack_index = merged.subtrack == k
                    subtrack = merged[subtrack_index].copy()
                    subtrack['DistanceCumsum'] = subtrack['DistanceCumsum'] - \
                        subtrack['DistanceCumsum'].min()
                    subtrack['TimeDelta'] = (
                        subtrack['Timestamp']-subtrack['Timestamp'].min()).dt.total_seconds()

                    if (subtrack.Speed.max() > forward_threshold_velocity) \
                        & (subtrack.DistanceCumsum.max() > forward_threshold_distance) \
                            & (subtrack.TimeDelta.max() > forward_threshold_time):
                        merged.loc[subtrack_index, 'Forward'] = 1
                    else:
                        merged.loc[subtrack_index, 'Forward'] = 0


                df1['track_data_fails'] = df1['track_data_fails'].astype(
                    'bool')

                a = df1[['data_Fuel', 'diff_xy',
                         'fuel_data_fails', 'track_data_fails']]
                yeah = df2[['Altitude', 'VerticalAccuracy', 'track_data_fails']]

                feat = ['data_Fuel', 'DistanceCumsum',
                        'Altitude', 'VerticalAccuracy']

                helper.plots_of_routes(df2, df1, TripLogId)

                ##############################################################################
                ##################### Splitting data in trustable tracks #####################
                ##############################################################################

                # If there is something failing in the data, then call the funciton again with splitted TripLogId
                if len(time_slots_start) > 0:

                    # Add more timeslots, if the data is ok in the start and the end
                    if df1['data_fails'][0] == False:
                        time_slots_start = time_slots_start.insert(
                            0, df1.index[0])
                        time_slots_end = time_slots_end.insert(0, df1.index[0])

                    if df1['data_fails'][-1] == False:
                        time_slots_start = time_slots_start.insert(
                            len(time_slots_start), df1.index[-1])
                        time_slots_end = time_slots_end.insert(
                            len(time_slots_end), df1.index[-1])
                        
                    new_trip_log_ids = []

                    # Wanna keep intervals [time_slots[0], time_slots[1], [time_slots[2], time_slots[3], ...
                    for i in range(0, len(time_slots_start) - 1, 2):
                        try:
                            trip_log_id = merged.TripLogId[str(time_slots_start[i]):str(
                                time_slots_end[i+1])][0] + f'__{i}'
                            new_trip_log_ids.append(trip_log_id)
                            df2.TripLogId[str(time_slots_start[i]):str(
                                time_slots_end[i+1])] = trip_log_id
                        except Exception as e:
                            print(
                                f"\n\n\n >>>>>>>>>>>>>>>>>>> FEIL: {e} \n\n\n")
                            embed()

                        # Switch to the correct pd.dataframe to send further
                        track.TripLogId = df2.TripLogId.values

                    splitted_merged = add_fuel(tracking_subset=track,
                                               fuel_subset=df1.copy(),
                                               TripLogIds=new_trip_log_ids,
                                               id_mapping=id_mapping,
                                               dev_outer_threshold=dev_outer_threshold,
                                               dev_points_threshold=dev_points_threshold,
                                               interpolation_cols=interpolation_cols,
                                               ff_cols=ff_cols,
                                               plotting=False,
                                               num_self_call=num_self_call + 1,
                                               threshold_diff_time_track_data=threshold_diff_time_track_data,
                                               threshold_diff_time_fuel_data=threshold_diff_time_fuel_data,
                                               threshold_vertical_accuracy=threshold_vertical_accuracy,
                                               threshold_fuel=threshold_fuel,
                                               interpolate_over_one_zero_fuel=interpolate_over_one_zero_fuel,
                                               remove_zeros_in_a_row=remove_zeros_in_a_row
                                               )

                    new_df = new_df.append(splitted_merged)

                else:
                    # If the whole track will be used (if not failing everywhere)

                    if len(merged) >= threshold_length:
                        if not sum(df1.data_fails) == len(df1):
                            new_df = new_df.append(merged)

                            if ('__' not in TripLogId):
                                print(
                                    '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FULLY TRIP OK!')
                        else:
                            print(
                                '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cannot trust any of the trip')

                if plotting & (len(time_slots_start) == 2):
                    if (dev_outer < dev_outer_threshold) & (dev_points < dev_points_threshold) & (len(track_fuel) > 10):

                        N = 6
                        fig, axs = plt.subplots(1, N, figsize=(5*N, 6))
                        fig.suptitle(
                            f'{TripLogId} outer dev: {dev_outer:.0f}  Points dev: {dev_points:.0f}', fontsize=16)

                        # x, y
                        i = 0
                        axs[i].scatter(
                            df2.x, df2.y, label=f'gps {df2.x.mean():.0f} {df2.y.mean():.0f}')
                        axs[i].scatter(
                            df2.iloc[0].load_x, df2.iloc[0].load_y, label=f'gps load', marker='v')
                        axs[i].scatter(
                            df2.iloc[0].dump_x, df2.iloc[0].dump_y, label=f'gps dump', marker='x')
                        axs[i].scatter(
                            df1.x, df1.y, label=f'fuel {df1.x.mean():.0f} {df1.y.mean():.0f}')
                        axs[i].ticklabel_format(style='plain')
                        axs[i].set_ylabel('y')
                        axs[i].set_xlabel('x')
                        df1.loc[:, 'TimeDelta'] = (
                            df1.index-df1.index.min()).total_seconds().values
                        df2.loc[:, 'TimeDelta'] = (
                            df2.index-df2.index.min()).total_seconds().values

                        # raw fuel vs merged fuel
                        i_fuel = i = 1
                        axs[i].scatter(df1.TimeDelta, df1.FuelCumsum, s=50,
                                       label='raw fuel', marker='x', color='black')
                        axs[i].plot(merged.TimeDelta, merged.FuelCumsum,
                                    label='merged', alpha=0.5)
                        axs[i].set_xlabel('Time [s]')
                        axs[i].set_ylabel('FuelCumsum')

                        # raw distance vs merged distance
                        i_dist = i = 2
                        axs[i].plot(df2.TimeDelta, df2.DistanceCumsum,
                                    label='raw gps', alpha=0.5, linestyle='dashed')
                        axs[i].plot(
                            merged.TimeDelta, merged.DistanceCumsum, label='merged', alpha=0.5)
                        axs[i].set_xlabel('Time [s]')
                        axs[i].set_ylabel('DistanceCumsum')

                        # Course
                        i = 3
#                            axs[i].scatter(merged.TimeDelta, merged.Course, label=f'Course')
                        axs[i].scatter(merged.TimeDelta,
                                       merged.Course_ff, label=f'Course ff')

#                            axs[-3].scatter(merged.TimeDiff, merged.AngleChange, label=f'AngleChange')
                        axs[i].scatter(track.loc[angle_index].TimeDelta, track.loc[angle_index].AngleChange,
                                       marker='D', color='black')
                        axs[i].scatter(track.TimeDelta, track.AngleChange,
                                       marker='o', alpha=0.5)
                        axs[i].axhline(360)
                        axs[i].axhline(anglemin)
                        axs[i].axhline(anglemax)
                        axs[i].set_xlabel('Time [s]')
                        axs[i].set_ylabel('Course')

                        # Velocity
                        i_vel = i = 4
                        axs[i].axhline(forward_threshold_velocity)
                        axs[i].scatter(track.loc[angle_index].TimeDelta, track.loc[angle_index].Speed,
                                       marker='D', color='black')

                        axs[i].set_xlabel('Time [s]')
                        axs[i].set_ylabel('Speed')

                        # x, y subtracks
                        i = 5

                        for k in merged.subtrack.unique():
                            subtrack = merged[merged.subtrack == k]
                            if subtrack.iloc[0].Forward == 1:
                                marker = 'x'
                            else:
                                marker = 'o'
                            axs[i].scatter(
                                subtrack.x, subtrack.y, label=subtrack.Type.mean(), marker=marker)
                            axs[i_vel].scatter(
                                subtrack.TimeDelta, subtrack.Speed, marker=marker, label=subtrack.DistanceCumsum.max())
                            axs[i_fuel].scatter(
                                subtrack.TimeDelta, subtrack.FuelCumsum, marker=marker, alpha=0.2, label=k)
                            axs[i_dist].scatter(
                                subtrack.TimeDelta, subtrack.DistanceCumsum, marker=marker, alpha=0.2, label=k)

                        for i in range(N):
                            axs[i].legend()

                        plt.savefig(f'tmp/{TripLogId}_012.png')
                        plt.close()
                else:
                    if plotting:
                        print('Will not plot, because the route has been splitted')

    return new_df


def merge_frames_together(data, date, fuel, destination, id_mapping, proj_info):
    from pandas.core.common import SettingWithCopyWarning
    import warnings
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    merge_tracks = 1

    if merge_tracks == 1:
        tracking_subset = data.copy().drop('index', axis=1)
        fuel_subset = aux.get_fuel_subset(fuel, tracking_subset, id_mapping)
        fuel_subset, _ = rdm.add_meter_columns(fuel_subset, proj_info=proj_info,
                                               load=False, dump=False, latlon_varnames=['latitude', 'longitude'])

        #check_cols ['Distance', 'DistanceDriven']
        interpolation_cols = ['FuelCumsum', 'Longitude', 'Latitude',
                              'Altitude', 'DistanceCumsum', 'Speed', 'x', 'y', 'Course', 'data_Fuel']
        ff_cols = ['TripLogId', 'Type', 'DumperId', 'DumperMachineName', 'DumperMachineTypeId',
                   'LoaderId', 'LoaderMachineName', 'LoaderMachineTypeId'
                   'CompanyId', 'ProjectCompanyId', 'ProjectId', 'ProjectNumber', 'ProjectExternalNumber', 'ProjectName', 'TaskId', 'TaskDescription',
                   'HorizontalAccuracy', 'VerticalAccuracy',
                   'LoadDateTime', 'DumpDateTime', 'MassTypeId', 'MassTypeName',
                   'MassTypeMaterial', 'Quantity', 'LoadGeoFenceId', 'DumpGeoFenceId',
                   'LoadLatitude', 'LoadLongitude', 'LoadHorizontalAccuracy',
                   'DumpLatitude', 'DumpLongitude', 'DumpHorizontalAccuracy',
                   'LoadGeoFenceLongitudes', 'LoadGeoFenceLatitudes',
                   'DumpGeoFenceLongitudes', 'DumpGeoFenceLatitudes',
                   'load_x', 'load_y', 'dump_x', 'dump_y', 'Course_ff', 'subtrack', '']

    # Change so ffill for all that's not interpolated?

        TripLogIds = [  # '616857635e10000000a9f3fd',
            # '61685f6e12b9300000007791',
            # '616848348642160000b03a8e',
            '61684a37c082ba0000a1a5e0',
            # '616845fe841aa40000b57bc9'
        ]

        TripLogIds_uniq = tracking_subset.TripLogId.unique()
        tracking_subset['Course_ff'] = tracking_subset['Course'].values
    #    TripLogIds = tracking_subset.TripLogId.unique()[0:3]
        merged = add_fuel(tracking_subset=tracking_subset,
                          fuel_subset=fuel_subset,
                          TripLogIds=TripLogIds_uniq,
                          id_mapping=id_mapping,
                          dev_outer_threshold=50,
                          dev_points_threshold=50,
                          interpolation_cols=interpolation_cols,
                          ff_cols=ff_cols,
                          plotting=False,
                          threshold_diff_time_track_data=10,
                          threshold_diff_time_fuel_data=10,
                          threshold_vertical_accuracy=12,
                          threshold_fuel=2,
                          interpolate_over_one_zero_fuel=True,
                          remove_zeros_in_a_row=2)
        filename = f'{destination}/{date}_to_{date}_merged.h5'
        merged.to_hdf(filename, key='merged')
    else:
        filename = f'{destination}/{date}_to_{date}_merged.h5'
        merged = pd.read_hdf(filename)
        print(f'Merged data loaded from {filename}')

    return merged


def main(date, folder_name, print_info=False):
    destination, eventsFile, trackingFile, date, id_mapping = specification(
        date, folder_name)
    # merged_df = load_data(destination)

    data, proj_info = get_ditio_data(
        eventsFile, trackingFile, id_mapping, destination)
    fuel = get_fuel_data(date, destination)
    merged = merge_frames_together(
        data, date, fuel, destination, id_mapping, proj_info)

    try:
        merged_forward = merged[merged.Forward > 0].copy()
    except AttributeError:
        print(f'FAILED: ' + date)
        return

    ########################################################################
    ####################### SAVE INFO OF DATESETS ##########################
    ########################################################################

    create_file = open(f'{destination}/information.txt', 'a+')
    create_file.close()

    with open(f'{destination}/information.txt', 'r') as f:
        contents = f.readlines()

    infile = open(f'{destination}/information.txt', 'r')
    insert_index = 0
    replacing = False

    insert_text = (
        date + f' {len(merged_forward.TripLogId.unique())} {len(merged_forward)}\n')

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

    ###################################################################
    ####################### SOME PREDICTIONS ##########################
    ###################################################################

    if print_info:
        # Quantity is stored per trip and not per trip type. Hence we reset quantity to 0 for all type 0 and 2 to 0
        index = merged_forward.Type.isin([0, 2])
        merged_forward.loc[index, 'Quantity'] = 0
        merged_forward['TypeTripLogId'] = merged_forward['TripLogId'] + \
            '_' + merged_forward['subtrack'].astype('int').astype('str')
        merged_group = merged_forward.groupby(
            'TypeTripLogId').size().sort_values()
        trip = merged_forward[merged_forward.TypeTripLogId ==
                              merged_group.index[-1]].copy()

        infile = open(helper.PATH_QUICK_RUN +
                      '/models_and_results10/features.txt', 'r')
        X_feat = []
        for line in infile:
            X_feat.append(line.strip())
        infile.close()

        for i in range(1, len(merged_group) + 1):
            trip = merged_forward[merged_forward.TypeTripLogId ==
                                  merged_group.index[-i]].copy()

            pred_fuel_route = helper.estimate_fuel_consumption_of_a_route(
                fully_route=trip,
                XGBmodel_name=helper.PATH_QUICK_RUN +
                'models_and_results10/models/without_2021-10-18.pkl',
                X_feat=X_feat,
            )
            fuel_actual = trip['FuelCumsum'].max() - trip['FuelCumsum'].min()

            print(
                f'data_points ({merged_group[-i]})            diff ({abs(pred_fuel_route - fuel_actual):.3f})')
            print(f'    preds: {pred_fuel_route: .4f}')
            print(f'    actual: {fuel_actual: .4f}')
            print(f'    last_part_trip_len: {merged_group[-i] % 30: .0f}')

            print('\n')


if __name__ == '__main__':

    folder_name = 'raw_2809'
    dates = [
        '2021-09-28',
        '2021-09-29',
        '2021-10-11',
        '2021-10-13',
        '2021-10-14',
        '2021-10-15',
        '2021-10-18',
        '2021-10-19',
        '2021-10-20',
        '2021-10-21',
        '2021-11-01',
        '2021-11-03',
        '2021-11-04',
        '2021-11-15',
    ]

    # for date in tqdm(dates):
    #     main(date, folder_name, print_info = False)

    date = '2021-10-19'
    main(date, folder_name, print_info=True)

  # get observations from 1 date pandas?
