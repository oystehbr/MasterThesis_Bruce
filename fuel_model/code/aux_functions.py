# import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###################################
###### Data merger functions ######
###################################
# Should be merged with Helga's functions
def get_fuel_subset(fuel, tracking_subset, id_mapping):
    starttime = tracking_subset.Timestamp.min()
    endtime = tracking_subset.Timestamp.max()
    fuel_index_latlong = (fuel.latitude > 0)  # & (fuel.latitude > 59.9)
    fuel_index_time = (fuel.data_Time > starttime) & (fuel.data_Time < endtime)
    fuel_index_id = fuel.unitId.isin(
        [id_mapping[x] for x in tracking_subset.DumperId.unique() if x in id_mapping.keys()])
    fuel_index = fuel_index_time & fuel_index_id & fuel_index_latlong
    fuel_subset = fuel.loc[fuel_index].copy()
    return fuel_subset


def get_fuel_subset_10_min_data(fuel, tracking_subset, id_mapping):
    starttime = tracking_subset.Timestamp.min()
    endtime = tracking_subset.Timestamp.max()
    fuel_index_time = (fuel.data_Time > starttime) & (fuel.data_Time < endtime)

    fuel_index_id = fuel.unitId.isin(
        [id_mapping[x] for x in tracking_subset.DumperId.unique() if x in id_mapping.keys()])
    fuel_index = fuel_index_time & fuel_index_id
    fuel_subset = fuel.loc[fuel_index].copy()
    fuel_subset = fuel_subset.sort_values('data_Time')
    return fuel_subset


def merge_interpolate(df1, df2, interpolation_cols=[], ff_cols=[], plotting=False):
    """Resampling and interpolating one data frame to match the dataframe of another. The procedure is to first create a 
    dataframe with all data from the two. The new dataframe will have nan values at the columns/indices not present in the dataframe.
    The nans are then obtained by sorting the new dataframe in time and interpolating.

    Parameters
    ----------
    df1 : DataFrame
        dataframe with datetime index
    df2 : DataFrame
        dataframe with datetime index
    interpolation_cols : list
        List of columns to be interpolated between the two dataframes.
    interpolation_cols : list
        List of columns to be forward filled between the two dataframes.
    plotting : boolean
        Flag whether to generate plots

    Returns
    -------
    DataFrame
    """
    # idea from this oneliner which also removes the timestamps from one of the df
    #  frame1_r = frame1.reindex(frame1.index.union(frame2.index))\
    #                 .interpolate(method='index').reindex(frame2.index)

    merge = pd.concat([df1, df2], axis=1, join="outer")
    merge.sort_index(inplace=True)
    merge.loc[:, interpolation_cols] = merge.loc[:,
                                                 interpolation_cols].interpolate(method='index')
    merge.loc[:, ff_cols] = merge.loc[:, ff_cols].ffill()
    merge.loc[merge.index.isin(df1.index), 'df1index'] = True
    merge.loc[merge.index.isin(df2.index), 'df1index'] = False

    if plotting == True:
        cols = interpolation_cols
        N = len(cols)
        fig, axs = plt.subplots(1, N, figsize=(5*N, 6))
        for i in range(N):
            if cols[i] in df1.columns:
                axs[i].plot(df1.index, df1[cols[i]], marker='o', alpha=0.5)
            elif cols[i] in df2.columns:
                axs[i].plot(df2.index, df2[cols[i]], marker='o', alpha=0.5)

            axs[i].plot(merge.index, merge[cols[i]], marker='x',
                        color='green', alpha=0.5, label=cols[i])
            axs[i].legend()

    return merge


def interpol_fillforward_old(df, interpolation_cols, ff_cols, timestamps):
    union_timestamps = df.index.union(timestamps).unique()
    df_reindex = df.reindex(union_timestamps)
    df_interpolation_cols = [
        col for col in interpolation_cols if col in df.columns]
    df_interpolated = df_reindex.loc[:, df_interpolation_cols].interpolate(
        method='index')
    df_ff_cols = [col for col in ff_cols if col in df.columns]
    if len(df_ff_cols) > 0:
        df_interpolated[df_ff_cols] = np.nan
#        display(df[df_ff_cols].values)
#        display(df_interpolated.loc[df.index, df_ff_cols])

        df_interpolated.loc[df.index, df_ff_cols] = df[df_ff_cols].values
        df_interpolated.loc[:,
                            df_ff_cols] = df_interpolated.loc[:, df_ff_cols].ffill()
    df_reindex = df_interpolated.reindex(timestamps)

    return df_reindex


def interpol_fillforward(df, interpolation_cols, ff_cols, timestamps):
    union_timestamps = df.index.union(timestamps).unique()
    df_reindex = df.loc[~df.index.duplicated(), :]
    df_reindex = df_reindex.reindex(union_timestamps)
    df_interpolation_cols = [
        col for col in interpolation_cols if (col in df.columns)]
    df_interpolation_cols = []
    for col in interpolation_cols:
        if col in df.columns:
            df_interpolation_cols.append(col)
    df_interpolated = df_reindex.loc[:, df_interpolation_cols].interpolate(
        method='index')

    df_ff_cols = [col for col in ff_cols if col in df.columns]
    if len(df_ff_cols) > 0:
        df_interpolated[df_ff_cols] = np.nan
        for col in df_ff_cols:
            df_interpolated.loc[df.index, col] = df[col].values
        df_interpolated.loc[:,
                            df_ff_cols] = df_interpolated.loc[:, df_ff_cols].ffill()
    df_reindex = df_interpolated.reindex(timestamps)

    return df_reindex


def merge_resample_interpolate(df1, df2, interpolation_cols=[], ff_cols=[], frequency='s', plotting=False):
    """Resampling and interpolating two dataframes to same time series. The procedure is to first create a timeseries for the required 
    timestamps, add those to the two dataframes, use interpolation to replace all nan values, remove original timestamps and merge the two dataframes. 

    Parameters
    ----------
    df1 : DataFrame
        dataframe with datetime index
    df2 : DataFrame
        dataframe with datetime index
    frequency : string
        Frequency to do resampling over. If false, then the dataframe is returned with joint timesteps from both frames
    interpolation_cols : list
        List of columns to be interpolated between the two dataframes.
    interpolation_cols : list
        List of columns to be forward filled between the two dataframes.
    plotting : boolean
        Flag whether to generate plots

    Returns
    -------
    DataFrame
    """
    # idea from this oneliner which also removes the timestamps from one of the df
    #  frame1_r = frame1.reindex(frame1.index.union(frame2.index))\
    #                 .interpolate(method='index').reindex(frame2.index)

    timestamps = pd.date_range(start=df1.index.min(
    ), end=df1.index.max(), freq=frequency, tz='UTC')

    df1_reindex = interpol_fillforward(
        df1, interpolation_cols, ff_cols, timestamps)

    df2_reindex = interpol_fillforward(
        df2, interpolation_cols, ff_cols, timestamps)
    merge = pd.concat([df1_reindex, df2_reindex], axis=1, join="outer")

    if plotting == True:
        cols = interpolation_cols
        N = len(cols)
        fig, axs = plt.subplots(1, N, figsize=(5*N, 6))
        for i in range(N):
            if cols[i] in df1.columns:
                axs[i].plot(df1.index, df1[cols[i]], marker='o', alpha=0.5)
            elif cols[i] in df2.columns:
                axs[i].plot(df2.index, df2[cols[i]], marker='o', alpha=0.5)

            axs[i].plot(merge.index, merge[cols[i]], marker='x',
                        color='green', alpha=0.5, label=cols[i])
            axs[i].legend()

    return merge


def add_fuel(tracking_subset: pd.core.frame.DataFrame,
             fuel_subset: pd.core.frame.DataFrame,
             TripLogIds: np.ndarray,
             id_mapping: dict,
             dev_outer_threshold: float,
             dev_points_threshold: float,
             interpolation_cols: list,
             ff_cols: list,
             plotting: bool):
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



    Returns
    -------
    DataFrame
        tracking_subset DataFrame with fuel information
    """
    new_df = pd.DataFrame()
    for TripLogId in TripLogIds:
        # type 0 is very often mis-aligned between fuel and ditio GPS data
        for triptype in [0, 1, 2]:
            track = tracking_subset.loc[(tracking_subset.TripLogId == TripLogId) & (
                tracking_subset.Type == triptype)]
            if len(track) > 0:
                starttime = track.Timestamp.min()
                endtime = track.Timestamp.max()
                fuel_index_time = (fuel_subset.data_Time > starttime) & (
                    fuel_subset.data_Time < endtime)
                fuel_index_id = fuel_subset.unitId == id_mapping[track.iloc[0].DumperId]
                track_fuel = fuel_subset.loc[fuel_index_time & fuel_index_id].copy(
                )

            # The two data sets have similar time resolution. Before interpolation we remove tracks with very differen
            #length_ratio = np.abs(len(track)-len(track_fuel))/len(track_fuel)
            #    if (length_ratio < threshold_fraction) & (len(track_fuel) > 10):
            # lengths as a simple way of getting rid of problematic data.

            # The euclidian distance between deviation of outer points defining the tracks
                dev_outer = np.sqrt((track.x.min()-track_fuel.x.min())**2 + (track.x.max()-track_fuel.x.max())**2 +
                                    (track.y.min()-track_fuel.y.min())**2 + (track.y.max()-track_fuel.y.max())**2)

                # The max distance between to subsequent coordinates in the GPS data
                dev_points = np.sqrt(
                    (track.x.diff()**2 + track.x.diff()**2).max())

                if (dev_outer < dev_outer_threshold) & (dev_points < dev_points_threshold) & (len(track_fuel) > 10):
                    track_fuel.loc[:, 'TimeDiff'] = track_fuel['data_Time'].diff(
                    ).dt.total_seconds()
                    track_fuel.loc[:, 'FuelFromPrevious'] = track_fuel['data_Fuel'] / \
                        3600*track_fuel['TimeDiff']
                    track_fuel.loc[:, 'FuelCumsum'] = track_fuel['FuelFromPrevious'].cumsum(
                    )
                    track.loc[:, 'DistanceCumsum'] = track.Distance.cumsum()
                    df1 = track_fuel.set_index('data_Time')
                    df2 = track.set_index('Timestamp')

                    merged = merge_resample_interpolate(df1.drop(['x', 'y'], axis=1), df2,
                                                        interpolation_cols=interpolation_cols,
                                                        ff_cols=ff_cols, frequency='s', plotting=False)


#                Compute differences so the quantities become comparable -> fuel per time unit and distance per time unit -> STATS
                    merged.loc[:, 'DistanceCumsumDiff'] = merged.loc[:,
                                                                     'DistanceCumsum'].diff()
                    merged.loc[:, 'FuelCumsumDiff'] = merged.loc[:,
                                                                 'FuelCumsum'].diff()
                    merged.loc[:, 'Timestamp'] = merged.index
                    merged.loc[:, 'TimeDiff'] = (
                        merged['Timestamp']-merged['Timestamp'].min()).dt.total_seconds()

                    new_df = new_df.append(merged)

                if plotting == True:
                    xy_cols = [['TimeDiff', 'DistanceCumsum'], ['TimeDiff', 'FuelCumsum'], [
                        'TimeDiff', 'FuelCumsumDiff'], ['TimeDiff', 'Course']]
                    Nfixed = 2
                    N = len(xy_cols)

                    fig, axs = plt.subplots(
                        1, N+Nfixed, figsize=(5*(N+Nfixed), 6))
                    fig.suptitle(
                        f'{TripLogId}  Type: {triptype}  outer dev: {dev_outer:.0f}  Points dev: {dev_points:.0f}', fontsize=16)
                    axs[-1].scatter(df2.x, df2.y,
                                    label=f'gps {df2.x.mean():.0f} {df2.y.mean():.0f}')
                    axs[-1].scatter(df2.iloc[0].load_x, df2.iloc[0].load_y,
                                    label=f'gps load', marker='v')
                    axs[-1].scatter(df2.iloc[0].dump_x, df2.iloc[0].dump_y,
                                    label=f'gps dump', marker='x')
                    axs[-1].scatter(df1.x, df1.y,
                                    label=f'fuel {df1.x.mean():.0f} {df1.y.mean():.0f}')
                    axs[-1].legend()
                    axs[-1].ticklabel_format(style='plain')

                    if (dev_outer < dev_outer_threshold) & (dev_points < dev_points_threshold) & (len(track_fuel) > 10):
                        axs[-2].scatter(df2.TimeDiff, df2.DistanceCumsumDiff,
                                        label=f'gps distance cumsum diff')
                        axs[-2].scatter(df2.TimeDiff, df2.Speed,
                                        label=f'gps distance cumsum diff')

                        for i in range(N):
                            # /merged[xy_cols[i][0]].max()
                            x = merged[xy_cols[i][0]]
                            # /merged[xy_cols[i][1]].max()
                            y = merged[xy_cols[i][1]]
                            axs[i].scatter(x, y, marker='o', alpha=0.5)
                            axs[i].set_xlabel(xy_cols[i][0])
                            axs[i].set_ylabel(xy_cols[i][1])

                        img = axs[-1].scatter(merged.x, merged.y,
                                              label='merged', marker='x', c=merged.TimeDiff)
                        plt.colorbar(img)

                    plt.savefig(f'tmp/{TripLogId}_{triptype}.png')

    return new_df


def angle_change(df):
    # Convert course to angle-distances via (sin(angle), cos(angle)) and compute distance between subsequent pairs
    radians = df['Course']/360*np.pi
    delta_x = np.cos(radians).diff()
    delta_y = np.sin(radians).diff()
    distances = np.sqrt(delta_x**2 + delta_y**2)/np.pi*360
    return distances


def angle_integral(df):
    # Convert course to angle-distances via (sin(angle), cos(angle)) and compute distance between subsequent pairs
    radians = df['Course']/360*np.pi
    delta_x = np.cos(radians).diff()
    delta_y = np.sin(radians).diff()
    distances = np.sqrt(delta_x**2 + delta_y**2)
    # *df.TimeDelta Integrate over time or simply summing?
    integral = (distances).sum()/np.pi*360

    return integral


def sample_trips(merged,
                 trip_types=[1],
                 minlength=30,
                 maxlength=300,
                 Nsamples=500,
                 idling_threshold=1,
                 acceleration_threshold=1,
                 altitude_threshold=1,
                 Nplots=False):
    """
    Sample trip pieces from df.
    :param merged: Formatted dataframe with Ditio and Fuel data.
    :param trip_types: List of trip types to be considered. Trips are split between trip types
    :param minlength: Minimum length of samples i seconds
    :param maxlength: Maximum length of samples i seconds
    :param Nsamples: Numper of samples
    :idling_threshold: The threshold to distinguish between idling and non-idling in m/s. If zero, the machine is never idling (unless there are negative values) which is rare due to uncertainty. 1 m/s = 3.6 km/h
    :acceleration_threshold: The granularity in speed difference in m/s between bins to define acceleration/deceleration
    :altitude_threshold: The granularity in height difference in meters per bin used to define uphill/downhill movement
    :Nplots: Number of desired plots. Has to be smaller than Nsamples. Negative if no plots are desired
    :return: dataframe with statistics from the samples
    """

    statdata = pd.DataFrame()

    # Removing pings that are not in relevant trip types
    merged = merged[merged.Type.isin(trip_types)].copy()
    # Creating new triplogid that takes type into account
    merged['TypeTripLogId'] = merged['TripLogId'] + \
        '_' + merged['Type'].astype('int').astype('str')

    # List of Trips in df to sample from
    TypeTripLogIds = merged.TypeTripLogId.unique()

    # Sample with weighs according to lenght of track (TimeDiff is already computed with respect to type)
    probabilities = merged.groupby('TypeTripLogId').TimeDiff.max()
    probabilities = probabilities/probabilities.sum()

    for i in range(Nsamples):
        TypeTripLogId = np.random.choice(
            TypeTripLogIds, p=probabilities.values)
        trip = merged[merged.TypeTripLogId == TypeTripLogId]
        # Check if trip longer than minimum required sample length
        if trip.TimeDiff.max() >= minlength:
            # For a fixed minimum length, the sample subset cannot starter later than max(time) minus minimum length
            max_feasible_start = np.max(
                np.min([trip.TimeDiff.max()-minlength, maxlength]), 0)
            start = int(np.random.uniform(low=0, high=max_feasible_start))
            max_feasible_length = np.min(
                [maxlength, trip.TimeDiff.max()-start])
            length = int(np.random.uniform(
                low=minlength, high=max_feasible_length))

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
            statdata_row['Type'] = subset.iloc[0].Type
            statdata_row['StartTime'] = subset.iloc[0].Timestamp
            statdata_row['Month'] = subset.iloc[0].Timestamp.month
            statdata_row['LengthTime'] = length
            statdata_row['LengthDistance'] = subset['DistanceCumsum'].max(
            ) - subset['DistanceCumsum'].min()
            statdata_row['SpeedMean'] = subset['Speed'].mean()
            statdata_row['SpeedVariance'] = subset['Speed'].var()
            statdata_row['Quantity'] = subset.iloc[0].Quantity

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
            index_downhill = subset['AltitudeDiff'] < altitude_threshold
            statdata_row['AltitudeLoss'] = subset.loc[index_downhill,
                                                      'AltitudeDiff'].sum()
            statdata_row['AltitudeGain'] = subset.loc[index_uphill,
                                                      'AltitudeDiff'].sum()

            statdata_row['AltitudeDeltaEndStart'] = subset['Altitude'].iloc[-1] - \
                subset['Altitude'].iloc[0]
            statdata_row['AltitudeDeltaMaxMin'] = subset['Altitude'].max(
            ) - subset['Altitude'].min()

            # Average inclinations
            statdata_row['DownInclination'] = statdata_row['AltitudeLoss'] / \
                subset.loc[index_downhill, 'DistanceCumsumDiff'].sum()
            statdata_row['UpInclination'] = statdata_row['AltitudeGain'] / \
                subset.loc[index_uphill, 'DistanceCumsumDiff'].sum()

            # Time with idling and fuel during idling
            index_idling = subset['Speed'] < idling_threshold
            statdata_row['IdlingTime'] = subset.loc[index_idling,
                                                    'TimeDelta'].sum()
            statdata_row['IdlingFuel'] = (
                subset.loc[index_idling, 'TimeDelta']*subset.loc[index_idling, 'FuelCumsumDiff']).sum()

            # Sum of changes in direction
            statdata_row['SumRotation'] = angle_integral(subset)

            # Fuel
            statdata_row['Fuel'] = subset['FuelCumsum'].max() - \
                subset['FuelCumsum'].min()

            if statdata_row.Quantity > 0:
                statdata_row['FuelPerTonPerMeter'] = statdata_row['Fuel'] / \
                    statdata_row['Quantity']/statdata_row['LengthDistance']

            statdata = statdata.append(statdata_row, ignore_index=True)

            Nplots = min(Nsamples, Nplots)
            if (Nplots > 0) & (i % (Nsamples/Nplots) == 0):
                N = 5
                fig, axs = plt.subplots(1, N, figsize=(5*N, 6))

                i = 0
                axs[i].scatter(trip.x, trip.y, marker='o', alpha=0.5)
                axs[i].scatter(subset.x, subset.y, marker='x')
                axs[i].ticklabel_format(style='plain')

                i = 1
                xy_cols = ['TimeDiff', 'DistanceCumsum']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                i = 2
                xy_cols = ['TimeDiff', 'Speed']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                x = subset.loc[index_positive, xy_cols[0]]
                y = (subset.loc[index_positive, 'SpeedDiff'] *
                     subset['TimeDelta'].loc[index_positive]).cumsum()
                axs[i].plot(x, y, alpha=0.5)
                x = subset.loc[index_negative, xy_cols[0]]
                y = (subset.loc[index_negative, 'SpeedDiff'] *
                     subset['TimeDelta'].loc[index_negative]).cumsum()
                axs[i].plot(x, y, alpha=0.5)

                i = 3
                xy_cols = ['TimeDiff', 'Altitude']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                x = subset.loc[index_uphill, xy_cols[0]]
                y = subset.loc[index_uphill, 'AltitudeDiff'].cumsum(
                ) + subset.iloc[0, xy_cols[1]]
                axs[i].plot(x, y, alpha=0.5)
                x = subset.loc[index_downhill, xy_cols[0]]
                y = subset.loc[index_downhill, 'AltitudeDiff'].cumsum(
                ) + subset.iloc[0, xy_cols[1]]
                axs[i].plot(x, y, alpha=0.5)

                i = 4
                xy_cols = ['TimeDiff', 'FuelCumsum']
                x = subset[xy_cols[0]]
                y = subset[xy_cols[1]]
                axs[i].scatter(x, y, marker='o', alpha=0.5)
                axs[i].set_xlabel(xy_cols[0])
                axs[i].set_ylabel(xy_cols[1])

                x = subset.loc[index_idling, xy_cols[0]]
                y = (subset.loc[index_idling, 'TimeDelta']*subset.loc[index_idling,
                     'FuelCumsumDiff']).cumsum() + subset.iloc[0, xy_cols[1]]
                axs[i].plot(x, y, alpha=0.5)
                x = subset.loc[index_positive, xy_cols[0]]
                y = (subset.loc[index_positive, 'FuelCumsumDiff'] *
                     subset['TimeDelta'].loc[index_positive]).cumsum() + subset.iloc[0, xy_cols[1]]

                axs[i].plot(x, y, alpha=0.5)
                plt.show()
                plt.close()
                
    return statdata

