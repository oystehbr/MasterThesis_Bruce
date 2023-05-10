import logging
import pandas as pd
import numpy as np
import geopy as gp
import geopy.distance

logger = logging.getLogger('producegraph')


def trips_processing(data, endpoint_threshold, remove_endpoints, variables=[]):
    """
    Initiates trip list from data, potentially removing the beginning and end of each trip.

    :param data: Input data from Ditio.
    :param endpoint_threshold: Minimum distance in m that the trip must traverse.
    :param remove_endpoints: If true, removes the ends of the trip.
    :return A modified data frame with trips added.
    """

    trips = data.sort_values('Timestamp').groupby(['TripLogId'])
    df_list = []
    for _, group in trips:
        # Add cumulative distance
        group['DistanceDriven'] = group['Distance'].cumsum()

        # Endpoint removal
        if remove_endpoints:
            dist_min, dist_max = group['DistanceDriven'].min(
            ), group['DistanceDriven'].max()

            lower = group['DistanceDriven'] > dist_min + endpoint_threshold
            upper = group['DistanceDriven'] < dist_max - endpoint_threshold
            if dist_max - endpoint_threshold < 0:
                continue  # The candidate trip was too short
            else:
                # Filter to only get internal points
                group = group[lower & upper]

        # Append the trip data to the data frame
        df_list.append(group)

    # TODO the section with the concatenation and the weird variables needs changing
    df = pd.concat(df_list, axis=0)
    if len(variables) < 1:
        variables = ['Latitude', 'Longitude', 'theta', 'delta_theta', 'TripLogId', 'Timestamp', 'Speed', 'Course',
                     'LoadLatitude', 'LoadLongitude', 'DumpLatitude', 'DumpLongitude', 'Distance', 'DistanceDriven',
                     'LoadGeoFenceId', 'DumpGeoFenceId'] + ['LoadDateTime', 'DumpDateTime', 'MassTypeId', 'MassTypeName',
                                                            'MassTypeMaterial', 'Quantity', 'LoaderMachineName']
    variables = list(pd.Series(variables)[list(
        pd.Series(variables).isin(df.columns))])
    df = df[variables].reset_index(drop=True)
    # TODO from Torkel - I don't follow this logic, but maybe a pandas-knower finds it reasonable. This function is called from preprocess_data in produce_graph_elemens,
    #                    and the index will be reset later anyway

    logger.debug(
        f"Removed {len(trips) - df['TripLogId'].unique().shape[0]} trips due to endpoint being under {endpoint_threshold}m away from startpoint.")

    return df


def add_meter_columns(input_df, proj_info=None, load=False, dump=False, latlon_varnames=['Latitude', 'Longitude']):
    """
    Add columns representing meter positions to match stored geocoordinate positions.

    :param input_df Original dataframe.
    :param proj_info For converting latlong to xy. If not given this object will be created.
    :param load if True, columns are added for load positions, named 'load_x' and 'load_y' using 'LoadLongitude', 'LoadLatitude'
    :param dump if True, columns are added for dump positions, named 'dump_x' and 'dump_y' using 'DumpLongitude', 'DumpLatitude'
    :return (Resulting data frame with new positions in 'x', 'y', columns, proj_info)
    """
    (x, y), proj_info = latlon_to_xy(input_df, proj_info=proj_info,
                                     latlon_varnames=latlon_varnames)

    # To make things easier to read below
    delta_x = proj_info["delta_x"]
    delta_y = proj_info["delta_y"]
    origin = proj_info["origin"]

    df = input_df.copy()
    df['x'] = x
    df['y'] = y

    if load:
        df['load_x'] = ((input_df['LoadLongitude'].to_numpy() -
                        origin[1]) / delta_x).astype(int)
        df['load_y'] = ((input_df['LoadLatitude'].to_numpy() -
                        origin[0]) / delta_y).astype(int)
    if dump:
        df['dump_x'] = ((input_df['DumpLongitude'].to_numpy() -
                        origin[1]) / delta_x).astype(int)
        df['dump_y'] = ((input_df['DumpLatitude'].to_numpy() -
                        origin[0]) / delta_y).astype(int)

    return df.copy(), proj_info


def latlon_to_xy(input_df, origin=None, proj_info=None, latlon_varnames: list = ['Latitude', 'Longitude']):
    """
    For the latitude, longitude columns in the input data frame, return new series
    that contain x,y changes
    """
    if proj_info is None:

        origin = calculate_origin(input_df)
        start = gp.Point(origin)

        # We now set up a local Cartesian coordinate system
        meters_to_angle = 1
        d = gp.distance.geodesic(meters=meters_to_angle)
        dist = d.destination(point=start, bearing=90)
        # 1 metre in "positive" longitude direction as vector in latlon space
        delta_x = abs(start.longitude - dist.longitude)
        dist = d.destination(point=start, bearing=0)
        # 1 metre in "positive" latitude direciton as vector in latlon space
        delta_y = abs(start.latitude - dist.latitude)
        proj_info = {'delta_x': delta_x, 'delta_y': delta_y, 'origin': origin}

        logger.debug(
            f"{meters_to_angle}m corresponds to {delta_x:.3e} deg longitude and {delta_y:.3e} deg latitude")

    x, y = latlon_to_xy_array(
        input_df[latlon_varnames[0]], input_df[latlon_varnames[1]], proj_info)

    return (x, y), proj_info


def calculate_origin(latlon_df):
    """ Find a suitable point (in geocoords) to use as origin, for simplification elsewhere. 

    :return (lat, lon) pair defining the origin."""
    return (latlon_df['Latitude'].min(), latlon_df['Longitude'].min())


def latlon_to_xy_array(lat, lon, proj_info):
    """
    """
    x = coord_to_metres(lon, proj_info['origin'][1], proj_info['delta_x'])
    y = coord_to_metres(lat, proj_info['origin'][0], proj_info['delta_y'])

    return x, y


def coord_to_metres(coord, origin, delta):
    """
    Translate from latitude-longitude pairs into (x,y) coordinates in Cartesian system.

    :param coord Coordinate array
    :param origin Geo-coordinate representation of (0,0)
    :param delta Offset in geo-coordinates corresponding to one unit in Cartesian system

    :return Cartesian coordinate array.
    """
    if isinstance(np.array(coord), np.ndarray) == False:
        coord = coord.to_numpy()
    return ((coord - origin) / delta).astype(int)  # TODO why is this an int???


def read_data(events, tracking):
    """
    Reads in the data as downloaded from Ditio.
    :return data: Pandas dataframe containing the combined data of gps signals and metadata for the trips.
    :return proj_info: The proj_info dictionary gives the conversions between the xy coordinate system and the longitude
    latitude. It is important to always use the the same proj_info after a xy coordinate system has been defined.
    """
    tracking = tracking.drop('Id', axis=1).reset_index()
    data = tracking.merge(events, how='inner',
                          left_on='TripLogId', right_on='Id')
    data["Date"] = data["Timestamp"].apply(lambda x: x.date())
    data.rename(columns={"Coordinate.Latitude": "Latitude",
                "Coordinate.Longitude": "Longitude"}, inplace=True)
    data, proj_info = add_meter_columns(data, load=True, dump=True)
    return data, proj_info
