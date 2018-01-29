import os
from os import path
import numpy as np
from netCDF4 import Dataset
import pyproj
from shapely.ops import transform
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from shapely.prepared import prep
from functools import partial
from shyft import api
from shyft import shyftdata_dir
from .. import interfaces
from .time_conversion import convert_netcdf_time
import warnings

UTC = api.Calendar()


class ConcatDataRepositoryError(Exception):
    pass


class ConcatDataRepository(interfaces.GeoTsRepository):
    _G = 9.80665  # WMO-defined gravity constant to calculate the height in metres from geopotential

    # Constants used in RH calculation
    __a1_w = 611.21  # Pa
    __a3_w = 17.502
    __a4_w = 32.198  # K

    __a1_i = 611.21  # Pa
    __a3_i = 22.587
    __a4_i = -20.7  # K

    __T0 = 273.16  # K
    __Tice = 205.16  # K

    def __init__(self, epsg, filename, nb_pads=0, nb_fc_to_drop=0, nb_lead_intervals=None, fc_periodicity=1, selection_criteria=None, padding=5000.):
        # TODO: check all versions of get_forecasts
        # TODO: set ut get_forecast_ensembles
        # TODO: extend so that we can chose periodicity (i.e onle pick EC00 , etc)
        # TODO: check if relative humidity is in file
        # TODO: move configuration to config
        # TODO: _tranform, conversion parameters should be moved to config
        # TODO: documentation
        self.selection_criteria = selection_criteria
        # filename = filename.replace('${SHYFTDATA}', os.getenv('SHYFTDATA', '.'))
        filename = path.expandvars(filename)
        if not path.isabs(filename):
            # Relative paths will be prepended the data_dir
            filename = path.join(shyftdata_dir, filename)
        if not path.isfile(filename):
            raise ConcatDataRepositoryError("No such file '{}'".format(filename))

        self._filename = filename
        self.nb_pads = nb_pads
        self.nb_fc_to_drop = nb_fc_to_drop  # index of first lead time: starts from 0
        self.nb_lead_intervals = nb_lead_intervals
        self.fc_periodicity = fc_periodicity
        # self.nb_fc_interval_to_concat = 1  # given as number of forecast intervals
        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        self.padding = padding

        with Dataset(self._filename) as dataset:
            self._get_time_structure_from_dataset(dataset)

        # TODO: move all mappings to config file
        # Field names and mappings netcdf_name: shyft_name. See also _transform_raw
        self._shyft_map = {"dew_point_temperature_2m": "dew_point_temperature_2m",
                           "surface_air_pressure": "surface_air_pressure",
                           "relative_humidity_2m": "relative_humidity",
                           "air_temperature_2m": "temperature",
                           "precipitation_amount": "precipitation",
                           "precipitation_amount_acc": "precipitation",
                           "x_wind_10m": "x_wind",
                           "y_wind_10m": "y_wind",
                           "windspeed_10m": "wind_speed",
                           "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": "radiation"}

        self.var_units = {'dew_point_temperature_2m': ['K'],
                          'surface_air_pressure': ['Pa'],
                          "relative_humidity_2m": ['1'],
                          "air_temperature_2m": ['K'],
                          "precipitation_amount": ['kg/m^2'],
                          "precipitation_amount_acc": ['kg/m^2'],
                          "x_wind_10m": ['m/s'],
                          "y_wind_10m": ['m/s'],
                          "windspeed_10m": ['m/s'],
                          "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": ['W s/m^2']}

        self._shift_fields = ("precipitation_amount_acc",
                              "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time")

        self.create_geo_ts_type_map = {"relative_humidity": api.create_rel_hum_source_vector_from_np_array,
                                      "temperature": api.create_temperature_source_vector_from_np_array,
                                      "precipitation": api.create_precipitation_source_vector_from_np_array,
                                      "radiation": api.create_radiation_source_vector_from_np_array,
                                      "wind_speed": api.create_wind_speed_source_vector_from_np_array}

        self.series_type = {"relative_humidity": api.POINT_INSTANT_VALUE,
                            "temperature": api.POINT_INSTANT_VALUE,
                            "precipitation": api.POINT_AVERAGE_VALUE,
                            "radiation": api.POINT_AVERAGE_VALUE,
                            "wind_speed": api.POINT_INSTANT_VALUE}

        if self.selection_criteria is not None: self._validate_selection_criteria()

    def _get_time_structure_from_dataset(self, dataset):
        nb_fc_to_drop = self.nb_fc_to_drop
        fc_periodicity = self.fc_periodicity
        time = dataset.variables.get("time", None)
        lead_time = dataset.variables.get("lead_time", None)
        if not all([time, lead_time]):
            raise ConcatDataRepositoryError("Something is wrong with the dataset"
                                            "time or lead_time not found")
        time = convert_netcdf_time(time.units, time)
        self.time = time
        self.lead_time = lead_time[:]
        self.lead_times_in_sec = lead_time[:] * 3600.
        if self.nb_lead_intervals is None:
            self.nb_lead_intervals = len(self.lead_times_in_sec) - nb_fc_to_drop
        nb_lead_intervals = self.nb_lead_intervals
        if nb_fc_to_drop + nb_lead_intervals > len(self.lead_times_in_sec):
            raise  ConcatDataRepositoryError("'nb_fc_to_drop' + 'nb_lead_intervals' i too large")
        # self.fc_interval = time[fc_periodicity] - time[0]
        time_shift_with_drop = time + self.lead_times_in_sec[nb_fc_to_drop]
        # TODO: Errorhandling for idx_max required?
        idx_max = np.argmax(time[0] + self.lead_times_in_sec >= time_shift_with_drop[fc_periodicity])
        self.fc_len_to_concat = idx_max - nb_fc_to_drop

    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        """Get shyft source vectors of time series for input_source_types
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation, temperature..)
        geo_location_criteria: object, optional
            bbox or shapely polygon
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.
        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by time series name, where values are api vectors of geo
            located timeseries.
        """
        no_shift_fields = set([self._shyft_map[k] for k in self._shift_fields]).isdisjoint(input_source_types)
        if self.fc_len_to_concat < 0 \
            or no_shift_fields and self.nb_fc_to_drop + self.fc_len_to_concat > len(self.lead_time) \
            or not no_shift_fields and self.nb_fc_to_drop + self.fc_len_to_concat + 1 > len(self.lead_time):
                err_msg = "'nb_fc_to_drop={}' is too large for concatination".format(self.nb_fc_to_drop)
                raise ConcatDataRepositoryError(err_msg)
        with Dataset(self._filename) as dataset:
            fc_selection_criteria ={'forecasts_with_start_within_period': utc_period}
            extracted_data, geo_pts = self._get_data_from_dataset(dataset, input_source_types, fc_selection_criteria,
                                                                  geo_location_criteria,
                                                                  nb_lead_intervals=self.fc_len_to_concat, concat=True)
            # check if extra_intervals are required
            ta = list(extracted_data.values())[0][1] # time axis of first item
            ta_end = ta.total_period().end
            if ta_end < utc_period.end: # try to extend extracted data with remainder of last forecast
                sec_to_extend = utc_period.end - ta_end
                drop = self.nb_fc_to_drop + self.fc_len_to_concat
                idx = np.argmax(self.lead_times_in_sec[drop:] - self.lead_times_in_sec[drop] >= sec_to_extend)
                if idx == 0:
                    raise ConcatDataRepositoryError("The latest time in repository is earlier than the end of the "
                                                    "period for which data is requested")
                extra_data, _ = self._get_data_from_dataset(dataset, input_source_types,
                                    {'latest_available_forecasts': {'number of forecasts': 1,
                                                                    'forecasts_older_than': ta_end}},
                                    geo_location_criteria, nb_fc_to_drop=drop, nb_lead_intervals=idx,
                                    concat=False) # note: no concat here
                ta_extra = list(extra_data.values())[0][1][0]
                ta = api.TimeAxis(api.UtcTimeVector.from_numpy(np.append(ta.time_points, ta_extra.time_points[1:])))
                extracted_data = {k: (np.concatenate((extracted_data[k][0], np.squeeze(extra_data[k][0], axis=0))), ta)
                                                                                for k in list(extracted_data.keys())}
            return self._convert_to_geo_timeseries(extracted_data, geo_pts, concat=True)

    def get_forecast_ensembles(self, input_source_types, fc_selection_criteria, geo_location_criteria):
        k, v = list(fc_selection_criteria.items())[0]
        if k == 'forecasts_created_within_period':
            if not isinstance(v, api.UtcPeriod):
                raise ConcatDataRepositoryError(
                    "'forecasts_created_within_period' selection criteria should be of type api.UtcPeriod.")
        elif k == 'forecasts_with_start_within_period':
            if not isinstance(v, api.UtcPeriod):
                raise ConcatDataRepositoryError(
                    "'forecasts_with_start_within_period' selection criteria should be of type api.UtcPeriod.")
        elif k == 'forecasts_that_cover_period':
            if not isinstance(v, api.UtcPeriod):
                raise ConcatDataRepositoryError(
                    "'forecasts_that_cover_period' selection criteria should be of type api.UtcPeriod.")
        elif k == 'latest_available_forecasts':
            if not all([isinstance(v, dict), isinstance(v['number of forecasts'], int),
                        isinstance(v['forecasts_older_than'], int)]):
                raise ConcatDataRepositoryError(
                    "'latest_available_forecasts' selection criteria should be of type dict.")
        elif k == 'forecasts_at_reference_times':
            if not isinstance(v, list):
                raise ConcatDataRepositoryError(
                    "'forecasts_at_reference_times' selection criteria should be of type list.")
        else:
            raise ConcatDataRepositoryError("Unrecognized forecast selection criteria.")

        if k == 'forecasts_at_reference_times':
            return [self.get_forecast_ensembles(input_source_types, {'latest_available_forecasts':
                                                            {'number of forecasts': 1, 'forecasts_older_than': t_c}},
                                       geo_location_criteria)[0] for t_c in v]
        else:
            with Dataset(self._filename) as dataset:
                data, geo_pts = self._get_data_from_dataset(dataset, input_source_types, fc_selection_criteria,
                                                                      geo_location_criteria, concat=False)
                return self._convert_to_geo_timeseries(data, geo_pts, concat=False)

    def get_forecasts(self, input_source_types, fc_selection_criteria, geo_location_criteria):
        # Returns first ensemble member of each forecast
        return [fcst[0] for fcst in
                self.get_forecast_ensembles(input_source_types, fc_selection_criteria, geo_location_criteria)]

    def get_forecast_ensemble(self, input_source_types, utc_period,
                              t_c, geo_location_criteria=None):
        with Dataset(self._filename) as dataset:
            fc_selection_criteria = {'forecasts_that_cover_period': utc_period}
            time_slice, lead_time_slice, m_t = self._make_time_slice(self.nb_fc_to_drop, self.nb_lead_intervals,
                                                                     fc_selection_criteria)
            # time = self.time[time_slice]
            time = self.time[time_slice][m_t[time_slice]]
            ref_time = time[np.argmin(time <= t_c) - 1]
            if ref_time.size == 0:
                raise ConcatDataRepositoryError(
                    "Not able to find forecast that cover the requested period with the provided restrictions "
                    "'nb_fc_to_drop'={}, 'nb_lead_intervals'={}, 'fc_periodicity'={} and 't_c'={}". \
                        format(self.nb_fc_to_drop, self.nb_lead_intervals, self.fc_periodicity, t_c))
            fc_selection_criteria = {'forecasts_at_reference_times': [int(ref_time)]}
            return self.get_forecast_ensembles(input_source_types, fc_selection_criteria, geo_location_criteria)[0]

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria):
        # Returns first ensemble member of forecast
        """
        Parameters:
        see get_timeseries
        semantics for utc_period: Get the forecast closest up to utc_period.start
        """
        return self.get_forecast_ensemble(input_source_types, utc_period, t_c, geo_location_criteria)[0]

    def _validate_selection_criteria(self):
        s_c = self.selection_criteria
        if list(s_c)[0] == 'unique_id':
            if not isinstance(s_c['unique_id'], list):
                raise ConcatDataRepositoryError("Unique_id selection criteria should be a list.")
        elif list(s_c)[0] == 'polygon':
            if not isinstance(s_c['polygon'], (Polygon, MultiPolygon)):
                raise ConcatDataRepositoryError(
                    "polygon selection criteria should be one of these shapley objects: (Polygon, MultiPolygon).")
        elif list(s_c)[0] == 'bbox':
            if not (isinstance(s_c['bbox'], tuple) and len(s_c['bbox']) == 2):
                raise ConcatDataRepositoryError("bbox selection criteria should be a tuple with two numpy arrays.")
            self._bounding_box = s_c['bbox']
        else:
            raise ConcatDataRepositoryError("Unrecognized selection criteria.")

    def _get_data_from_dataset(self, dataset, input_source_types, fc_selection_criteria, geo_location_criteria,
                               nb_fc_to_drop=None, nb_lead_intervals=None, concat=False, ensemble_member=None):

        # validate input and adjust input_source_types
        ts_id, input_source_types, no_temp, rh_not_ok = self._validate_input(dataset, input_source_types, geo_location_criteria)

        # find geo_slice for slicing dataset
        geo_pts, m_xy, xy_slice, dim_grid = self._get_geo_slice(dataset, ts_id)

        # find time_slice and lead_time_slice for slicing dataset
        lead_times_in_sec = self.lead_times_in_sec
        if nb_fc_to_drop is None:
            nb_fc_to_drop = self.nb_fc_to_drop
        if nb_lead_intervals is None:
            if self.nb_lead_intervals is None:
                nb_lead_intervals = len(lead_times_in_sec) - nb_fc_to_drop
            else:
                nb_lead_intervals = self.nb_lead_intervals
        if nb_fc_to_drop + nb_lead_intervals > len(lead_times_in_sec):
            raise  ConcatDataRepositoryError("'nb_fc_to_drop' + 'nb_lead_intervals' i too large")
        issubset = True if len(lead_times_in_sec) > nb_fc_to_drop + nb_lead_intervals + 1 else False
        time_slice, lead_time_slice, m_t = self._make_time_slice(nb_fc_to_drop, nb_lead_intervals, fc_selection_criteria)
        # time = self.time[time_slice]
        time = self.time[time_slice][m_t[time_slice]]
        if np.size(time)==0:
            raise ConcatDataRepositoryError('No forecast found that meet condiotions')

        # Get data by slicing into dataset
        raw_data = {}
        for k in dataset.variables.keys():
            if self._shyft_map.get(k, None) in input_source_types:
                if k in self._shift_fields and issubset:  # Add one to lead_time slice
                    data_lead_time_slice = slice(lead_time_slice.start, lead_time_slice.stop + 1)
                else:
                    data_lead_time_slice = lead_time_slice

                data = dataset.variables[k]
                dims = data.dimensions
                data_slice = len(data.dimensions) * [slice(None)]
                if 'ensemble_member' in dims and ensemble_member is not None:
                    data_slice[dims.index("ensemble_member")] = ensemble_member

                data_slice[dims.index(dim_grid)] = xy_slice
                data_slice[dims.index("lead_time")] = data_lead_time_slice
                data_slice[dims.index("time")] = time_slice  # data_time_slice
                xy_slice_mask = [m_xy[xy_slice] if dim == dim_grid else slice(None) for dim in dims]
                time_slice_mask = [m_t[time_slice] if dim == 'time' else slice(None) for dim in dims]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
                    warnings.filterwarnings("ignore", message="invalid value encountered in less_equal")
                    pure_arr = data[data_slice][xy_slice_mask][time_slice_mask]

                if 'ensemble_member' not in dims: # add axis for 'ensemble_member'
                    pure_arr = pure_arr[:,:,np.newaxis,:]

                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    pure_arr = pure_arr.filled(np.nan)
                if np.isnan(pure_arr).any():
                    print("NaN found in pure_arr for {} see indices {}".format(k, np.unravel_index(np.argmax(np.isnan(pure_arr)), pure_arr.shape)))

                raw_data[self._shyft_map[k]] = pure_arr, k

        # TODO: do check if wind speed in dataset
        # Replace x/y-wind with wind speed
        if set(("x_wind", "y_wind")).issubset(raw_data):
            x_wind, _ = raw_data.pop("x_wind")
            y_wind, _ = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), "wind_speed"

        # Calculate relative humidity if required
        if rh_not_ok:
            if set(("surface_air_pressure", "dew_point_temperature_2m")).issubset(raw_data):
                sfc_p, _ = raw_data.pop("surface_air_pressure")
                dpt_t, _ = raw_data.pop("dew_point_temperature_2m")
                if no_temp:
                    sfc_t, _ = raw_data.pop("temperature")
                else:
                    sfc_t, _ = raw_data["temperature"]
                ncf_name_rh = next((n_nm for n_nm, s_nm in self._shyft_map.items() if s_nm == "relative_humidity"), None)
                raw_data["relative_humidity"] = self.calc_RH(sfc_t, dpt_t, sfc_p), ncf_name_rh
            else:
                raise ConcatDataRepositoryError("Not able to retrieve Relative Humidity from dataset")

        data_lead_time_slice = slice(lead_time_slice.start, lead_time_slice.stop + 1)
        extracted_data = self._transform_raw(raw_data, time, lead_times_in_sec[data_lead_time_slice], concat)
        return extracted_data, geo_pts

    def _validate_input(self, dataset, input_source_types, geo_location_criteria):
        # Validate geo_location criteria
        ts_id = None
        if geo_location_criteria is not None:
            self.selection_criteria = geo_location_criteria
        self._validate_selection_criteria()
        if list(self.selection_criteria)[0] == 'unique_id':
            ts_id_key = [k for (k, v) in dataset.variables.items() if getattr(v, 'cf_role', None) == 'timeseries_id'][0]
            ts_id = dataset.variables[ts_id_key][:]

        # TODO: do check if x_vind in dataset
        # Process input source types
        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("wind_speed")
            input_source_types.append("x_wind")
            input_source_types.append("y_wind")

        no_temp = False
        if "temperature" not in input_source_types: no_temp = True

        # Need extra variables to calculate Relative Humidity if not available in dataset
        ncf_nm = next((n_nm for n_nm, s_nm in self._shyft_map.items() if s_nm == "relative_humidity"), None)
        rh_not_ok = "relative_humidity" in input_source_types and (ncf_nm is None or ncf_nm not in dataset.variables)
        if rh_not_ok:
            if not isinstance(input_source_types, list):
                input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("relative_humidity")
            input_source_types.extend(["surface_air_pressure", "dew_point_temperature_2m"])
            if no_temp: input_source_types.extend(["temperature"])

        # Check units match
        unit_ok = {k: dataset.variables[k].units in self.var_units[k]
                   for k in dataset.variables.keys() if self._shyft_map.get(k, None) in input_source_types}
        if not all(unit_ok.values()):
            raise ConcatDataRepositoryError("The following variables have wrong unit: {}.".format(
                ', '.join([k for k, v in unit_ok.items() if not v])))

        return ts_id, input_source_types, no_temp, rh_not_ok

    def _get_geo_slice(self, dataset, ts_id):
        # Find xy slicing and z
        x = dataset.variables.get("x", None)
        y = dataset.variables.get("y", None)
        dim_grid = [dim for dim in dataset.dimensions if dim not in ['time', 'lead_time', 'ensemble_member']][0]
        if not all([x, y]):
            raise ConcatDataRepositoryError("Something is wrong with the dataset"
                                              " x/y coords or time not found")
        data_cs = dataset.variables.get("crs", None)
        if data_cs is None:
            raise ConcatDataRepositoryError("No coordinate system information in dataset.")
        x, y, m_xy, xy_slice = self._limit(x[:], y[:], data_cs.proj4, self.shyft_cs, ts_id)

        # Find height
        if 'z' in dataset.variables.keys():
            data = dataset.variables['z']
            dims = data.dimensions
            data_slice = len(data.dimensions) * [slice(None)]
            data_slice[dims.index(dim_grid)] = m_xy
            z = data[data_slice]
        else:
            raise ConcatDataRepositoryError("No elevations found in dataset")

        return api.GeoPointVector.create_from_x_y_z(x, y, z), m_xy, xy_slice, dim_grid

    def _limit(self, x, y, data_cs, target_cs, ts_id):
        """
        Parameters
        ----------
        x: np.ndarray
            X coordinates in meters in cartesian coordinate system
            specified by data_cs
        y: np.ndarray
            Y coordinates in meters in cartesian coordinate system
            specified by data_cs
        data_cs: string
            Proj4 string specifying the cartesian coordinate system
            of x and y
        target_cs: string
            Proj4 string specifying the target coordinate system
        Returns
        -------
        x: np.ndarray
            Coordinates in target coordinate system
        y: np.ndarray
            Coordinates in target coordinate system
        x_mask: np.ndarray
            Boolean index array
        y_mask: np.ndarray
            Boolean index array
        """
        # Get coordinate system for netcdf data
        data_proj = pyproj.Proj(data_cs)
        target_proj = pyproj.Proj(target_cs)

        if (list(self.selection_criteria)[0] == 'bbox'):
            # Find bounding box in netcdf projection
            bbox = np.array(self.selection_criteria['bbox'])
            bbox[0][0] -= self.padding
            bbox[0][1] += self.padding
            bbox[0][2] += self.padding
            bbox[0][3] -= self.padding
            bbox[1][0] -= self.padding
            bbox[1][1] -= self.padding
            bbox[1][2] += self.padding
            bbox[1][3] += self.padding
            bb_proj = pyproj.transform(target_proj, data_proj, bbox[0], bbox[1])
            x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
            y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

            # Limit data
            xy_mask = ((x <= x_max) & (x >= x_min) & (y <= y_max) & (y >= y_min))

        if (list(self.selection_criteria)[0] == 'polygon'):
            poly = self.selection_criteria['polygon']
            pts_in_file = MultiPoint(np.dstack((x, y)).reshape(-1, 2))
            project = partial(pyproj.transform, target_proj, data_proj)
            poly_prj = transform(project, poly)
            p_poly = prep(poly_prj.buffer(self.padding))
            xy_mask = np.array(list(map(p_poly.contains, pts_in_file)))

        if (list(self.selection_criteria)[0] == 'unique_id'):
            xy_mask = np.array([id in self.selection_criteria['unique_id'] for id in ts_id])

        # Check if there is at least one point extaracted and raise error if there isn't
        if not xy_mask.any():
            raise ConcatDataRepositoryError("No points in dataset which satisfy selection criterion '{}'.".
                                              format(list(self.selection_criteria)[0]))
        xy_inds = np.nonzero(xy_mask)[0]
        # Transform from source coordinates to target coordinates
        xx, yy = pyproj.transform(data_proj, target_proj, x[xy_mask], y[xy_mask])
        return xx, yy, xy_mask, slice(xy_inds[0], xy_inds[-1] + 1)

    def _make_time_slice(self, nb_fc_to_drop, nb_lead_intervals, fc_selection_criteria):
        time = self.time
        lead_times_in_sec = self.lead_times_in_sec
        # Find periodicity mask
        fc_periodicity = self.fc_periodicity
        m_t = np.zeros(time.shape, dtype=bool)
        m_t[::-fc_periodicity] = True # newest forecast last

        k, v = list(fc_selection_criteria.items())[0]
        nb_extra_intervals = 0
        if k == 'forecasts_created_within_period':
            time_slice = ((time >= v.start) & (time <= v.end))
            if not any(time_slice):
                raise ConcatDataRepositoryError(
                    "No forecasts found with creation time within period {}.".format(v.to_string()))
        elif k == 'forecasts_with_start_within_period':
            # shift utc period with nb_fc_to drop
            v_shift = api.UtcPeriod(int(v.start - lead_times_in_sec[nb_fc_to_drop]),
                                    int(v.end - lead_times_in_sec[nb_fc_to_drop]))
            time_slice = ((time >= v_shift.start) & (time <= v_shift.end))
            if not any(time_slice):
                raise ConcatDataRepositoryError(
                    "No forecasts found that start within period for period {} and "
                    "'fc_nb_to_drop'={}.".format(v_shift.to_string(),nb_fc_to_drop))
        elif k == 'forecasts_that_cover_period':
            # shift utc period with nb_fc_to drop
            v_shift_first = int(v.start - lead_times_in_sec[nb_fc_to_drop])
            # shift utc period with nb_fc_to drop + nb_lead_intervals
            v_shift_last = int(v.end - lead_times_in_sec[nb_fc_to_drop + nb_lead_intervals - 1])
            time_slice = ((time <= v_shift_first) & (time >= v_shift_last))
            if not any(time_slice):
                raise ConcatDataRepositoryError(
                    "No forecasts found that cover period {} with restrictions 'fc_nb_to_drop'={} "
                    "and 'nb_lead_intervals'={}.".format(v_shift.to_string(), nb_fc_to_drop, nb_lead_intervals))
        elif k == 'latest_available_forecasts':
            t = v['forecasts_older_than']
            n = v['number of forecasts']
            idx = np.argmin(time <= t) - 1
            if idx < 0:
                first_lead_time_of_last_fc = int(time[-1])
                if first_lead_time_of_last_fc <= t:
                    idx = len(time) - 1
                else:
                    raise ConcatDataRepositoryError(
                        "The earliest time in repository ({}) is later than or at the start of the period for which data is "
                        "requested ({})".format(UTC.to_string(int(time[0])), UTC.to_string(t)))
            if idx + 1 < n * fc_periodicity:
                raise ConcatDataRepositoryError(
                    "The number of forecasts available in repo ({}) and earlier than the parameter "
                    "'forecasts_older_than' ({}) is less than the number of forecasts requested ({}) " ""
                    "for the specified periodicity ({})".format(idx + 1, UTC.to_string(t), n, fc_periodicity))
            time_slice = slice(idx - n * fc_periodicity + 1, idx + 1)
        elif k == 'forecasts_at_reference_times':
            raise ConcatDataRepositoryError(
                "'forecasts_at_reference_times' selection criteria not supported yet.")
        lead_time_slice = slice(nb_fc_to_drop, nb_fc_to_drop + nb_lead_intervals)
        return time_slice, lead_time_slice, m_t

    def _transform_raw(self, data, time, lead_time, concat):
        # TODO; check time axis type (fixed ts_delta or not)
        # TODO: check robustness off all converiosn for flexible lead_times
        """
        We need full time if deaccumulating
        """

        def concat_t(t):
            t_stretch = np.ravel(np.repeat(t, self.fc_len_to_concat).reshape(len(t), self.fc_len_to_concat)
                                 + lead_time[0:self.fc_len_to_concat])
            dt_last = lead_time[-1] - lead_time[-2]
            if np.all(lead_time[1:]-lead_time[:-1] == dt_last): # fixed_dt time axis
                return api.TimeAxis(int(t_stretch[0]), int(t_stretch[1]) - int(t_stretch[0]), len(t_stretch))
            else: # point_type time axis
                return api.TimeAxis(api.UtcTimeVector.from_numpy(t_stretch.astype(int)), int(t_stretch[-1] + dt_last))

        def forecast_t(t, daccumulated_var=False):
            nb_ext_lead_times = len(lead_time) - 1 if daccumulated_var else len(lead_time)
            t_all = np.repeat(t, nb_ext_lead_times).reshape(len(t), nb_ext_lead_times) + lead_time[0:nb_ext_lead_times]
            return t_all.astype(int)

        def pad(v, t):
            if not concat:
                # Extend forecast by duplicating last nb_pad values
                if self.nb_pads > 0:
                    nb_pads = self.nb_pads
                    t_padded = np.zeros((t.shape[0], t.shape[1] + nb_pads), dtype=t.dtype)
                    t_padded[:, :-nb_pads] = t[:, :]
                    t_add = t[0, -1] - t[0, -nb_pads - 1]
                    # print('t_add:',t_add)
                    t_padded[:, -nb_pads:] = t[:, -nb_pads:] + t_add

                    v_padded = np.zeros((v.shape[0], t.shape[1] + nb_pads, v.shape[2]), dtype=v.dtype)
                    v_padded[:, :-nb_pads, :, :] = v[:, :, :, :]
                    v_padded[:, -nb_pads:, :, :] = v[:, -nb_pads:, :, :]
                else:
                    t_padded = t
                    v_padded = v
                dt_last = t_padded[0, -1] - t_padded[0, -2]
                if np.all(t_padded[:,1:] - t_padded[:,:-1] == dt_last): # fixed_dt time axis
                    return (v_padded, [api.TimeAxis(int(tp[0]), int(dt_last), len(tp)) for tp in t_padded])
                else: # point_type time axis
                    return (v_padded,
                            [api.TimeAxis(api.UtcTimeVector.from_numpy(tp), int(tp[-1] + dt_last)) for tp in t_padded])
            else:
                return (v, t)

        def concat_v(x):
            return x.reshape(-1, * x.shape[-2:])  # shape = (nb_forecasts*nb_lead_times, nb_ensemble_members, nb_points)

        def forecast_v(x):
            return x  # shape = (nb_forecasts, nb_lead_times, nb_ensemble_members, nb_points)

        def air_temp_conv(T, fcn):
            return fcn(T - 273.15)

        def prec_conv(v, ak, fcn):
            p = v
            if ak == "precipitation_amount_acc":
                # De-accumulate
                f = api.deltahours(1) / (lead_time[1:] - lead_time[:-1])  # conversion from mm/delta_t to mm/1hour
                res = fcn(np.clip((p[:, 1:, :, :] - p[:, :-1, :, :]) * f[np.newaxis, :, np.newaxis, np.newaxis], 0.0, 1000.0))
            elif ak == "precipitation_amount":
                # TODO: check with Yisak that this is understood correctly
                f = api.deltahours(1) / lead_time[1:]  # conversion from mm/delta_t to mm/1hour
                res = fcn(np.clip(p[:, 1:, :, :] * f[np.newaxis, :, np.newaxis, np.newaxis], 0.0, 1000.0))
            return res

        def rad_conv(r, fcn):
            dr = r[:, 1:, :, :] - r[:, :-1, :, :]
            return fcn(np.clip(dr / (lead_time[1:] - lead_time[:-1])[np.newaxis, :, np.newaxis, np.newaxis], 0.0, 5000.0))

        # Unit- and aggregation-dependent conversions go here
        if concat:
            convert_map = {"wind_speed": lambda v, ak, t: (concat_v(v), concat_t(t)),
                           "relative_humidity": lambda v, ak, t: (concat_v(v), concat_t(t)),
                           "temperature": lambda v, ak, t: (air_temp_conv(v, concat_v), concat_t(t)),
                           "radiation": lambda v, ak, t: (rad_conv(v, concat_v), concat_t(t)),
                           "precipitation": lambda v, ak, t: (prec_conv(v, ak, concat_v), concat_t(t))}
        else:
            convert_map = {"wind_speed": lambda v, ak, t: (forecast_v(v), forecast_t(t)),
                           "relative_humidity": lambda v, ak, t: (forecast_v(v), forecast_t(t)),
                           "temperature": lambda v, ak, t: (air_temp_conv(v, forecast_v), forecast_t(t)),
                           "radiation": lambda v, ak, t: (rad_conv(v, forecast_v), forecast_t(t, True)),
                           "precipitation": lambda v, ak, t: (prec_conv(v, ak, forecast_v), forecast_t(t, True))}
        res = {}
        for k, (v, ak) in data.items():
            res[k] = pad(*convert_map[k](v, ak, time))
        return res

    def _convert_to_geo_timeseries(self, data, geo_pts, concat):
        """Convert timeseries from numpy structures to shyft.api geo-timeseries.
        Returns
        -------
        timeseries: dict
            Time series arrays keyed by type
        """
        nb_ensemble_members = list(data.values())[0][0].shape[-2]
        if concat:
            geo_ts = [{key: self.create_geo_ts_type_map[key](ta, geo_pts, arr[:, j, :].transpose(), self.series_type[key])
                       for key, (arr, ta) in data.items()}
                       for j in range(nb_ensemble_members)]
        else:
            nb_forecasts = list(data.values())[0][0].shape[0]
            geo_ts = [[{key:
                        self.create_geo_ts_type_map[key](ta[i], geo_pts, arr[i,:,j,:].transpose(), self.series_type[key])
                       for key, (arr, ta) in data.items()}
                       for j in range(nb_ensemble_members)] for i in range(nb_forecasts)]
        return geo_ts

    @classmethod
    def calc_q(cls, T, p, alpha):
        e_w = cls.__a1_w * np.exp(cls.__a3_w * ((T - cls.__T0) / (T - cls.__a4_w)))
        e_i = cls.__a1_i * np.exp(cls.__a3_i * ((T - cls.__T0) / (T - cls.__a4_i)))
        q_w = 0.622 * e_w / (p - (1 - 0.622) * e_w)
        q_i = 0.622 * e_i / (p - (1 - 0.622) * e_i)
        return alpha * q_w + (1 - alpha) * q_i

    @classmethod
    def calc_alpha(cls, T):
        alpha = np.zeros(T.shape, dtype='float')
        # alpha[T<=Tice]=0.
        alpha[T >= cls.__T0] = 1.
        indx = (T < cls.__T0) & (T > cls.__Tice)
        alpha[indx] = np.square((T[indx] - cls.__Tice) / (cls.__T0 - cls.__Tice))
        return alpha

    @classmethod
    def calc_RH(cls, T, Td, p):
        alpha = cls.calc_alpha(T)
        qsat = cls.calc_q(T, p, alpha)
        q = cls.calc_q(Td, p, alpha)
        return q / qsat