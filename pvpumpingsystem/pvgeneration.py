# -*- coding: utf-8 -*-
"""
Module for containing pvlib-python in a more convenient way.

@author: tylunel
"""

import pvlib
import difflib


# TODO: add way to directly give the pv module specs
class PVGeneration:
    """
    Class representing the power generation through the photovoltaic system.
    It is a container of pvlib.ModelChain [1].

    Attributes
    ----------
    pv_module_name: str,
        The name of the PV module used. Should preferentially follow the form:
        '(company_name)_(reference_code)_(peak_power)'

    weather_data_and_metadata: str or dict (containing pd.DataFrame and dict),
        Path to the weather file if it is .epw file,
        or the weather data itself otherwise.
        In the latter case, the dict must contains keys 'weather_data'
        and 'weather_metadata'. It should be created prior to the PVGeneration
        with the help of the corresponding pvlib function:
        (see https://pvlib-python.readthedocs.io/en/stable/api.html#io-tools).
        Possible weather file formats are numerous, including tmy2, tmy3, epw,
        and other more US related format. Note that Function 'get_pvgis_tmy'
        allows to get a tmy file according to the latitude and longitude of a
        location.

    price_per_watt: float, default is 2.5
        Price per watt for the module referenced by pv_module_name [US dollars]

    surface_tilt: float, default is 0
        Angle the PV modules have with ground [°]
        Overwritten if orientation_strategy is not None.

    surface_azimuth: float, default is 180 (oriented South)
        Azimuth of the PV array [°]
        Overwritten if orientation_strategy is not None.

    albedo: float, default 0
        Albedo of the soil around.

    modules_per_string: integer, default is 1
        Number of module put in a string.

    strings_in_parallel: integer, default is 1
        Number of PV module strings.
        Note that 'strings_in_parallel' is called 'strings_per_inverter' in
        pvlib.PVSystem. Name has been changed to simplify life of beginner
        user, but will complicate life of intermediate user.

    racking_model: str, default is 'open_rack'
        The type of racking for the PV array.s

    system: pvlib.PVSystem
        A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
        the connected set of modules, inverters, etc. Uses the previous
        attributes.

    location : Location
        A :py:class:`~pvlib.location.Location` object that represents
        the physical location at which to evaluate the model.

    orientation_strategy : None or str, default None
        The strategy for aligning the modules. If not None, overwrites the
        ``surface_azimuth`` and ``surface_tilt`` properties of the
        ``system``. Allowed strategies include 'flat',
        'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

    clearsky_model : str, default 'ineichen'
        Passed to location.get_clearsky.

    transposition_model : str, default 'haydavies'
        Passed to system.get_irradiance.

    solar_position_method : str, default 'nrel_numpy'
        Passed to location.get_solarposition.

    airmass_model : str, default 'kastenyoung1989'
        Passed to location.get_airmass.

    dc_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'desoto' and 'cec',
        unlike in pvlib.ModelChain because PVPS modeling needs a SDM.

    ac_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.inverter_parameters and system.module_parameters. Valid
        strings are 'snlinverter', 'adrinverter', 'pvwatts'. The
        ModelChain instance will be passed as the first argument to a
        user-defined function.

    aoi_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'physical',
        'ashrae', 'sapm', 'martin_ruiz', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    spectral_model: None, str, or function, default 'no_loss'
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm',
        'first_solar', 'no_loss'. The ModelChain instance will be passed
        as the first argument to a user-defined function.
        'no_loss' is recommended if the user is not sure that the weather
        file contains complete enough information like for example
        'precipitable_water'.

    temperature_model: None, str or function, default None
        Valid strings are 'sapm' and 'pvsyst'. The ModelChain instance will be
        passed as the first argument to a user-defined function.

    losses_model: str or function, default 'no_loss'
        Valid strings are 'pvwatts', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    name: None or str, default None
        Name of ModelChain instance.

    **kwargs
        Arbitrary keyword arguments. Included for compatibility, but not
        used.

    Reference
    ---------
    [1] William F. Holmgren, Clifford W. Hansen, Mark A. Mikofski,
    "pvlib python: a python package for modeling solar energy systems",
    2018, Journal of Open Source Software
    """

    def __init__(self,
                 # Weather
                 weather_data_and_metadata,  # path or weather data
                 # PV array parameters
                 pv_module_name,  # As precised as possible
                 price_per_watt=float('NaN'),  # in US dollars
                 surface_tilt=0,  # 0 = horizontal, 90 = vertical
                 surface_azimuth=180,  # 180 = South, 90 = East
                 albedo=0,  # between 0 and 1
                 modules_per_string=1,
                 strings_in_parallel=1,
                 racking_model='open_rack',
                 losses_parameters=None,
                 surface_type=None,
                 module_type='glass_polymer',
                 # PV module glazing parameters (not always given in specs)
                 glass_params={'K': 4,  # extinction coefficient[1/m]
                               'L': 0.002,  # thickness [m]
                               'n': 1.526},  # refractive index
                 # PV database
                 pv_database_name='cecmod',  # for advanced user only
                 # Models used:
                 orientation_strategy=None,
                 clearsky_model='ineichen',
                 transposition_model='isotropic',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model='desoto',  # to choose between 'desoto' and 'cec'.
                 ac_model='pvwatts',
                 aoi_model='physical',
                 spectral_model='no_loss',
                 temperature_model='sapm',
                 losses_model='pvwatts',
                 **kwargs
                 ):

        self.pv_database_name = pv_database_name
        self.price_per_watt = price_per_watt
        # goes through setter (picks right module and sets self.pv_module)
        self.pv_module_name = pv_module_name
        # Import of weather
        self.weather_data_and_metadata = weather_data_and_metadata

        # Definition of PV generator
        self.system = pvlib.pvsystem.PVSystem(
                    surface_tilt=surface_tilt,
                    surface_azimuth=surface_azimuth,
                    albedo=albedo,
                    surface_type=surface_type,
                    module=self.pv_module,
                    module_parameters={**dict(self.pv_module),
                                       **glass_params},
                    module_type=module_type,
                    modules_per_string=modules_per_string,
                    strings_per_inverter=strings_in_parallel,
                    inverter=None,  # fixed as AC pumps are not considered yet
                    inverter_parameters={'pdc0': 700},  # fixed (cf above)
                    racking_model=racking_model,
                    losses_parameters=losses_parameters,
                    name=None  # fixed (overwritten in PVGeneration object)
                    )

        # Choices of models to use
        self.modelchain = pvlib.modelchain.ModelChain(
                    system=self.system,
                    location=self.location,
                    orientation_strategy=orientation_strategy,
                    clearsky_model=clearsky_model,
                    transposition_model=transposition_model,
                    solar_position_method=solar_position_method,
                    airmass_model=airmass_model,
                    dc_model=dc_model,
                    ac_model=ac_model,
                    aoi_model=aoi_model,
                    spectral_model=spectral_model,
                    temperature_model=temperature_model,
                    losses_model=losses_model,
                    name=None)

    def __repr__(self):
        text = "PV generator made of: " + \
                 "\npv module: " + str(self.pv_module.name) + \
                 "\nnumber of modules: " + \
                 str(self.system.modules_per_string *
                     self.system.strings_per_inverter) + \
                 "\nin: " + str(self.location)
        return text

    @property  # getter
    def weather_data_and_metadata(self):
        return {'weather_data': self.weather_data,
                'weather_metadata': self.location}

    # setter: allows to change weather data and dependances at the same time
    @weather_data_and_metadata.setter
    def weather_data_and_metadata(self, value):
        if isinstance(value, str):  # assumed to be the path of weather
            self.weather_data, metadata = pvlib.iotools.epw.read_epw(
                    value, coerce_year=2005)
            self.location = pvlib.location.Location.from_epw(metadata)
        else:  # assumed to be dict with weather data (pd.df) and metadata
            self.weather_data = value['weather_data']
            self.location = pvlib.location.Location.from_epw(
                        value['weather_metadata'])
        if hasattr(self, 'modelchain'):  # adapt modelchain to new data
            self.modelchain.location = self.location
            # activates the setting of array tilt according to location:
            self.modelchain.orientation_strategy = \
                self.modelchain.orientation_strategy

    @property  # getter
    def pv_module_name(self):
        return self.pv_module.name

    # setter: allows to change weather data
    @pv_module_name.setter
    def pv_module_name(self, simple_name):
        # Retrieve SAM PV module database
        pv_database = pvlib.pvsystem.retrieve_sam(self.pv_database_name)
        # search pv_database to find the pv module which corresponds to name
        pv_idname = difflib.get_close_matches(simple_name,
                                              pv_database.columns,
                                              n=1,
                                              cutoff=0.5)  # %min of similarity
        if pv_idname == []:
            raise FileNotFoundError(
                'The pv module entered could not be found in the database.'
                'Check the name you entered. To give more chance to find it, '
                'write the name as follows:'
                '(company_name)_(reference_code)_(peak_power)')
        # Retrieve the pv module concerned from the pv database, and
        # convert in pandas.Series by selecting first column: iloc[:, 0]
        self.pv_module = pv_database[pv_idname].iloc[:, 0]  # to remove
        if hasattr(self, 'system'):
            # update system.module
            self.system.module = pv_database[pv_idname].iloc[:, 0]
            # update system.module_parameters:
            for key in dict(self.pv_module):
                self.system.module_parameters[key] = dict(self.pv_module)[key]

        self.price_per_module = self.price_per_watt * self.pv_module.STC

    def run_model(self):
        """
        Runs the modelchain of the PV generation.

        See pvlib.modelchain.run_model() for more details.
        """
        # Running of the PV generation model
        self.modelchain.run_model(weather=self.weather_data)
