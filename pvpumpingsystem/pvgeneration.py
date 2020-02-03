# -*- coding: utf-8 -*-
"""
Module for containing pvlib-python in a more convenient way.

WORK IN PROGRESS !!

@author: tylunel
"""

# TODO: or add way to give the pv module specs

import pvlib
import difflib


class PVGeneration:
    """
    Class representing the power generation through the photovoltaic system.
    It is a container of pvlib.ModelChain.

    Attributes
    ----------
    pv_module_name: str,
        The name of the PV module used. Should preferentially follow the form:
        '(company_name)_(reference_code)_(peak_power)'

    weather_data: str or pvlib.weatherdata,
        Path to the weather file if string recognized,
        or the weather data itself if not string.

    price_per_module: float, default is 200
        Price of one PV module referenced in pv_module_name, in US dollars.

    surface_tilt: float, default is 0
        Angle the PV modules have with ground [°]

    surface_azimuth: float, default is 180 (oriented South)
        Azimuth of the PV array [°]

    albedo: float, default 0
        Albedo of the soil around.

    modules_per_string: integer, default is 1
        Number of module put in a string.

    strings_in_parallel: integer, default is 1
        Number of PV module strings.

    racking_model: str, default is 'open_rack'

    system : pvlib.PVSystem
        A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
        the connected set of modules, inverters, etc. Uses the previous
        attributes.

    location : Location
        A :py:class:`~pvlib.location.Location` object that represents
        the physical location at which to evaluate the model.

    orientation_strategy : None or str, default None
        The strategy for aligning the modules. If not None, sets the
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

    spectral_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm',
        'first_solar', 'no_loss'. The ModelChain instance will be passed
        as the first argument to a user-defined function.

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

    """

    def __init__(self,
                 # Weather
                 weather_data,  # path or weather data
                 # PV array parameters
                 pv_module_name,  # As precised as possible
                 price_per_module=200,  # in US dollars
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
                 spectral_model='first_solar',
                 temperature_model='sapm',
                 losses_model='pvwatts',
                 **kwargs
                 ):

        # Retrieve SAM PV module database
        pv_database = pvlib.pvsystem.retrieve_sam(pv_database_name)

        # search pv_database to find the pv module which corresponds to name
        pv_idname = difflib.get_close_matches(pv_module_name,
                                              pv_database.columns,
                                              n=1,
                                              cutoff=0.5)  # %min of similarity
        if pv_idname == []:
            raise FileNotFoundError(
                'The pv module entered could not be found in the database.'
                'Check the name you entered. To give more chance to find it, '
                'write the name as follows:'
                '(company_name)_(reference_code)_(peak_power)')

        # Retrieve the pv module concerned from the pv database
        # convert in pandas.Series by selecting first column: iloc[:, 0]
        self.pv_module = pv_database[pv_idname].iloc[:, 0]

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

        # Import of weather
        if isinstance(weather_data, str):  # assumed to be the path of weather
            self.weatherdata, metadata = pvlib.iotools.epw.read_epw(
                    weather_data, coerce_year=2005)
            self.location = pvlib.location.Location.from_epw(metadata)
        else:  # assumed to be dict with weather data (pd.df) and metadata
            self.weatherdata = weather_data['weatherdata']
            self.location = pvlib.location.Location.from_epw(
                    weather_data['metadata'])

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

    def run_model(self):
        """
        Runs the modelchain of the PV generation.
        """
        # Running of the PV generation model
        self.modelchain.run_model(weather=self.weatherdata)


if __name__ == '__main__':

    pv_power = PVGeneration(
        './data/weather_files/CAN_PQ_Montreal.Intl.AP' +
        '.716270_CWEC_truncated.epw',
        'Canada solar 270W'
        )
