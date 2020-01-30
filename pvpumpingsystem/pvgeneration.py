# -*- coding: utf-8 -*-
"""
Module for containing pvlib-python in a more convenient way.

WORK IN PROGRESS !!

@author: tylunel
"""

import pvlib


class PVGeneration:
    """
    Class representing the power generation through the photovoltaic system.
    It is a container of pvlib.ModelChain

    """
    # TODO: add a way to price the pv system (pv module + mppt? + rack?)

    def __init__(self,
                 pv_module,
                 surface_tilt=45,  # to adapt: 0 = horizontal, 90 = vertical
                 surface_azimuth=180,  # to adapt: 180 = South, 90 = East
                 albedo=0,  # to adapt: between 0 and 1
                 surface_type=None,
                 module_type='glass_polymer',  # to adapt
                 modules_per_string=1,  # to adapt
                 strings_per_inverter=1,  # to adapt
                 racking_model='open_rack',  # to adapt
                 losses_parameters=None,
                 # PV module glazing parameters (not always given in specs)
                 glass_params={'K': 4,  # to adapt: extinction coefficient[1/m]
                               'L': 0.002,  # to adapt: thickness [m]
                               'n': 1.526},  # to adapt: refractive index
                 price_per_module=100,
                 # weather
                 weather_selected,
                 ):

        # Definition of PV generator
        self.pvsystem = pvlib.pvsystem.PVSystem(
                    surface_tilt=surface_tilt,
                    surface_azimuth=surface_azimuth,
                    albedo=albedo,
                    surface_type=surface_type,
                    module=pv_module,
                    module_parameters={**dict(pv_module),
                                       **glass_params},
                    module_type=module_type,
                    modules_per_string=modules_per_string,
                    strings_per_inverter=strings_per_inverter,
                    inverter=None,  # fixed as AC pumps are not considered yet
                    inverter_parameters={'pdc0': 700},  # fixed (cf above)
                    racking_model=racking_model,
                    losses_parameters=losses_parameters,
                    name=None  # fixed (overwritten in PVGeneration object)
                    )

        # Import of weather
        self.weatherdata, self.metadata = pvlib.iotools.epw.read_epw(
                weather_selected, coerce_year=2005)
        self.location = pvlib.location.Location.from_epw(metadata)

        # Choices of models to use
        self.chain1 = pvlib.modelchain.ModelChain(
                    system=self.pvsystem,
                    location=self.location,
                    orientation_strategy=None,  # to adapt: can be ...
                    clearsky_model='ineichen',  # to choose
                    transposition_model='isotropic',  # to choose
                    solar_position_method='nrel_numpy',  # to choose
                    airmass_model='kastenyoung1989',  # to choose
                    dc_model='desoto',  # to choose between 'desoto' and 'cec'.
                    # Others will yield an error in pvpumpingsystem.
                    ac_model='pvwatts',  # to choose
                    aoi_model='physical',  # to choose
                    spectral_model='first_solar',  # to choose
                    temperature_model='sapm',  # to choose
                    losses_model='pvwatts',  # to choose
                    name=None)

        # Running of the PV generation model
        self.chain1.run_model(weather=self.weatherdata)
