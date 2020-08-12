.. _introtutorial:

Beginner Guide
==============

This page aims at explaining the global functioning of pvpumpingsystem.
Two examples are provided in `\pvpumpingsystem\pvpumpingsystem\examples`
are also available with extensive docstring in order to quickly understand
core features of *pvpumpingsystem*.


.. _modeling-paradigms:

Modeling paradigms
------------------

*pvpumpingsystem* strucure is based on object-oriented programming.
For users unfamiliar with the object-oriented programming, we recommend
to invest some time to understand and discover the power of this programming
paradigm.

In *pvpumpingsystem*, in order to increase the understandability of the code,
the physical components of the PV pumping system corresponds to a class
when possible, like for example the classes Pump(), MPPT(), PipeNetwork(),
Reservoir() and PVGeneration().
Moreover, each of these classes are gathered into separate modules with
appropriate names (`pump.py`, `mppt.py`, etc).
The previous objects are then gathered in the class PVPumpSystem() which
allows running partial or comprehensive modeling of the pumping system.

A separate module `sizing.py` is dedicated to methods allowing to size these
systems.



.. _simulation:

Simulation
----------

Inputs
^^^^^^

In order to model a system, the simulation must take as input:


- PV generation characterictics:

This part is basically a wrapper of pvlib-python functionnalities. The only
new feature added is the automatic selection of the PV module reference which
is the closest to the string given in ``pv_module_name`` attribute.

  - ``pv_module_name``: reference of the PV module

  - ``weather_data_and_metadata**``: weather file. Can be any format supported
    by pvlib-python

  - optional: ``surface_tilt``, ``surface_azimuth``, ``price_per_watt``,
    ``modules_per_string``, ``strings_in_parallel``, ``glass_params``,
    and other options like the models to use can be given as well.


- Electronics:

In version 1.0, only DC motor-pump are considered, therefore only MPPT-DC/DC
converters are considered (no inverter). It is called mppt, and it is actually
used only if the coupling method chosen is 'mppt'.

  - ``efficiency``, ``price`` are the 2 main attributes.


- Motor-pump:

The characteristics of the motor_pump must be given in a separate text file,
and the path to the text file given as input then. A limited DC motor-pump
database is natively available in `pvpumpingsystem/data/pump_files/`.
In this same folder, a file named `0_template_for_pump_specs` is proposed
to help enter another pump specifications. Good specification must give
the maximum data points possible (current I, voltage V, head TDH, flow rate Q).
Note that the class corresponding to the motor-pump has been shortened to
``Pump`` for simplification.

  - ``path``

  - ``modeling_method``

  - ``price``, ``idname``, ``motor_electrical_architecture`` can be given in
    the text file or directly in the code.


- Pipes:

The main characteristics of the PVPS pipes network are needed. Note that
for now, the fittings are not considered in the calculations. It is a feature
to develop later.

  - ``h_stat``: static head, or difference of height between pump and reservoir

  - ``l_tot``: total length of pipes

  - ``diam``: diameter of pipes


- Reservoir:

The water tank properties.

  - ``size``, ``price``, ``water_volume`` can be defined. If nothing is
    given, the size will be considered as null, which is equivalent to
    having no reservoir.


- Consumption:

This information consists in the hourly water consumption profile through
one year.

  - ``flow-rate`` must take an array with all hourly flow rates (8760 lines).
    To avoid constructing this long file manually, the two following attributes
    can also be used.

  - ``constant_flow`` considers the same flow rate given on every hour of the
    year.

  - ``repeated_flow`` repeats the flow rate sequence given until creating a
    8760 lines file. If a 24 lines array is given it will correspond to the
    daily consumption for the year.


- PVPumpSystem:

All preceding components and information are then gathered in a PVPumpSystem
class, from which it is possible to choose the type of coupling between the
PV generator and the pump ('direct' or 'mppt'), and some financial parameters
(``discount_rate`` or ``labour_price``).

Then the simulation can be run with ``run_model()`` function.


Outputs
^^^^^^^

All computation are made on a hourly basis, and most of the output are also
available hourly. Nevertheless the most interesting outputs are:

- Load Losses Probability (``llp``): the equivalent to the LPSP (Loss of
  Power Supply Probability), but here applied to the lack of water. It could be
  called Water Shortage Probability as well.

- Net Present Value (``npv``): Total cost of the system over its life cycle.
  Computed on condition that all ``price`` attributes of components were given
  before. The lifetime of the PVPS is considered as the same than the
  lifetime of the PV module by default.



.. _sizing:

Sizing
------

Sizing methods are contained in a separate module named `sizing.py`.
These sizing methods are globally numerical method, relying on numerous
simulations run according to an algorithm or to a factorial design.

This module can be expanded a lot as many strategies can be imagined to
size such a system according to the maximum computation time accepted, the
optimization level wanted, the number of parameters to size, etc.
