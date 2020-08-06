.. _introtutorial:

Beginner Guide
==============

This page aims at explaining the global functioning of pvpumpingsystem. 
Two examples are provided in `<\pvpumpingsystem\pvpumpingsystem\examples>`_ 
are also available with extensive docstring in order to quickly understand 
core features of *pvpumpingsystem*.

.. _modeling-paradigms:

Modeling paradigms
------------------

*pvpumpingsystem* strucure is based on object-oriented programming.
We recommend to users with little knowledge in object-oriented code to
invest some time to understand the power of this programming paradigm.

In order to increase the understandability of the code, the physical components 
of the PV pumping system corresponds to a class when possible, like for example 
the classes Pump(), MPPT(), PipeNetwork(), Reservoir() and PVGeneration(). 
Moreover, each of these classes are gathered into separate modules with 
appropriate names (pump.py, mppt.py, etc)
The previous objects are then gathered in the class PVPumpSystem() which allows 
running a comprehensive modeling of the pumping system. 


Most of the code concerns the simulation of the PV pumpins system,
except one module which is dedicated to methods allowing to size these systems.

.. _simulation:

Simulation
^^^^^^^^^^

In order to model a system, the simulation must take as input:

- PV system characterictics: (through a wrapper of pvlib-python)
  - truc
  - muche
- Motor-pump:
- Pipes:


.. _sizing:

Sizing
^^^^^^

Sizing methods are contained in a separate module named `sizing.py`.
These sizing methods are globally numerical method, relying on numerous 
simulations run according to an algorithm or to a factorial design.

This module can be expanded a lot as many strategies can be imagined to
size such a system according to the maximum computation time accepted, the 
optimization level wanted, the number of parameters to size, etc.
