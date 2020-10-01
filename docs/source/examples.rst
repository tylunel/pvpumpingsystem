.. _examples:

Examples
========

Three examples of how the software can be used are in the folder
``docs/examples``.
The examples are provided under two forms, as Jupyter Notebook files or as Python files.


Jupyter Notebook
----------------

Following examples can be run locally with Jupyter Notebook, or by clicking on the
corresponding icon in the upper right-hand corner of nbviewer pages, or by
accessing through the
`binder build <https://mybinder.org/v2/gh/tylunel/pvpumpingsystem/master>`_.


Simulation
^^^^^^^^^^
The first two examples focus on simulation. These examples are important to understand because the modeling tools used here are the core of the software. These tools can be used later to get programs that fit a more particular use (for ex.: sizing process, parametric study, etc).
For a given system, the examples show how to obtain the values of interest for the user (output flow rates, total water pumped in a year, loss of load probability (llp), net present value (npv), efficiencies and others):

`Basic usage example <https://nbviewer.jupyter.org/github/tylunel/pvpumpingsystem/blob/master/docs/examples/simulation_tunis_basic.ipynb>`_

`More advanced usage example <https://nbviewer.jupyter.org/github/tylunel/pvpumpingsystem/blob/master/docs/examples/simulation_tunis_advanced.ipynb>`_

Once you went through these 2 examples, you are quite ready to dive into the code and adapt it to your needs.

Sizing
^^^^^^
The third example shows how to use a sizing function written from the modeling tools presented in the two examples above. This function aims at optimizing the selection of the pump and the PV module, based on user requirements.

`Sizing example <https://nbviewer.jupyter.org/github/tylunel/pvpumpingsystem/blob/master/docs/examples/sizing_example.ipynb>`_


Python files
------------
These examples are also available in the form of python files in order to
freely adapt the code to your wishes. Directly check out in ``docs/examples``.
