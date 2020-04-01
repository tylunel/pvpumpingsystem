# pvpumpingsystem
"pvpumpingsystem" is a package providing tools for modeling and sizing photovoltaic water pumping systems.

It allows to find the quantity of water pumped from the pumping station characteristic given,
or to choose some elements of the pumping station depending on the consumption of water needed.

It relies on pvlib-python for the photovoltaic power generation, and implements
different models of pump for the simulation. 


# Installation

For manual installation, the following dependencies are required:

- Python 3.7 (https://www.python.org/download/releases/3.7/)
- fluids (pip install fluids)
- pvlib-python 0.7 (pip install pvlib==0.7)
- numpy-financial (pip install numpy_financial)

and common scientific dependencies that you can more easily install via Anaconda:
- conda (pip install conda)


# Hands-on start

Two examples of how the software can be run are in the folder 'examples'. 
The first shows how to model the water output from a given pumpset, and the second shows how to optimize the selection of one or more component on the pumping station.


# Contributions

All kind of contributions (documentation, testing, bug reports, new features, suggestions...) are highly appreciated.
They can be reported as issues, pull requests, or simple message via Github (prefered) or via mail of the maintainer.
