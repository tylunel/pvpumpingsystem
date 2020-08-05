![CI general checks](https://github.com/tylunel/pvpumpingsystem/workflows/CI%20general%20checks/badge.svg?branch=master)

# pvpumpingsystem
*pvpumpingsystem* is a package providing tools for modeling and sizing photovoltaic water pumping systems.

It allows to find the quantity of water pumped from the pumping station characteristic given,
or to choose some elements of the pumping station depending on the consumption of water needed.

It relies on pvlib-python for the photovoltaic power generation, and implements
different models of pump for the simulation. 

# Installation 
*pvpumpingsystem* works with Python 3.5 and superior only.

## For active users

If you are new to Python we recommend to use Anaconda to install and use Python. You can find it here: https://www.anaconda.com/products/individual.

If you plan on editing the software and you are on Windows, install Git to simplify the download and the versioning of the package (on Linux, Git should be native).

Once you have Anaconda and git installed, open the command line interface 'Anaconda prompt', change directory to the one you want to install pvpumpingsystem in, and type in:
```
pip install -e git+https://github.com/tylunel/pvpumpingsystem#egg=pvpumpingsystem
```


To ensure 'pvpumpingsystem' and its dependencies are properly installed, run the tests by going to the directory of pvpumpingsystem and by running pytest:
```
cd src/pvpumpingsystem
pytest
```

## For more passive users
----Still to come----

# Hands-on start

Two examples of how the software can be run are in the folder 'examples'. 
The first shows how to model the water output from a given pumpset, and the second shows how to optimize the selection of one or more component on the pumping station.


# Contributions

All kind of contributions (documentation, testing, bug reports, new features, suggestions...) are highly appreciated.
They can be reported as issues, pull requests, or simple message via Github (prefered) or via mail of the maintainer.
