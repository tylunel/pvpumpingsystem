[![CI general checks](https://github.com/tylunel/pvpumpingsystem/workflows/CI%20general%20checks/badge.svg)](https://github.com/tylunel/pvpumpingsystem/actions)
[![Coverage](https://codecov.io/gh/tylunel/pvpumpingsystem/branch/master/graph/badge.svg)](https://codecov.io/gh/tylunel/pvpumpingsystem)
[![Documentation Status](https://readthedocs.org/projects/pvpumpingsystem/badge/?version=latest)](https://pvpumpingsystem.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tylunel/pvpumpingsystem/master)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02637/status.svg)](https://doi.org/10.21105/joss.02637)

![Logo](/docs/images/logo_pvpumpingsystem.jpg =100x70)
# pvpumpingsystem
*pvpumpingsystem* is a package providing tools for modeling and sizing
photovoltaic water pumping systems.

![Schema of a PV pumping system](/docs/images/schema_pvps.jpg)

It can model the whole functioning of such pumping system on an hourly basis
and eventually provide key financial and technical findings on a year.
Conversely it can help choose some elements of the pumping station
depending on output values wanted (like daily water consumption and
acceptable risk of water shortage). Further details are provided on the [scope page of the documentation](https://pvpumpingsystem.readthedocs.io/en/latest/package_overview.html).


# Documentation
The full package documentation is available on readthedocs:

[pvpumpingsystem docs](https://pvpumpingsystem.readthedocs.io/en/latest/?badge=latest)


# Installation
*pvpumpingsystem* works with Python 3.5 and superior only.

## With pip

For a rapid installation with pip, type in a command line interface:
```
python -m pip install pvpumpingsystem
```

Consult the docs for more information on installation:
https://pvpumpingsystem.readthedocs.io/en/latest/installation.html


# Hands-on start

Three examples of how the software can be used are in the folder
'docs/examples'.

For a given system, the first two show how to obtain the outflows,
probability of water shortage, life cycle cost and many other results:

[Basic usage example](https://nbviewer.jupyter.org/github/tylunel/pvpumpingsystem/blob/master/docs/examples/simulation_tunis_basic.ipynb)

[More advanced usage example](https://nbviewer.jupyter.org/github/tylunel/pvpumpingsystem/blob/master/docs/examples/simulation_tunis_advanced.ipynb)

The third shows how to optimize the selection of one or more component
on the pumping station based on user requirements:

[Sizing example](https://nbviewer.jupyter.org/github/tylunel/pvpumpingsystem/blob/master/docs/examples/sizing_example.ipynb)


# Contributions

All kind of contributions (documentation, testing, bug reports,
new features, suggestions...) are highly appreciated.
They can be reported as issues, pull requests, or simple message via
Github (prefered) or via mail of the maintainer.
