.. _installation: pvpumpingsystem

Installation
============

Installing pvpumpingsystem can be done through different processes. Two of
them are detailled here, mainly thought for newcomers. Experienced users
can modify it to their liking.


    For people uncomfortable with package management, but who still plan on contributing or editing the code, follow the :ref:`anacondagit` instructions to install pvpumpingsystem along with Anaconda and Git.

    For people only interested in the use of the package, follow the :ref:`simple` instructions to install pvpumpingsystem alone.


Installing pvpumpingsystem is similar to installing most scientific python
packages, so in case of trouble see the :ref:`references` section
for further help.

Please see the :ref:`compatibility` section for information on the
optional packages that are needed for some pvpumpingsystem features.

.. _anacondagit:

Install pvpumpingsystem with Anaconda and Git
---------------------------------------------


- Anaconda:

The Anaconda distribution is an open source distribution providing Python
and others softwares and libraries useful for data science. Anaconda includes
many of the libraries needed for pvpumpingsystem (Pandas, NumPy, SciPy, etc).

Anaconda Python distribution is available at `<https://www.anaconda.com/download/>`_.

See `What is Anaconda? <https://www.anaconda.com/what-is-anaconda/>`_
and the `Anaconda Documentation <https://docs.anaconda.com/anaconda/>`_
for more information.


- Git:

Git is a version control system that widely help contribution and development
for open source softwares. Git should be native on most of Linux distribution,
but must be installed on Windows.

Git for Windows is available at `<https://gitforwindows.org/>`_.


- pvpumpingsystem:

Once you have Anaconda and git installed, open a command line interface
('Anaconda Prompt' on Windows, terminal in Linux and macOS), change
directory to the one you want to install pvpumpingsystem in, and type::

    pip install -e git+https://github.com/tylunel/pvpumpingsystem#egg=pvpumpingsystem



- Test pvpumpingsystem:

To ensure *pvpumpingsystem* and its dependencies are properly installed,
run the tests by going to the directory of pvpumpingsystem and by running
pytest::

    cd <relative/path/to/pvpumpingsystem/directory>
    pytest


.. _simple:

Install pvpumpingsystem alone
-----------------------------

.. note::

    Even if you decide not to use Anaconda or Git, you minimally need a Python
    version superior to 3.5, and to have pip and setuptools installed (installed
    by default with recent version of Python).

This second option simply uses pip::

    pip install pvpumpingsystem


If you want to install it in editable mode, use the `-e` option::

    pip install -e pvpumpingsystem


If you have troubles with the use of pip, here is the
`pip documentation <https://pip.pypa.io/en/stable/user_guide/#installing-packages>`_
to help you.


- Test pvpumpingsystem:

To ensure *pvpumpingsystem* and its dependencies are properly installed,
run the tests by going to the directory of pvpumpingsystem and by running
pytest::

    cd <relative/path/to/pvpumpingsystem/directory>
    pytest



.. _compatibility:

Compatibility
-------------

*pvpumpingsystem* is compatible with Python 3.5 and above.

Besides the libraries contained in Anaconda, *pvpumpingsystem* also requires:

* pvlib-python
* fluids
* numpy-financial

The full list of dependencies is detailled in
`setup.py <https://github.com/tylunel/pvpumpingsystem/docs/environment.rst>`_.


.. _references:

References
----------

.. note::

    This section was adapted from the pvlib-python documentation.
    Thanks to them for this useful listing!

Here are a few recommended references for installing Python packages:

* `Python Packaging Authority tutorial
  <https://packaging.python.org/tutorials/installing-packages/>`_
* `Conda User Guide
  <http://conda.pydata.org/docs/index.html>`_

Here are a few recommended references for git and GitHub:

* `The git documentation <https://git-scm.com/doc>`_:
  detailed explanations, videos, more links, and cheat sheets. Go here first!
* `Forking Projects <https://guides.github.com/activities/forking/>`_
* `Fork A Repo <https://help.github.com/articles/fork-a-repo/>`_
* `Cloning a repository
  <https://help.github.com/articles/cloning-a-repository/>`_


