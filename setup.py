from setuptools import setup

MAINTAINER_EMAIL='tanguy.lunel.1@ens.etsmtl.ca'
AUTHORS='Tanguy LUNEL'

setup(
   name='pvpumpingsystem',
   version='0.1',
   description='Module for simulating off-grid photovoltaic powered pumping station',
   license=GPL3,
   author=AUTHORS,
   author_email=MAINTAINER_EMAIL,
   url='https://github.com/tylunel/pvpumpingsystem',
   packages=['pvpumpingsystem'],  #same as name
   install_requires=['numpy', 'pandas', 'scipy', 'pvlib'], #external packages as dependencies
)