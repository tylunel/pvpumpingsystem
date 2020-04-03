---
title: 'pvpumpingsystem: a python package for modeling and sizing photovoltaic water pumping systems'
tags:
  - Python
  - sizing
  - modeling 
  - water pumping
  - photovoltaics
  - solar energy
authors:
  - name: Tanguy R. Lunel
    orcid: 0000-0003-3008-1422
    affiliation: "1, 2"
  - name: Daniel Rousse
    orcid: 0000-0002-7247-5705
    affiliation: 1
affiliations:
 - name: Industrial research group in technologies of energy and energy efficiency (t3e), Department of Mechanical Sciences, Ecole de Technologie Supérieure Montreal
   index: 1
 - name: Department of Material Science and Engineering, Institut National des Sciences Appliquées Rennes
   index: 2
date: 25 March 2020
bibliography: paper.bib
---

# Summary

According to The World Health Organisation, one tenth of world’s population still lacks access to basic water supply. One of the reasons for this is the remoteness of these populations from modern water collection and distribution technologies, often coupled with an unfavourable socio-economic situation. Photovoltaic (PV) pumping technology makes it possible to respond both to this problem and to the criteria of sustainable development. However, these pumping systems must be carefully modeled and sized in order to make the water supply cost efficient and reliable. 
Pvpumpingsystem was conceived in order to tackle this issue. It is an open source package providing various tools aimed at facilitating the modeling and the sizing of photovoltaic powered water pumping systems. Even though the package is originally targeted at researchers and engineers, two practical examples are provided in order to help anyone to use pvpumpingsystem.
Python is the programming language used in the software, and the code is structured within an object-oriented approach. Continuous integration services allow to check for lint in the code and to automatize the tests. Each class and function are documented with reference to the literature when applicable. Pvpumpingsystem is released under a GPL-v3 license.
Pvpumpingsystem relies on already existing packages for photovoltaic and fluid mechanics modeling, namely “pvlib-python” [@pvlib-python] and “fluids” [@fluids]. pvpumpingsystem’s originality lies in the implementation of various motor-pump models for finite power sources and in the coupling of the distinct component models. In order to increase the understandability of the code, each physical component of the PV pumping system corresponds to a class, like for example the classes Pump(), MPPT(), PipeNetwork(), Reservoir() and PVGeneration(). The previous objects are then gathered in the class PVPumpSystem() which allows to run a comprehensive modeling of the pumping system. 
The main inputs to the model are hourly weather file, water source characteristics, expected water consumption profile, and specifications of photovoltaic array, motor-pump and water reservoir. Typical outputs are hourly flow rates, unused electric power, efficiencies of components, life cycle cost and loss of load probability. The sizing module provides functions to help choose the best combination of components in order to minimize the total life cycle cost. Nevertheless, sizing such complex systems is still an active field of research, and this module is subsequently expected to be expanded with time.
Pvpumpingsystem is the first academic contribution of a broader research program on photovoltaic water pumping launched in T3E research group at ETS Montreal, and is expected to grow with new features and accuracy assessment provided by experimental studies. The authors also want to give full access and help to anyone else interested in the use of the software.


# Acknowledgements

The authors would like to acknowledge Mr. Michel Trottier for his generous support to the T3E research group, as well as the NSERC and the FRQNT for their grants and subsidies. 
The first author acknowledges the contributions and fruitful discussions with Louis Lamarche and Sergio Gualteros that inspired and helped with the current work.

# Reference

[@pvlib-python]
[@fluids]

