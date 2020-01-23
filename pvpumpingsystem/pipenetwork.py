# -*- coding: utf-8 -*-
"""
Module for defining a pipes network.
Uses fluids module.

@author: Tanguy Lunel
"""
# TODO: check friction.material_roughness(self.material), does not
# seem to return coherent values

# TODO: Implement fittings attribute


import numpy as np
import pvpumpingsystem.waterproperties as wp
import fluids as fl


class PipeNetwork(object):
    """
    Class representing a simple hydraulic network.

    Attributes
    ----------
        h_stat: float,
            static head [m]
        l_tot: float,
            total length of pipes (not necessarily horizontal) [m]
        diam: float,
            fixed pipe diameter for all the network (propose to correct
            with fluids.piping.nearest_pipe()? ) [m]
        roughness: float,
            roughness of pipes [m]
        fittings: dict, NOT IMPLEMENTED YET.
            dictionnary of fittings, with angles as keys and
            number as values (check in fluids module how to define it)
    """

    def __init__(self, h_stat, l_tot, diam, roughness=0, material=None,
                 fittings=None, optimism=None):
        self.h_stat = h_stat
        self.l_tot = l_tot
        self.diam = diam
        self.material = material
        self.fittings = fittings  # Not used yet
        self.roughness = roughness

        if roughness == 0 and material is not None:
            self.roughness = fl.friction.material_roughness(self.material,
                                                            optimism=optimism)

    def __repr__(self):
        return str('h_stat=', self.h_stat, '\nl_tot=', self.l_tot,
                   '\ndiameter=', self.diam)

    def dynamichead(self, Qlpm, T=20, verbose=False):
        """
        Calculates the dynamic head of the pipe network according to the
        flow given Q, and using the Darcy-Weisbach equation.

        Parameters
        -----------
        Q: numeric,
            water flow in liter per minute [lpm]
        T: numeric,
            water temperature [°C]
        verbose: boolean,
            allows printing of Re numbers of the computing
            more details (for improvements):
                'https://stackoverflow.com/questions/5980042/
                how-to-implement-the-verbose-or-v-option-into-a-script

        Returns
        --------
            h_dyna: dynamic head [m]
        """
        if Qlpm == 0:
            if verbose:
                print('Reynolds number Re = ', 0)
            return Qlpm
        else:
            Q = Qlpm/60000  # flow [m3/s]
            Ap = self.diam**2*np.pi/4  # Area of pipe section [m2]
            viscosity_dyn = wp.water_prop('nuf', T+273.15)  # [Pa.s¸]
            speed = Q/Ap
            Re = speed*self.diam/viscosity_dyn

            darcycoeff = fl.friction.friction_factor(
                    Re, eD=self.roughness/self.diam)
            # https://en.wikipedia.org/wiki/Darcy%E2%80%93Weisbach_equation
            rho = wp.water_prop('rhof', T+273.15)
            press_loss = self.l_tot * darcycoeff * rho/2 * speed**2/self.diam
            h_dyna = press_loss/(rho*9.81)

        return h_dyna


if __name__ == '__main__':
    pipes1 = PipeNetwork(10, 100, 0.08, material='glass', optimism=True)
    print('roughness=', pipes1.roughness)
    h_dyn = pipes1.dynamichead(10)
    print(h_dyn)
