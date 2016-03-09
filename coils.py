#! /usr/bin/env python

# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2016 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================
"""
Magnetic field map calculation for Helmholtz and some other variant
coil's arrangement.

Related to the development of a Xenon hyperpolarizer at LCS

@author: C. Fernandez

Notes
-----
This program requires python version > 2.7 (tested with 3.5)
as well as numpy, matplotlib, scipy and pint libraries


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from matplotlib.patches import Circle
from scipy import optimize

from pint import UnitRegistry

# =============================================================================
# Units
# =============================================================================
ur = UnitRegistry()

# =============================================================================
# Constants
# =============================================================================
DEBUG = False

pi = np.pi
MU0 = 4 * pi * 1.0e-7 * ur.H / ur.m  # MU0 is the permeability constant in H/m


# =============================================================================
#  Axial and Radial components of a circular coil
# =============================================================================

def field(radius, z=0.0, r=0.0, pos=0.0):
    """Compute axial and radial component for a current loop

    Based on formulas taken from
    <http://www.netdenizen.com/emagnettest/offaxis/?offaxisloop>_

    Parameters
    ----------
    radius: float
        Radius of the coil

    z : float
        The distance, on axis, from the center of the current loop
        to the field measurement point.

    r : float
        The the radial distance from the axis of the current loop
        to the field measurement point.

    pos : float
        The position of the center of the coil on the z axis

    Returns
    -------
    Bz, Br : field in units of (Amperes * mu0 /2/radius)

    """
    r = abs(r)
    z = z - pos

    alp = r / radius
    bet = z / radius

    Q = ((1 + alp) ** 2 + bet ** 2)  # work on magnitude to improve performance
    m = 4 * alp / Q

    # K(k) is the complete elliptic integral function, of the first kind.
    # E(k) is the complete elliptic integral function, of the second kind.

    K = special.ellipkm1(1 - m)
    E = special.ellipe(m)

    # Bo is the magnetic field at the center of the coil (AMPERES MU0/2a)
    # Bz the magnetic field component that is aligned with the coil axis and
    # Br the magnetic field component that is in a radial direction.

    Bz = Br = 0.0
    if Q != 0.0 and Q - 4 * alp != 0.0 and z != pos:
        Bz = (E * (1 - alp ** 2 - bet ** 2)
              / (Q - 4 * alp) + K) / pi / Q ** 0.5

    if r > 0.000000 and Q != 0.0 and Q - 4 * alp != 0.0:
        gam = z / r
        Br = gam * (E * (1 + alp ** 2 + bet ** 2)
                    / (Q - 4 * alp) - K) / pi / Q ** 0.5

    return Bz, Br


# =============================================================================
# Field mapping
# =============================================================================

def fieldmap(Z, R, radius, pos):
    """Compute the field mapping for a coil

    Parameters
    ----------
    Z : array
        longitudinal coordinate matrice.

    R : array
        radial coordinate matrice.

    radius : float
        coil radius

    pos : float
        The position of the center of the coil on the z axis

    Returns
    -------
    Bz, Br : field map matrices in units of (Amperes * mu0 /2/radius)

    """

    lr, lz = Z.shape
    Bz = np.zeros((lr, lz))
    Br = np.zeros((lr, lz))
    ir0 = int(lr // 2)

    # be sure we work with the same units
    #  (and thus magnitude calculation is correct)
    pos = pos.to(radius.units).magnitude
    Zu = Z.to(radius.units).magnitude
    Ru = R.to(radius.units).magnitude
    radius = radius.magnitude

    # compute field in magnitude
    for i in range(lz):
        for j in range(ir0, lr):
            z = Zu[j, i]
            r = Ru[j, i]
            Bz[j, i], Br[j, i] = field(radius, z, r, pos)
            Bz[lr - j - 1, i] = Bz[j, i]
            Br[lr - j - 1, i] = -Br[j, i]

    Bsav = Bz.copy()
    Bz = Bz + Bsav[:, ::-1]

    Bsav = Br.copy()
    Br = Br - Bsav[:, ::-1]

    return Bz, Br


# =============================================================================
# coil
# =============================================================================

def coil(Z, R, coil_parameters):
    """Compute the field for a given coil

    Parameters
    ----------
    Z : array
        longitudinal coordinate matrice.

    R : array
        radial coordinate matrice.

    coil_parameters : list
        coil parameters

    Return
    ------
    Bz, Br: array
        Longitudinal and radial field matrices

    """

    Bz = np.zeros_like(Z) * ur.mT
    Br = np.zeros_like(R) * ur.mT

    # loop on the coils in the arrangement

    for id, par in enumerate(coil_parameters):

        if DEBUG:
            print("\tCoil ID: ", id, par)

        distance = par[0]
        radius = par[1]
        nturns = par[2]
        amperes = par[3]

        fact = 1.
        if distance < 0.001 * ur.mm:
            fact = .5  # This factor is to divide the center coil in two parts

        Bzt, Brt = fieldmap(Z, R, radius, distance)
        Bzt = Bzt * amperes * MU0 / 2. / radius
        Brt = Brt * amperes * MU0 / 2. / radius
        Bz = Bz + Bzt * nturns * fact
        Br = Br + Brt * nturns * fact

    return Bz.to('mT'), Br.to('mT')


# =============================================================================
# Helmholtz coils (two coils)
# =============================================================================

def helmholtzcoil(radius, amperes, var):
    """Helmholtz coils parameters

    Parameters
    ----------
    radius : float
        coil radius

    amperes : float
        current in the coils

    var : list
        coil parameters (turns: turns,
                         dist: relative distance between coils)

    Returns
    -------
    arrangement : list
        A list of parameters for each coil (actually half, as the arrangement
        is always symmetrical with respect to the origin on the z axis)

    """
    nturns = var[0]
    distance = radius * var[1]  # distance from the center of the arrangement

    # make sure the number of turns is an integer and adjust amperes accordingly
    iturns = int(nturns)
    amperes = amperes * nturns / iturns

    arrangements = [(+distance, radius, iturns, amperes)]

    return arrangements


# =============================================================================
# Maxwell coils (3 coils)
# =============================================================================

def maxwellcoil(radius, amperes, var):
    """Helmholtz coils parameters

    Parameters
    ----------
    radius : float
        coil radius

    amperes : float
        current in the coils

    var : list
        coil parameters (turns: turns,
                         dist: relative distance between coils)

    Returns
    -------
    arrangement : list
        A list of parameters for each coil (actually half, as the arrangement
        is always symmetrical with respect to the origin on the z axis)

    """

    nturns = var[0]

    # make sure the number of turns is an integer and adjust amperes accordingly
    iturns = int(nturns)
    amperes = amperes * nturns / iturns

    distance = radius * var[1]
    smallradius = radius * var[2]
    smallnturns = int(iturns * var[3])

    arrangements = [(0.00000001 * radius.units, radius, iturns, amperes),
                    (+distance, smallradius, smallnturns, amperes)]

    return arrangements


# =============================================================================
# five coils
# =============================================================================

def fivecoils(radius, amperes, var):
    """Helmholtz coils parameters

    Parameters
    ----------
    radius : float
        coil radius

    amperes : float
        current in the coils

    var : list
        coil parameters (turns: turns,
                         dist: relative distance between coils)

    Returns
    -------
    arrangement : list
        A list of parameters for each coil (actually half, as the arrangement
        is always symmetrical with respect to the origin on the z axis)

    """
    nturns = var[0]

    # make sure the number of turns is an integer and adjust amperes accordingly
    iturns = int(nturns)
    amperes = amperes * nturns / iturns

    distance1 = radius * var[1]
    distance2 = radius * var[2]
    interradius = radius * var[3]
    smallradius = radius * var[4]
    internturns = int(iturns * var[5])
    smallnturns = int(iturns * var[6])

    arrangements = [(0.000001 * radius.units, radius, iturns, amperes),
                    (+distance1, interradius, internturns, amperes),
                    (+distance2, smallradius, smallnturns, amperes)]

    return arrangements


# ===============================================================================
# OPTIMISATION'S PROGRAM
# ===============================================================================
if __name__ == '__main__':

    FIT =  True # False

    # select the type of coil arrangements
    ARRANGEMENTS = [
        # (name, callable, nturns in the center coil, [,...])
        ("HELMHOTZ", helmholtzcoil, [
            441., 0.5]),
         ("MAXWELL", maxwellcoil, [
             320., (3. / 7.) ** .5,
             (4. / 7.) ** .5,
             49. / 64.]),
        ("FIVE-COILS", fivecoils, [
            152.,
            186.3 / 300.,
            361.5 / 300.,
            601. / 600.,
            391. / 600.,
            223. / 152.,
            151. / 152.]),
    ]

    # =========================================================================
    # Basic definition of the main coil
    # =========================================================================
    SECTION = 2.5 * (ur.mm) ** 2  # wire section in mm^2
    FIELD_REQUEST = 5. * ur.mT  # in mT
    RADIUS = 30 * ur.cm  # coil radius in centimeter
    AMPERES = 5 * ur.A  # current in a coil in A

    wire_diameter = 2.0 * np.sqrt(SECTION / pi)

    n = 101 # must be odd

    # make a grid to sample the space around the coils
    z = np.linspace(-2.5 * RADIUS.magnitude, 2.5 * RADIUS.magnitude,
                    2 * n + 1) * RADIUS.units
    r = np.linspace(-2.5 * RADIUS.magnitude, 2.5 * RADIUS.magnitude,
                    2 * n + 1) * RADIUS.units
    Z, R = np.meshgrid(z, r) * RADIUS.units

    lz = z.size
    lr = r.size
    rangz = np.where((z > -20 * ur.cm) & (z < 20 * ur.cm))[0]
    rangr = np.where((r > -10 * ur.cm) & (r < 10 * ur.cm))[0]

    ir0 = lr // 2
    iz0 = lz // 2

    nzi = rangz[0] - 1
    nzf = rangz[-1] + 1

    nri = rangr[0] - 1
    nrf = rangr[-1] + 1

    # =========================================================================
    # Compute the field for the different coil arrangements
    # =========================================================================
    Bzs = []
    Brs = []
    for item in range(len(ARRANGEMENTS)):

        name = ARRANGEMENTS[item][0]

        print("*" * 80)
        print(" Arrangement type: {}".format(name))
        print("*" * 80)
        print()

        callable = ARRANGEMENTS[item][1]
        var = ARRANGEMENTS[item][2]


        def minimized_func(v):

            for i in v:
                # avoid zero and negative numbers
                if i < 0.0001:
                    return 1.e30

            arrangements = callable(radius=RADIUS, amperes=AMPERES, var=v)

            Bz, Br = coil(Z, R, arrangements)

            frm = FIELD_REQUEST.magnitude
            fru = FIELD_REQUEST.units
            B = (Bz.to(fru) ** 2 + Br.to(fru) ** 2) ** 0.5
            B0 = B[ir0, iz0].to(fru)
            B = B.magnitude
            B0 = B0.magnitude

            dz = (z[1] - z[0]).magnitude
            dr = (r[1] - r[0]).magnitude
            dbz = np.gradient(B[ir0], dz)
            dbr = np.gradient(B[:, iz0], dr)

            # Cost function:
            #---------------
            # deviation from the goal (minimum gradients
            # and the closest of the field requested

            f = np.sum((dbz[nzi:nzf]) ** 4) \
                * np.sum((dbr[nri:nrf]) ** 4) \
                * np.sum((B[nri:nrf, nzi:nzf] - frm) ** 2) \
                * (B[ir0, iz0] - frm) ** 2

            if DEBUG:
                print("-" * 30 + "> ", v, f)

            return f


        varopt = var[:]

        if FIT:
            varopt = optimize.fmin(minimized_func,
                                   var,
                                   xtol=0.001,
                                   ftol=0.001,
                                   maxiter=1000,
                                   maxfun=5000)

        arrangements = callable(radius=RADIUS, amperes=AMPERES, var=varopt)

        Bz, Br = coil(Z, R, arrangements)
        B = (Bz ** 2 + Br ** 2) ** 0.5

        B0 = B[ir0, iz0]
        Bzs.append(B[ir0])
        Brs.append(B[:,iz0])

        print("\tCenter field : {:~.2fC} [{:~.1fC}]".format(np.max(B[ir0]),
                                                            np.max(B[ir0]).to(
                                                                "gauss")))

        mi = np.min(B[nri:nrf, nzi:nzf])
        ma = np.max(B[nri:nrf, nzi:nzf])
        hom = (ma - mi) * 100. / ma
        print("\tField homogeneity : {}%".format(hom.m))
        print("\tComputation region : ")
        print("\t\t\t\t x = [{:~.2fC}, {:~.2fC}]".format(z[nzi], z[nzf]))
        print("\t\t\t\t r = [{:~.2fC}, {:~.2fC}]".format(r[nri], r[nrf]))

        # =====================================================================
        #  plot
        # =====================================================================
        fig = plt.figure(item + 1, figsize=(5,6))

        ax = plt.subplot(212)
        step = int(3 * (n / 100.))

        Q = ax.quiver(Z[::step, ::step].m, R[::step, ::step].m, # xy coordinates
                Bz[::step, ::step].m, Br[::step, ::step].m, # arrow components
                      np.log(B[::step, ::step].m),
                      scale_units='xy', angles='xy', scale=1.,
                      pivot='middle', width=0.0015)

        # Coil draw

        narrangements = len(arrangements)
        nocenter = (narrangements == 1 )  # no center coil

        S = pi * (wire_diameter / 2.0) ** 2
        L = 0.0
        P = 0.0

        for id, bob in enumerate(arrangements):

            print()

            distance = bob[0]
            radius = bob[1]
            nturns = bob[2]
            amperes = bob[3]

            # dessin
            units = radius.units
            xa, ya, ra = distance.to(units).m, \
                         radius.m, wire_diameter.to(units).m * np.sqrt(
                nturns) / 2.0
            circ = Circle((xa, ya), radius=ra)
            ax.add_patch(circ)
            circ = Circle((xa, -ya), radius=ra)
            ax.add_patch(circ)

            # infos
            print("\t>>> Coil {} :".format(id + 1))
            print("\t\tDistance from center : {:~.2fC}:".format(distance))
            print("\t\tDiameter : {:~.1fC}".format(2 * radius))

            turn_length = 2. * pi * radius
            length = turn_length * nturns
            L = L + length

            print("\t\tNumber of turns : {}".format(nturns))
            print("\t\tDiameter of wire : {:~.2fC}".format(wire_diameter))
            print("\t\tLength of a turn : {:~.2fC}".format(turn_length))
            print(
            "\t\tTotal wire length in a coil : {:~.2fC}".format(length.to('m')))

            if nocenter or id > 0:
                print()
                circ = Circle((-xa, ya), radius=ra)
                ax.add_patch(circ)
                circ = Circle((-xa, -ya), radius=ra)
                ax.add_patch(circ)
                print("\t>>> Coil -{} :".format(id + 1))
                print("\t\t...same as coil {}".format(id + 1))
                L = L + length

        print()
        print("\t", "-" * 50)
        print("\t Electric features")
        print("\t", "-" * 50)

        rho = 1.7e-8 * ur.ohm * ur.m
        resistance = rho * L / S
        U = resistance * amperes
        W = U * amperes

        print("\tCopper resistivity : {:~.3gC}".format(rho))
        print("\tResistance : {:.3fC}".format(resistance.to('ohm')))
        print("\tIntensity I : {:.2fC}".format(amperes))
        print("\tElectric potential U = RI : {:.4fC}".format(U.to('volt')))
        print("\tPower : {:~.3fC}".format(W.to('W')))

        Brel = (np.ones_like(B) * np.max(B[ir0]) - B) * 100. / np.max(B[ir0])

        ax2 = plt.subplot(211, sharex=ax)
        ax3 = ax2.twinx()
        ax4 = ax2.twinx()

        ax2.plot(Z[0], B[ir0], 'b')
        dx = Z[0, 1] - Z[0, 0]
        dbx = np.gradient(B[ir0], dx)
        # unit lost in np.gradient!
        dbx = dbx * B.units / dx.units
        dbx = dbx.to('mT/m')
        ax3.plot(Z[0], dbx.m, 'b', linestyle='dashed')

        ax2.plot(R[:, 0], B[:, iz0], 'r')
        dr = R[1, 0] - R[0, 0]
        dbr = np.gradient(B[:, iz0], dr)
        dbr = dbr * B.units / dr.units
        dbr = dbr.to('mT/m')
        ax4.plot(R[:, 0], dbr.m, 'r', linestyle='dashed')

        ax.set_xlim(-2.5 * RADIUS.m, 2.5 * RADIUS.m)
        ax2.set_ylim(0., 7.)
        lim = 50.
        ax3.set_ylim(-lim, lim)
        ax4.set_ylim(-lim, lim)
        ax2.set_ylabel('Long. and trans. field/mT')
        ax3.set_ylabel('Field gradient/mT.m^-1')
        ax.set_ylabel('Radial position/cm')
        ax.set_xlabel('Axial position/cm')

        levels = np.array([0.90, 0.95, 0.98, 1., 1.02, 1.05, 1.1]) * B0.m

        cs = ax.contour(Z.m, R.m, B.m, levels=levels)
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_aspect(1.)

        plt.subplots_adjust(right = .85)
        fig.savefig("{}.png".format(name.lower()))

    # =========================================================================
    # Comparison plot
    # =========================================================================

    fig = plt.figure(50)
    axz = plt.subplot(211)

    for Bs in Bzs:
        axz.plot(z, Bs)

    axz.set_ylim(0., 7.)
    axz.set_xlim(-2 * RADIUS.m, 2 * RADIUS.m)
    axz.set_ylabel('Long. field/mT')
    axz.set_xlabel('Axial position/cm')

    axr = plt.subplot(212, sharex=axz, sharey=axz)
    for Br in Brs:
        axr.plot(z, Br)

    axr.set_ylim(0., 7.)
    axr.set_xlim(-2 * RADIUS.m, 2 * RADIUS.m)
    axr.set_ylabel('Radial. field/mT')
    axr.set_xlabel('Radial position/cm')

    fig.savefig("comparison.png")

    plt.show()
