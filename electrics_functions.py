from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.special import lambertw


def calcSingleElectrics(jsc, Rs, Rsh, j0, n, T):
    """
    Function to calculate and output Voc, FF, Pmax, JMPP, and VMPP for a single-diode model.

    Parameters:
    - jsc : Short-circuit current density (mA/cm^2)
    - Rs : Series resistance (Ohm)
    - Rsh : Shunt resistance (Ohm)
    - j0 : Diode reverse saturation current (A/cm^2)
    - n : Ideality factor
    - T : Temperature (Celsius)

    Returns:
    - Voc  : Open Circuit Voltage (V)
    - FF   : Fill Factor
    - Pmax : Maximum Power (mW/cm^2)
    - JMPP : Current Density at Maximum Power Point (mA/cm^2)
    - VMPP : Voltage at Maximum Power Point (V)
    """
    if jsc == 0:
        return 0, 0, 0, 0, 0

    q = 1.602E-19  # Charge (Coulombs)
    k = 1.3806488E-23  # Boltzmann constant (J/K)
    T += 273.15  # Converting °C to K

    # Thermal voltage at room temperature in V
    Vth = k * T / q

    # Directly use the diode equation for Voc
    Voc = n * Vth * np.log(jsc / j0 + 1)

    # Current density in mA/cm^2
    j = np.linspace(-5, 30, 100)

    # Define lambw helper variable
    lambw = (j0 / 1000 * Rsh) * np.exp(Rsh * (-j + jsc + j0) / (1000 * n * Vth)) / (n * Vth)

    # Calculate the Voltage
    V = -j / 1000 * (Rs + Rsh) + jsc / 1000 * Rsh - n * Vth * lambertw(lambw).real + j0 / 1000 * Rsh

    # Remove inf values from j and V
    valid_indices = np.isfinite(j) & np.isfinite(V)  # Boolean array for non-inf values
    j = j[valid_indices]
    V = V[valid_indices]


    # If Voc is non-physical, check if we need to estimate it
    if Voc < 0 or not np.isfinite(Voc):
        print(f"Voc is non-physical, calculated: {Voc:.4f} V")
        Voc = n * Vth * np.log(jsc / j0 + 1)
        print(f"Using asymptotic estimation for Voc: {Voc:.4f} V")

    # Calculate the Power and maximum power point P_max
    P = 10 * j * V
    if P.size == 0:
        return 0, 0, 0, 0, 0

    P_max = np.max(P)

    # Index of maximum power point
    I = np.argmax(P)

    # Calculate JMPP and VMPP
    JMPP = j[I]  # Current density at maximum power point
    VMPP = V[I]  # Voltage at maximum power point

    VMPP = max(0, VMPP)
    JMPP = max(0, JMPP)
    # Estimate the fill factor FF
    FF = max(0, (JMPP * VMPP) / (jsc * Voc))

    # Return values in order: Voc, FF, P_max, JMPP, VMPP
    return Voc, FF, P_max, JMPP, VMPP


'''
def calcSingleElectrics(jsc, Rs, Rsh, j0, n, T):

    q = 1.602E-19
    k = 1.3806488E-23
    T = T + 273.15  # Convert to Kelvin

    Vth = k * T / q
    Voc = n * Vth * np.log(jsc / j0 + 1)
    
    j = np.linspace(-5, 30, 100)

    lambw = (j0 / 1000 * Rsh) * np.exp(Rsh * (-j + jsc[:, None] + j0) / (1000 * n * Vth)) / (n * Vth)
    
    V = -j / 1000 * (Rs + Rsh) + jsc[:, None] / 1000 * Rsh - n * Vth * lambertw(lambw).real + j0 / 1000 * Rsh
    
    j = np.tile(j, (8760, 1))
    
    P = 10 * j * V
    
    P_max = np.max(P, axis=1)
    
    I = np.argmax(P, axis=1)
    
    # Get the indices of the max values in P for each hour
    I = np.argmax(P, axis=1)  # Shape (8760,)
    
    # Use the indices to get the JMPP and VMPP
    JMPP = j[np.arange(8760), I]  # Shape (8760,)

    VMPP = V[np.arange(8760), I]  # Shape (8760,)

    FF = np.maximum(0, (JMPP * VMPP) / (jsc * Voc))

    return Voc, FF, P_max, JMPP, VMPP
'''


def calctandem_with_single(electrics, jsc_RT):
    """
    Replace calctandemelectrics with a loop using calcSingleElectrics for multiple junctions (with shunt resistance).

    Parameters:
    - electrics: Dictionary containing electrical parameters for each junction (sub-cell).
    - jsc_RT: Short-circuit current density (mA/cm^2) at reference temperature for the tandem.

    Returns:
    - TandemVOC: Total Voc for the tandem cell.
    - TandemFF: Fill factor for the tandem.
    - TandemP_el: Maximum power for the tandem.
    - TandemJMPP: Current density at maximum power point for the tandem.
    - TandemVMPP: Voltage at maximum power point for the tandem.
    """

    k = len(electrics['Rs'])  # Number of junctions
    Vth = 0.02569  # Thermal voltage approximation at room temperature

    TandemVOC = 0  # Total Voc for tandem
    TandemJSC = np.inf  # Tandem current limited by the lowest jsc
    TandemVMPP = 0
    TandemP_el = 0

    JMPP_list = []
    VMPP_list = []
    FF_list = []

    for i in range(k):
        # Extract the parameters for the i-th junction
        jsc = jsc_RT * (1 + electrics['tcJsc'][i] * (electrics['Temp'][i] - 25))
        Rs = electrics['Rs'][i]
        Rsh = electrics['Rsh'][i]
        j0 = electrics['j0'][i]
        n = electrics['n'][i]
        T = electrics['Temp'][i]

        # Use calcSingleElectrics for the current sub-cell
        Voc, FF, P_max, JMPP, VMPP = calcSingleElectrics(jsc, Rs, Rsh, j0, n, T)

        # Sum the open-circuit voltages for the tandem Voc
        TandemVOC += Voc

        # Tandem current is limited by the smallest short-circuit current
        TandemJSC = min(TandemJSC, jsc)

        # Add the maximum power points (voltage and current)
        JMPP_list.append(JMPP)
        VMPP_list.append(VMPP)
        FF_list.append(FF)

    # Tandem JMPP is the minimum of the JMPP values
    TandemJMPP = min(JMPP_list)

    # Tandem VMPP is the sum of the VMPP values
    TandemVMPP = sum(VMPP_list)

    # Calculate the fill factor for the tandem
    TandemFF = (TandemJMPP * TandemVMPP) / (TandemJSC * TandemVOC)

    # Calculate the maximum power for the tandem
    TandemP_el = TandemJMPP * TandemVMPP * 10

    return TandemVOC, TandemFF, TandemP_el, TandemJMPP, TandemVMPP

