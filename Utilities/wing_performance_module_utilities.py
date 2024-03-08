import numpy as np
import scipy
import math
import neuralfoil as nf
import matplotlib.pyplot as plt
from config import (
    Free_Stream_Speed_Base,
    Lift_Over_Drag_Base,
    Standard_Atmosphere_Tables,
    Fuselage_Performance_Data,
)


def l_over_d_objective_value(
    V_optimised,
    l_over_d_optimised,
    V_baseline,
    l_over_d_baseline,
):
    # Fit polynomial to baseline lift-to-drag data
    z_obj = np.polyfit(x=V_baseline, y=l_over_d_baseline, deg=3)

    p_obj = np.poly1d(z_obj)

    return l_over_d_optimised - p_obj(V_optimised)


# Function to read airfoil data at each station


def GetAirfoilData(Filename):
    with open(Filename, "r") as infile:
        AirfoilData = np.loadtxt(infile, dtype=str, skiprows=0, unpack=True)
    return AirfoilData


# Function to read wing geometry


def GetWingData(Filename):
    with open(Filename, "r") as infile:
        WingData = np.loadtxt(infile, dtype=float, skiprows=0, unpack=True)
    area = 0
    span = 0

    for i in range(0, len(WingData[0]), 1):
        span = span + WingData[2][i]
        area = area + ((WingData[0][i] + WingData[1][i]) / 2) * WingData[2][i]

    AR = 2 * (span**2) / area

    return area, span, AR, WingData


def Get_Chord(WingData, y_pos):
    chord = 0
    number_of_panels = WingData[0].size

    # determine wing span from wing file

    span = 0
    if number_of_panels > 1:
        for panel_length in WingData[2]:
            span = span + panel_length
    else:
        span = WingData[2]

    current_panel = 0
    current_span_pos = WingData[2][0]  # outer section of current panel

    while current_panel <= number_of_panels + 1:
        if y_pos < current_span_pos:
            chord = (
                (WingData[0][current_panel] - WingData[1][current_panel])
                / WingData[2][current_panel]
            ) * (current_span_pos - y_pos) + WingData[1][current_panel]
            break

        current_panel += 1
        current_span_pos = current_span_pos + WingData[2][current_panel]

    return chord


def Solve_LL(wingdata, span, AR, alpha, a, n):
    theta_range = np.linspace(90, 0.01, n)
    AMatrix = np.ones(shape=(n, n))
    X = np.ones(n)
    delta = 0
    Cl = np.zeros(n)
    y_pos = np.zeros(n)

    i = 0
    for theta in theta_range:
        unit_y = -math.cos(np.deg2rad(theta))
        c = Get_Chord(wingdata, -unit_y * span)
        mu = c * a / (8 * span)

        X[i] = mu * (np.deg2rad(alpha)) * (math.sin(np.deg2rad(theta)))
        for j in range(0, n, 1):
            AMatrix[i][j] = (math.sin(((j + 1) * 2 - 1) * np.deg2rad(theta))) * (
                ((j + 1) * 2 - 1) * mu + (math.sin(np.deg2rad(theta)))
            )

        y_pos[i] = -unit_y * span
        i += 1

    A = np.linalg.solve(AMatrix, X)

    for i in range(1, n, 1):
        delta = delta + ((i + 1) * 2 - 1) * (A[i] ** 2) / (A[0] ** 2)

    k = 0
    for theta in theta_range:
        for j in range(0, n, 1):
            unit_y = -math.cos(np.deg2rad(theta))
            c = Get_Chord(wingdata, -unit_y * span)
            Cl[k] = (
                Cl[k]
                + (8 * span)
                * (A[j] * (math.sin(((j + 1) * 2 - 1) * np.deg2rad(theta))))
                / c
            )

        k += 1

    CL = A[0] * (np.pi) * AR
    Cl_unit = Cl / CL
    return (Cl_unit, y_pos, CL, delta)


def Standard_Atmosphere(FileName, Elevation, standard_rho):
    with open(FileName, "r") as infile:
        S_atm_Data = np.loadtxt(infile, dtype=float, skiprows=1, unpack=True)

    pressure_interp = scipy.interpolate.interp1d(
        S_atm_Data[0], S_atm_Data[2], bounds_error=False, fill_value="extrapolate"
    )
    density_interp = scipy.interpolate.interp1d(
        S_atm_Data[0], S_atm_Data[3], bounds_error=False, fill_value="extrapolate"
    )
    Kin_viscosity_interp = scipy.interpolate.interp1d(
        S_atm_Data[0], S_atm_Data[4], bounds_error=False, fill_value="extrapolate"
    )

    pressure = pressure_interp(Elevation)
    rho = standard_rho * density_interp(Elevation)
    mu = rho * Kin_viscosity_interp(Elevation) * 1e-5

    return (pressure, rho, mu)


def Get_rho_mu(P_altitude, OAT, rho):
    # standard rho

    P, rho, mu = Standard_Atmosphere(Standard_Atmosphere_Tables, P_altitude, rho)

    R = 287.058
    T = OAT + 273.15

    P = P * 100000

    return ((P / (R * T)), mu)


def Wing_CD(
    WingFileName,
    AirfoilFileName,
    V,
    rho,
    mu,
    bank_angle,
    weight,
    Flap,
    alpha,
    a,
    winglet,
    plot=False,
):
    # get wing definition data
    Area, Span, AR, WingData = GetWingData(WingFileName)

    # calculte span efficiency and cl-distribution
    n = 60
    cl, y_pos, CL_theory, delta = Solve_LL(WingData, Span, AR, alpha, a, n)

    # set cl interpolation function
    cl_interp = scipy.interpolate.interp1d(
        y_pos, cl, bounds_error=False, fill_value="extrapolate"
    )

    CL = (2 * weight) / (rho * (V**2) * 2 * Area * np.cos(np.deg2rad(bank_angle)))

    # calculate profile drag of the wing

    AirfoilData = GetAirfoilData(AirfoilFileName)

    drag = 0
    area = 0
    span = 0.02

    Re_root = rho * V * float(AirfoilData[1][0]) / mu

    dl_aero_root = nf.get_aero_from_dat_file(
        filename=f"{AirfoilData[0][0]}",
        alpha=np.linspace(-20, 20, 100),
        Re=Re_root,
        model_size="xxxlarge",
    )

    difference_matrix_root = np.abs(dl_aero_root["CL"] - (cl_interp(span) * CL))

    min_difference_matrix_root = np.min(difference_matrix_root)

    index_root = np.where(difference_matrix_root == min_difference_matrix_root)

    Cd_root = dl_aero_root["CD"][index_root]

    Cd_root = Cd_root + (winglet / 100 * 0.0008) + 0.000009

    for i in range(0, len(WingData[0]), 1):
        Re_tip = rho * V * float(AirfoilData[1][i + 1]) / mu

        span = span + 0.995 * WingData[2][i]
        area = ((WingData[0][i] + WingData[1][i]) / 2) * WingData[2][i]

        # call XFOIL, NeuralFoil, WindTunnel

        dl_aero_tip = nf.get_aero_from_dat_file(
            filename=f"{AirfoilData[0][i+1]}",
            alpha=np.linspace(-20, 20, 100),
            Re=Re_tip,
            model_size="xxxlarge",
        )

        difference_matrix_tip = np.abs(dl_aero_tip["CL"] - (cl_interp(span) * CL))

        min_difference_matrix_tip = np.min(difference_matrix_tip)

        index_tip = np.where(difference_matrix_tip == min_difference_matrix_tip)

        Cd_tip = dl_aero_tip["CD"][index_tip]

        Cd_tip - (Cd_tip + (winglet / 100 * 0.0008) + 0.000009)

        drag = drag + 0.5 * rho * (V**2) * ((Cd_root + Cd_tip) / 2) * area

        Cd_root = Cd_tip

    CDi = ((CL**2) / (np.pi * AR)) * (
        1 + delta - delta * winglet / 100
    )  # include winglet contribution
    CD = 2 * drag / (rho * (V**2) * Area) + CDi

    return (CD, CL)


def Fuselage_Drag(FileName, V):
    with open(FileName, "r") as infile:
        Fuselage_Data = np.loadtxt(infile, dtype=float, skiprows=1, unpack=True)

    drag_interp = scipy.interpolate.interp1d(
        Fuselage_Data[0], Fuselage_Data[1], bounds_error=False, fill_value="extrapolate"
    )
    return drag_interp(V)
