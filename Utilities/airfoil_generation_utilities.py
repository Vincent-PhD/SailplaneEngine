import joblib
import tensorflow as tf
import pandas as pd
from config import (
    BLSTM_Model_Path,
    BLSTM_Scaler_Path,
    Generator_Modle_Path,
    Generator_Scaler_Path,
    GanColumns,
)
from circle_fit import (
    hyperSVD,
    plot_data_circle,
)  # hyperLSQ,hyperSVD,taubinSVD,kmh,lm,riemannSWFLa
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_parameters,
    get_kulfan_coordinates,
)

import numpy as np


# Define Global Variables
_NumPointsTopRad = 10


# GAN Models

blstm = tf.keras.models.load_model(BLSTM_Model_Path)

generator = tf.keras.models.load_model(Generator_Modle_Path)

blstm_scaler = joblib.load(BLSTM_Scaler_Path)

generator_scaler = joblib.load(Generator_Scaler_Path)

_LatentDimension = 10


def correct_sample(sample: dict) -> dict:
    if sample["MaxCamber"] < 0:
        sample["MaxCamber"] = 0

    if sample["MaxCamberLocation"] < 0:
        sample["MaxCamberLocation"] = 0

    if sample["MaxCamberRadius"] < 0:
        sample["MaxCamberRadius"] = 0

    if sample["MaxBottomThickness"] < 0:
        sample["MaxBottomThickness"] = 0

    if sample["MaxBottomThicknessLocation"] < 0:
        sample["MaxBottomThicknessLocation"] = 0

    if sample["MinBottomThicknessLocation"] < 0:
        sample["MinBottomThicknessLocation"] = 0

    if sample["MaxTopRadius"] < 0:
        sample["MaxTopRadius"] = 0

    if sample["MinBottomRadius"] < 0:
        sample["MinBottomRadius"] = 0

    if sample["BottomSurfaceTailAngle"] < 0:
        sample["BottomSurfaceTailAngle"] = 0

    if sample["LECamberAngle"] < 0:
        sample["LECamberAngle"] = 0

    if sample["TECamberAngle"] < 0:
        sample["TECamberAngle"] = 0

    if sample["TopLERadius"] < 0:
        sample["TopLERadius"] = 0

    if sample["BottomLERadius"] < 0:
        sample["BottomLERadius"] = 0

    if sample["TEGap"] < 0:
        sample["TEGap"] = 0

    if sample["MaxBottomThicknessLocation"] > 1:
        sample["MaxBottomThicknessLocation"] = 1

    if sample["MaxCamberLocation"] > 1:
        sample["MaxCamberLocation"] = 1

    if sample["MinTopThickness"] > 0:
        sample["MinTopThickness"] = 0

    if sample["MinTopThicknessLocation"] < 0:
        sample["MinTopThicknessLocation"] = 0

    if sample["MinTopThicknessLocation"] > 1:
        sample["MinTopThicknessLocation"] = 1

    if sample["y_centroid"] < 0:
        sample["y_centroid"] = 0

    if sample["x_centroid"] < 0:
        sample["x_centroid"] = 0

    if sample["MaxBottomRadius"] < 0:
        sample["MaxBottomRadius"] = 0

    return sample


def generate_sample(
    generator_model,
    X,
    column_names: list,
):
    sample = pd.DataFrame(
        generator_scaler.inverse_transform(
            generator.predict(tf.constant(X.reshape(1, _LatentDimension, 1)), verbose=0)
        ),
        columns=GanColumns,
    ).to_dict(orient="records")[0]

    sample = correct_sample(sample)

    return sample


def get_airfoil_properties_for_cst(coords):
    """# Split Upper and Lower Surfaces"""

    xPts2 = coords.T[0]
    yPts2 = coords.T[1]

    x_upper = xPts2[0 : int(len(xPts2) / 2 + 1)]
    y_upper = yPts2[0 : int(len(xPts2) / 2 + 1)]

    x_lower = xPts2[int(len(xPts2) / 2) :]
    y_lower = yPts2[int(len(xPts2) / 2) :]

    """# Determine Critical Curvature Points"""

    grad_top = []
    grad_bot = []
    top_grad_change = []
    bot_grad_change = []

    for i in range(1, len(x_upper)):
        grad_top.append(((y_upper[i - 1] - y_upper[i]) / (x_upper[i - 1] - x_upper[i])))
        grad_bot.append(((y_lower[i - 1] - y_lower[i]) / (x_lower[i - 1] - x_lower[i])))

    for i in range(1, len(grad_top)):
        if ((grad_top[i] < 0) & (grad_top[i - 1] > 0)) or (
            (grad_top[i] > 0) & (grad_top[i - 1] < 0)
        ):
            top_grad_change.append(i)
        if ((grad_bot[i] < 0) & (grad_bot[i - 1] > 0)) or (
            (grad_bot[i] > 0) & (grad_bot[i - 1] < 0)
        ):
            bot_grad_change.append(i)

    """# Determine Camber"""

    camber = [np.abs(y_lower[::-1][i] + y_upper[i]) / 2 for i in range(0, len(x_lower))]

    camber_max_x = x_upper[np.where(camber == np.max(camber))[0][0]]

    """# Determine Maximum Thickness"""

    thickness = [
        np.abs(y_lower[::-1][i]) + np.abs(y_upper[i]) for i in range(0, len(x_lower))
    ]

    thickness_max_x = x_upper[np.where(thickness == np.max(thickness))[0][0]]
    x_thickness_nearest_top_index = np.where(x_upper == thickness_max_x)[0][0]
    y_thickness_nearest_top = y_upper[x_thickness_nearest_top_index]

    thickness_min_x = x_lower[np.where(thickness[::-1] == np.max(thickness))[0][0]]
    x_thickness_nearest_bot_index = np.where(x_lower == thickness_min_x)[0][0]
    y_thickness_nearest_bot = y_lower[x_thickness_nearest_bot_index]

    x_plt_total_thickness = [thickness_max_x for i in range(0, 10)]
    y_plt_total_thickness = np.linspace(
        y_thickness_nearest_top, y_thickness_nearest_bot, 10
    )

    """# Determine the Maximum Tail Thicknerss @90% y/c"""

    loc_measure = 0.90
    x_90_nearest_top = x_upper[np.argmin([np.abs(x - loc_measure) for x in x_upper])]
    x_90_nearest_top_index = np.where(x_upper == x_90_nearest_top)[0][0]
    y_90_nearest_top = y_upper[x_90_nearest_top_index]

    x_90_nearest_bottom = x_lower[np.argmin([np.abs(x - loc_measure) for x in x_lower])]
    x_90_nearest_bottom_index = np.where(x_lower == x_90_nearest_bottom)[0][0]
    y_90_nearest_bottom = y_lower[x_90_nearest_bottom_index]

    tail_thickness_90_c = np.abs(y_90_nearest_bottom) + np.abs(y_90_nearest_top)

    x_plt_tail_thickness = [x_90_nearest_bottom for i in range(0, 10)]
    y_plt_tail_thickness = np.linspace(y_90_nearest_bottom, y_90_nearest_top, 10)

    """# Determine Max Bottom and Min Top Critical Points"""

    y_min_x_lower = x_lower[np.where(y_lower == np.min(y_lower))[0][0]]

    y_min_x_upper = x_upper[np.where(y_upper == np.min(y_upper))[0][0]]

    y_max_x_upper = x_upper[np.where(y_upper == np.max(y_upper))[0][0]]

    y_max_x_lower = x_lower[np.where(y_lower == np.max(y_lower))[0][0]]

    """# Determine the Top Surface Tail Angle"""

    loc_measure = 0.925
    x_90_nearest_top = x_upper[np.argmin([np.abs(x - loc_measure) for x in x_upper])]
    x_90_nearest_top_index = np.where(x_upper == x_90_nearest_top)[0][0]
    y_90_nearest_top = y_upper[x_90_nearest_top_index]

    x_ref = 1  # np.min(x_lower)
    y_ref = 0  # np.max(y_lower)

    o_dist = np.sqrt(
        (y_90_nearest_top - y_ref) ** 2 + (x_90_nearest_top - x_90_nearest_top) ** 2
    )

    a_dist = np.sqrt(
        (y_90_nearest_top - y_90_nearest_top) ** 2 + (x_90_nearest_top - x_ref) ** 2
    )

    theta_tail_top = np.arctan(o_dist / a_dist) * (
        180 / np.pi
    )  # np.arctan((y_90_nearest_top-y_ref)/(x_90_nearest_top-x_ref))*(180/np.pi)

    x_plt_top_theta = [i for i in np.linspace(loc_measure, 1, 10)]

    y_plt_top_theta = [
        (
            (y_90_nearest_top - y_ref) / (x_90_nearest_top - x_ref) * (x_v - x_ref)
            + y_ref
        )
        for x_v in x_plt_top_theta
    ]

    """# Determine the Bottom Surface Tail Angle"""

    x_90_nearest_bottom = x_lower[np.argmin([np.abs(x - loc_measure) for x in x_lower])]
    x_90_nearest_bottom_index = np.where(x_lower == x_90_nearest_bottom)[0][0]
    y_90_nearest_bottom = y_lower[x_90_nearest_bottom_index]

    x_ref = 1  # np.min(x_lower)
    y_ref = 0  # np.max(y_lower)

    o_dist = np.sqrt(
        (y_90_nearest_bottom - y_ref) ** 2
        + (x_90_nearest_bottom - x_90_nearest_bottom) ** 2
    )

    a_dist = np.sqrt(
        (y_90_nearest_bottom - y_90_nearest_bottom) ** 2
        + (x_90_nearest_bottom - x_ref) ** 2
    )

    theta_tail_bottom = np.arctan(o_dist / a_dist) * (
        180 / np.pi
    )  # np.arctan((y_90_nearest_bottom-y_ref)/(x_90_nearest_bottom-x_ref))*(180/np.pi)

    x_plt_bottom_theta = [i for i in np.linspace(loc_measure, 1, 10)]

    y_plt_bottom_theta = [
        (
            (y_90_nearest_bottom - y_ref)
            / (x_90_nearest_bottom - x_ref)
            * (x_v - x_ref)
            + y_ref
        )
        for x_v in x_plt_bottom_theta
    ]

    """# Determine LE Camber Angle"""

    loc_measure = 0.04
    x_90_nearest_top_camber = x_lower[
        np.argmin([np.abs(x - loc_measure) for x in x_lower])
    ]
    x_90_nearest_top_camber_index = np.where(x_lower == x_90_nearest_top_camber)[0][0]
    y_90_nearest_top_camber = camber[::-1][x_90_nearest_top_camber_index]

    x_ref = 0  # np.min(x_lower)
    y_ref = 0  # np.max(y_lower)

    o_dist = np.sqrt(
        (y_90_nearest_top_camber - y_ref) ** 2
        + (x_90_nearest_top_camber - x_90_nearest_top_camber) ** 2
    )

    a_dist = np.sqrt(
        (y_90_nearest_top_camber - y_90_nearest_top_camber) ** 2
        + (x_90_nearest_top_camber - x_ref) ** 2
    )

    theta_tail_top_camber = np.arctan(o_dist / a_dist) * (
        180 / np.pi
    )  # np.arctan((y_90_nearest_top_camber-y_ref)/(x_90_nearest_top_camber-x_ref))*(180/np.pi)

    x_plt_top_camber_theta = [i for i in np.linspace(0, loc_measure + 0.5, 10)]

    y_plt_top_camber_theta = [
        (
            (y_90_nearest_top_camber - y_ref)
            / (x_90_nearest_top_camber - x_ref)
            * (x_v - x_ref)
            + y_ref
        )
        for x_v in x_plt_top_camber_theta
    ]

    """# Determine TE Camber Angle"""

    x_90_nearest_bottom_camber = x_upper[
        np.argmin([np.abs(x - (1 - loc_measure)) for x in x_upper])
    ]
    x_90_nearest_bottom_camber_index = np.where(x_upper == x_90_nearest_bottom_camber)[
        0
    ][0]
    y_90_nearest_bottom_camber = camber[x_90_nearest_bottom_camber_index]

    x_ref = 1  # np.min(x_upper)
    y_ref = 0  # np.max(y_lower)

    o_dist = np.sqrt(
        (y_90_nearest_bottom_camber - y_ref) ** 2
        + (x_90_nearest_bottom_camber - x_90_nearest_bottom_camber) ** 2
    )

    a_dist = np.sqrt(
        (y_90_nearest_bottom_camber - y_90_nearest_bottom_camber) ** 2
        + (x_90_nearest_bottom_camber - x_ref) ** 2
    )

    theta_tail_bottom_camber = np.arctan(o_dist / a_dist) * (180 / np.pi)

    x_plt_bottom_camber_theta = [i for i in np.linspace(1, (1 - loc_measure) - 0.9, 10)]

    y_plt_bottom_camber_theta = [
        (
            (y_90_nearest_bottom_camber - y_ref)
            / (x_90_nearest_bottom_camber - x_ref)
            * (x_v - x_ref)
            + y_ref
        )
        for x_v in x_plt_bottom_camber_theta
    ]

    """# Determine Top Surface LE Radius"""

    r_xc_le_top, r_yc_le_top, r_le_top, r_s_le_top = hyperSVD(
        [(i, j) for i, j in zip(x_upper[::-1][0:4], y_upper[::-1][0:4])]
    )
    theta_fit = np.linspace(-np.pi, np.pi, 180)
    x_fit_le_top = r_xc_le_top + r_le_top * np.cos(theta_fit)
    y_fit_le_top = r_yc_le_top + r_le_top * np.sin(theta_fit)

    """# Determine Bottom Surface LE Radius"""

    r_xc_le_bottom, r_yc_le_bottom, r_le_bottom, r_s_le_bottom = hyperSVD(
        [(i, j) for i, j in zip(x_lower[0:4], y_lower[0:4])]
    )
    x_fit_le_bottom = r_xc_le_bottom + r_le_bottom * np.cos(theta_fit)
    y_fit_le_bottom = r_yc_le_bottom + r_le_bottom * np.sin(theta_fit)

    """# Determine Top Surface Max Radius"""

    r_xc_max_top, r_yc_maxc_top, r_max_top, r_s_max_top = hyperSVD(
        [
            (i, j)
            for i, j in zip(
                x_upper[
                    np.where(y_upper == np.max(y_upper))[0][0]
                    - _NumPointsTopRad : np.where(y_upper == np.max(y_upper))[0][0]
                    + _NumPointsTopRad
                ],
                y_upper[
                    np.where(y_upper == np.max(y_upper))[0][0]
                    - _NumPointsTopRad : np.where(y_upper == np.max(y_upper))[0][0]
                    + _NumPointsTopRad
                ],
            )
        ]
    )
    x_fit_max_top = r_xc_max_top + r_max_top * np.cos(theta_fit)
    y_fit_max_top = r_yc_maxc_top + r_max_top * np.sin(theta_fit)

    """# Determine Bottom Surface Min Radius"""

    r_xc_min_bottom, r_yc_min_bottom, r_min_bottom, r_s_min_bottom = hyperSVD(
        [
            (i, j)
            for i, j in zip(
                x_lower[
                    np.where(y_lower == np.min(y_lower))[0][0]
                    - _NumPointsTopRad : np.where(y_lower == np.min(y_lower))[0][0]
                    + _NumPointsTopRad
                ],
                y_lower[
                    np.where(y_lower == np.min(y_lower))[0][0]
                    - _NumPointsTopRad : np.where(y_lower == np.min(y_lower))[0][0]
                    + _NumPointsTopRad
                ],
            )
        ]
    )
    x_fit_min_bottom = r_xc_min_bottom + r_min_bottom * np.cos(theta_fit)
    y_fit_min_bottom = r_yc_min_bottom + r_min_bottom * np.sin(theta_fit)

    """# Determine Bottom Surface Max Radius"""

    if np.max(y_lower) == 0:
        x_fit_max_bottom = 0  # r_xc_max_bottom + r_max_bottom * np.cos(theta_fit)
        y_fit_max_bottom = 0  # r_yc_maxc_bottom + r_max_bottom * np.sin(theta_fit)
        r_xc_max_bottom, r_yc_maxc_bottom, r_max_bottom, r_s_max_bottom = [
            0,
            0,
            0,
            0,
        ]

    else:
        (
            r_xc_max_bottom,
            r_yc_maxc_bottom,
            r_max_bottom,
            r_s_max_bottom,
        ) = hyperSVD(
            [
                (i, j)
                for i, j in zip(
                    x_lower[
                        np.where(y_lower == np.max(y_lower))[0][0]
                        - _NumPointsTopRad : np.where(y_lower == np.max(y_lower))[0][0]
                        + _NumPointsTopRad
                    ],
                    y_lower[
                        np.where(y_lower == np.max(y_lower))[0][0]
                        - _NumPointsTopRad : np.where(y_lower == np.max(y_lower))[0][0]
                        + _NumPointsTopRad
                    ],
                )
            ]
        )
        x_fit_max_bottom = r_xc_max_bottom + r_max_bottom * np.cos(theta_fit)
        y_fit_max_bottom = r_yc_maxc_bottom + r_max_bottom * np.sin(theta_fit)

    """# Determine Radius at Maximum Camber"""

    r_xc_camber, r_yc_camber, r_camber, r_s_camber = hyperSVD(
        [
            (i, j)
            for i, j in zip(
                x_upper[
                    np.where(camber == np.max(camber))[0][0]
                    - _NumPointsTopRad : np.where(camber == np.max(camber))[0][0]
                    + _NumPointsTopRad
                ],
                camber[
                    np.where(camber == np.max(camber))[0][0]
                    - _NumPointsTopRad : np.where(camber == np.max(camber))[0][0]
                    + _NumPointsTopRad
                ],
            )
        ]
    )
    x_fit_camber = r_xc_camber + r_camber * np.cos(theta_fit)
    y_fit_camber = r_yc_camber + r_camber * np.sin(theta_fit)

    """# Determine TE Gap"""

    te_gap = np.abs(y_upper[-1]) + np.abs(y_lower[-1])

    """# Determine Airfoil Area"""

    def get_structural_properties(X_upper, X_lower, Y_upper, Y_lower, X, Y):
        """Calculates structural properties of a given airfoil.

        ARGS:

        X_upper: X top coordinates of the airfoil
        Y_upper: Y top coordinates of the airfoil
        X_lower: X bottom coordinates of the airfoil
        Y_lower: Y bottom coordinates of the airfoil
        X: Airfoil X coordinates
        Y: Airfoil Y coordinates

        RETURNS:

            Airfoil area, y center location, x center location and bending inertia about the y-y axis of the airfoil.

        """
        parameterise_f1 = np.polynomial.polynomial.Polynomial.fit(
            x=X_upper, y=Y_upper, deg=50, full=False
        )

        parameterise_f2 = np.polynomial.polynomial.Polynomial.fit(
            x=X_lower, y=Y_lower, deg=50, full=False
        )
        parameterise_foil = np.polynomial.polynomial.Polynomial.fit(
            x=X.reshape(1, -1)[0], y=Y.reshape(1, -1)[0], deg=50, full=False
        )

        x = np.arange(0, 1, 1 / 5_000)

        yt = parameterise_f1(x)
        yb = parameterise_f2(x)
        y = parameterise_foil(x)

        # plt.plot(X_upper, y_upper,".")
        # plt.plot(x_lower, y_lower,".")
        # plt.plot(xPts2,yPts2)
        # plt.plot(x,yt,".")
        # plt.plot(x,yb,".")

        cx = []
        cy = []
        A = []
        Ix = []
        Iy = []

        for i in range(0, len(x) - 1):
            A_i = np.array((x[i + 1] - x[i]) * (yt[i] - yb[i]))
            A.append(A_i)

        total_area = np.sum(A)

        for j in range(0, len(x) - 1):
            Cx_j = (A[j] * x[j]) / (total_area)
            Cy_j = (A[j] * y[j]) / (total_area)

            cx.append(Cx_j)
            cy.append(Cy_j)

        y_centroid = np.sum(cy)
        x_centroid = np.sum(cx)

        for k in range(0, len(x) - 1):
            Iy_i = (A[k]) * (y_centroid - y[k]) ** 2
            Ix_i = (A[k]) * (x_centroid - x[k]) ** 2

            Iy.append(Iy_i)
            Ix.append(Ix_i)

        Iyy = np.sum(Iy)
        bending_inertia_x = np.sum(Ix)

        return (total_area, y_centroid, x_centroid, bending_inertia_x)

    (
        total_area,
        y_centroid,
        x_centroid,
        bending_inertia_x,
    ) = get_structural_properties(
        X_upper=x_upper,
        X_lower=x_lower,
        Y_upper=y_upper,
        Y_lower=y_lower,
        X=xPts2,
        Y=yPts2,
    )

    """# Log Parameterisation Data"""

    # Log data in Avro Format

    if (np.count(top_grad_change) < 4) and (np.count(bot_grad_change) < 4):
        dump_dict = {
            "MaxCamber": np.max(camber),
            "MaxCamberLocation": camber_max_x,
            "MaxCamberRadius": r_camber,
            "MaxThickness": np.max(thickness),
            "MaxThicknessLocation": thickness_max_x,
            "MaxThicknessTail90c": tail_thickness_90_c,
            "MinBottomThickness": np.min(y_lower),
            "MinBottomThicknessLocation": y_min_x_lower,
            "MinBottomRadius": r_min_bottom,
            "MinTopThickness": np.min(y_upper),
            "MinTopThicknessLocation": y_min_x_upper,
            "MaxTopThickness": np.max(y_upper),
            "MaxTopThicknessLocation": y_max_x_upper,
            "MaxTopRadius": r_max_top,
            "MaxBottomThickness": np.max(y_lower),
            "MaxBottomThicknessLocation": y_max_x_lower,
            "TopSurfaceTailAngle": theta_tail_top,
            "BottomSurfaceTailAngle": theta_tail_bottom,
            "LECamberAngle": theta_tail_top_camber,
            "TECamberAngle": theta_tail_bottom_camber,
            "TopLERadius": r_le_top,
            "BottomLERadius": r_le_bottom,
            "MaxBottomRadius": r_max_bottom,
            "TEGap": te_gap,
            "total_area": total_area,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
        }

    else:
        dump_dict = {
            "MaxCamber": 00000000000000000000000000000000,
            "MaxCamberLocation": 00000000000000000000000000000000,
            "MaxCamberRadius": 00000000000000000000000000000000,
            "MaxThickness": 00000000000000000000000000000000,
            "MaxThicknessLocation": 00000000000000000000000000000000,
            "MaxThicknessTail90c": 00000000000000000000000000000000,
            "MinBottomThickness": 00000000000000000000000000000000,
            "MinBottomThicknessLocation": 00000000000000000000000000000000,
            "MinBottomRadius": 00000000000000000000000000000000,
            "MinTopThickness": 00000000000000000000000000000000,
            "MinTopThicknessLocation": 00000000000000000000000000000000,
            "MaxTopThickness": 00000000000000000000000000000000,
            "MaxTopThicknessLocation": 00000000000000000000000000000000,
            "MaxTopRadius": 00000000000000000000000000000000,
            "MaxBottomThickness": 00000000000000000000000000000000,
            "MaxBottomThicknessLocation": 00000000000000000000000000000000,
            "TopSurfaceTailAngle": 00000000000000000000000000000000,
            "BottomSurfaceTailAngle": 00000000000000000000000000000000,
            "LECamberAngle": 00000000000000000000000000000000,
            "TECamberAngle": 00000000000000000000000000000000,
            "TopLERadius": 00000000000000000000000000000000,
            "BottomLERadius": 00000000000000000000000000000000,
            "MaxBottomRadius": 00000000000000000000000000000000,
            "TEGap": 00000000000000000000000000000000,
            "total_area": 00000000000000000000000000000000,
            "x_centroid": 00000000000000000000000000000000,
            "y_centroid": 00000000000000000000000000000000,
        }

    return dump_dict
