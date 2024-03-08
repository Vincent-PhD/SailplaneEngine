from Utilities.airfoil_generation_utilities import *
import numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_parameters,
    get_kulfan_coordinates,
)
from config import Plot_Airfoil, GanColumns
import os

# Define global variables

_LatentDimension = 10  # Latent dimension of the GAN airfoil generation technique


def generate_airfoil(
    NoiseVector: list,
    SelectedGenerationMethod: str,
    AirfoilDumpFilePath: str,
    AllowAirfoilPlotting: bool,
):
    if SelectedGenerationMethod == "GAN":
        X = np.asarray(NoiseVector)
        sample = pd.DataFrame(
            generator_scaler.inverse_transform(
                generator.predict(
                    tf.constant(X.reshape(1, _LatentDimension, 1)), verbose=0
                )
            ),
            columns=GanColumns,
        ).to_dict(orient="records")[0]

        sample = correct_sample(sample)

        seq_preds_generated = blstm.predict(
            blstm_scaler.transform(
                (
                    pd.DataFrame(
                        np.array([val for val in sample.values()]).reshape(1, 27),
                        columns=GanColumns,
                    )
                )
            ).reshape(1, 27, 1),
            verbose=0,
        )

        x = seq_preds_generated[0, 0:99]
        y = seq_preds_generated[0, 99:]

        np.savetxt(
            fname=AirfoilDumpFilePath,
            X=np.transpose([x, y]),
            delimiter=" ",
        )

        return x, y, sample
    elif SelectedGenerationMethod == "CST":
        coords = get_kulfan_coordinates(
            lower_weights=np.array(
                [
                    NoiseVector[0],
                    NoiseVector[1],
                    NoiseVector[2],
                    NoiseVector[3],
                    NoiseVector[4],
                    NoiseVector[5],
                    NoiseVector[6],
                    NoiseVector[7],
                ]
            ),
            upper_weights=np.array(
                [
                    NoiseVector[8],
                    NoiseVector[9],
                    NoiseVector[10],
                    NoiseVector[11],
                    NoiseVector[12],
                    NoiseVector[13],
                    NoiseVector[14],
                    NoiseVector[15],
                ]
            ),
            TE_thickness=0.0,
            leading_edge_weight=(NoiseVector[16]),
            n_points_per_side=120,
        )
        np.savetxt(AirfoilDumpFilePath, coords)

        sample = pd.DataFrame(
            get_airfoil_properties_for_cst(coords),
            columns=GanColumns,
        ).to_dict(orient="records")[0]
        return coords, sample
    else:
        RuntimeError
