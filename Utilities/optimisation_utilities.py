import numpy as np
import pandas as pd
import polars
import joblib
from Modules.airfoil_generation_module import generate_airfoil
import tensorflow as tf
from config import (
    Generator_Scaler_Path,
    Generator_Modle_Path,
    Airfoil_Generation_Method,
    Station_1_Airfoil_Coordinate_File,
    Station_2_Airfoil_Coordinate_File,
    Station_3_Airfoil_Coordinate_File,
    Station_4_Airfoil_Coordinate_File,
    Station_5_Airfoil_Coordinate_File,
    Station_6_Airfoil_Coordinate_File,
    Plot_Airfoil,
    GanColumns,
    Station_1_Airfoil_Constraints,
    Station_2_Airfoil_Constraints,
    Station_3_Airfoil_Constraints,
    Station_4_Airfoil_Constraints,
    Station_5_Airfoil_Constraints,
    Station_6_Airfoil_Constraints,
    Prelim_Generated_Candidate_Airfoil_Solutions,
    N_Candidate_Solutions,
    Station_1_Chord_Bounds,
    Station_2_Chord_Bounds,
    Station_3_Chord_Bounds,
    Station_4_Chord_Bounds,
    Station_5_Chord_Bounds,
    Station_6_Chord_Bounds,
    Station_1_Wingspan_Increments_Bounds,
    Station_2_Wingspan_Increments_Bounds,
    Station_3_Wingspan_Increments_Bounds,
    Station_4_Wingspan_Increments_Bounds,
    Station_5_Wingspan_Increments_Bounds,
)

_LatentDimension = 10


def rectify_planform_sampling_error(chords, increments):
    if (
        chords[0][0] < Station_1_Chord_Bounds[0]
        or chords[0][0] > Station_1_Chord_Bounds[1]
    ):
        chords[0][0] = Station_1_Chord_Bounds[1]

    if (
        chords[0][1] < Station_2_Chord_Bounds[0]
        or chords[0][1] > Station_2_Chord_Bounds[1]
    ):
        chords[0][1] = Station_2_Chord_Bounds[1]

    if (
        chords[0][2] < Station_3_Chord_Bounds[0]
        or chords[0][2] > Station_3_Chord_Bounds[1]
    ):
        chords[0][2] = Station_3_Chord_Bounds[1]

    if (
        chords[0][3] < Station_4_Chord_Bounds[0]
        or chords[0][3] > Station_4_Chord_Bounds[1]
    ):
        chords[0][3] = Station_4_Chord_Bounds[1]

    if (
        chords[0][4] < Station_5_Chord_Bounds[0]
        or chords[0][4] > Station_5_Chord_Bounds[1]
    ):
        chords[0][4] = Station_5_Chord_Bounds[1]

    if (
        chords[0][5] < Station_6_Chord_Bounds[0]
        or chords[0][5] > Station_6_Chord_Bounds[1]
    ):
        chords[0][5] = Station_6_Chord_Bounds[1]

    if (
        increments[0][0] < Station_1_Wingspan_Increments_Bounds[0]
        or increments[0][0] > Station_1_Wingspan_Increments_Bounds[1]
    ):
        increments[0][0] = Station_1_Wingspan_Increments_Bounds[1]

    if (
        increments[0][1] < Station_2_Wingspan_Increments_Bounds[0]
        or increments[0][1] > Station_2_Wingspan_Increments_Bounds[1]
    ):
        increments[0][1] = Station_2_Wingspan_Increments_Bounds[1]

    if (
        increments[0][2] < Station_3_Wingspan_Increments_Bounds[0]
        or increments[0][2] > Station_3_Wingspan_Increments_Bounds[1]
    ):
        increments[0][2] = Station_3_Wingspan_Increments_Bounds[1]

    if (
        increments[0][3] < Station_4_Wingspan_Increments_Bounds[0]
        or increments[0][3] > Station_4_Wingspan_Increments_Bounds[1]
    ):
        increments[0][3] = Station_4_Wingspan_Increments_Bounds[1]

    if (
        increments[0][4] < Station_5_Wingspan_Increments_Bounds[0]
        or increments[0][4] > Station_5_Wingspan_Increments_Bounds[1]
    ):
        increments[0][4] = Station_5_Wingspan_Increments_Bounds[1]

    return chords, increments


def partition_noise_vector(NoiseVector: list, SelectedGenerationMethod: str):
    if SelectedGenerationMethod == "GAN":
        # Split the Noise Vector into:
        #       1) Airfoil Generation Partition,
        #       2) Planform Generation Partition (Wingspan increments),
        #       3) Planform Generation Partition (Chords increments).

        # (1)
        airfoil_noise_1 = [NoiseVector[0:10]]
        airfoil_noise_2 = [NoiseVector[10 : 10 * 2]]
        airfoil_noise_3 = [NoiseVector[10 * 2 : 10 * 3]]
        airfoil_noise_4 = [NoiseVector[10 * 3 : 10 * 4]]
        airfoil_noise_5 = [NoiseVector[10 * 4 : 10 * 5]]
        airfoil_noise_6 = [NoiseVector[10 * 5 : 10 * 6]]
        # (2)
        chords = [NoiseVector[10 * 6 : (10 * 6) + 6]]
        # (3)
        increments = [NoiseVector[(10 * 6) + 6 :]]
    elif SelectedGenerationMethod == "CST":
        # (1)
        airfoil_noise_1 = [NoiseVector[0:17]]
        airfoil_noise_2 = [NoiseVector[17 : 17 * 2]]
        airfoil_noise_3 = [NoiseVector[17 * 2 : 17 * 3]]
        airfoil_noise_4 = [NoiseVector[17 * 3 : 17 * 4]]
        airfoil_noise_5 = [NoiseVector[17 * 4 : 17 * 5]]
        airfoil_noise_6 = [NoiseVector[17 * 5 : 17 * 6]]
        # (2)
        chords = [NoiseVector[(17 * 6) : (17 * 6) + 6]]
        # (3)
        increments = [NoiseVector[(17 * 6) + 6 :]]

    else:
        RuntimeError

    return (
        airfoil_noise_1,
        airfoil_noise_2,
        airfoil_noise_3,
        airfoil_noise_4,
        airfoil_noise_5,
        airfoil_noise_6,
        chords,
        increments,
    )


def get_airfoil_penalty(
    parsimonious_vars: dict, feature_list: list, constraint_dictionary: dict
):
    # Calculate Penalties

    error = 0
    count = 0
    for feature in feature_list:
        if parsimonious_vars[feature] < constraint_dictionary[feature]["Lower"]:
            count += 1
            error += np.sqrt(
                np.abs(
                    parsimonious_vars[feature] - constraint_dictionary[feature]["Lower"]
                )
            )
        if parsimonious_vars[feature] > constraint_dictionary[feature]["Upper"]:
            count += 1
            error += np.sqrt(
                np.abs(
                    parsimonious_vars[feature] - constraint_dictionary[feature]["Upper"]
                )
            )

    return error, count


def planform_penalty(x, weight=50):
    if x[0][0] > x[0][1] > x[0][2] > x[0][3] > x[0][4] > x[0][5]:
        _PlanformPenalty = 0
    elif x[0][0] == x[0][1] == x[0][2] == x[0][3] == x[4] == x[5]:
        _PlanformPenalty = 0
    else:
        _PlanformPenalty = (
            np.sum(
                np.abs(
                    [
                        err
                        for err in [x[0][i + 1] - x[0][i] for i in range(0, 5)]
                        if err > 0
                    ]
                )
            )
            * weight
        )
    return _PlanformPenalty


def airfoil_thickness_distribution_penalty(x: np.array, weight):
    if x[0] > x[1] > x[2] > x[3] > x[4] > x[5]:
        _AirfoilThicknessDistributionPenalty = 0
    elif x[0] == x[1] == x[2] == x[3] == x[4] == x[5]:
        _AirfoilThicknessDistributionPenalty = 0
    else:
        _AirfoilThicknessDistributionPenalty = (
            np.sum(
                np.abs(
                    [err for err in [x[i + 1] - x[i] for i in range(0, 5)] if err > 0]
                )
            )
            * weight
        )
    return _AirfoilThicknessDistributionPenalty


def determine_valid_constraints(ConstraintDictionary: dict, Sample: pd.DataFrame):
    TruthChecker = []
    for key in ConstraintDictionary:
        if (Sample[key] >= ConstraintDictionary[key]["Lower"]) and (
            Sample[key] <= ConstraintDictionary[key]["Upper"]
        ):
            TruthChecker.append(True)
        else:
            TruthChecker.append(False)
    return False if False in TruthChecker else True


def generate_constrained_population(TotalSamples: int):

    TotalSamples = TotalSamples * 100

    generator_scaler = joblib.load(Generator_Scaler_Path)

    generator = tf.keras.models.load_model(Generator_Modle_Path)

    # Station 1

    Station_1_Noise_Vector = np.random.randn(TotalSamples * _LatentDimension).reshape(
        TotalSamples, _LatentDimension, 1
    )
    Station_1_Parsimonious_Prediction = generator_scaler.inverse_transform(
        generator.predict(tf.constant(Station_1_Noise_Vector), verbose=0)
    )
    Station_1_Data_Frame = polars.from_numpy(
        Station_1_Parsimonious_Prediction,
        schema=["ST1" + "_" + GanColumns[i] for i in range(len(GanColumns))],
    ).with_row_count("ST1_row_nr")

    # Station 2

    Station_2_Noise_Vector = np.random.randn(TotalSamples * _LatentDimension).reshape(
        TotalSamples, _LatentDimension, 1
    )
    Station_2_Parsimonious_Prediction = generator_scaler.inverse_transform(
        generator.predict(tf.constant(Station_2_Noise_Vector), verbose=0)
    )
    Station_2_Data_Frame = polars.from_numpy(
        Station_2_Parsimonious_Prediction,
        schema=["ST2" + "_" + GanColumns[i] for i in range(len(GanColumns))],
    ).with_row_count("ST2_row_nr")

    # Station 3

    Station_3_Noise_Vector = np.random.randn(TotalSamples * _LatentDimension).reshape(
        TotalSamples, _LatentDimension, 1
    )
    Station_3_Parsimonious_Prediction = generator_scaler.inverse_transform(
        generator.predict(tf.constant(Station_3_Noise_Vector), verbose=0)
    )
    Station_3_Data_Frame = polars.from_numpy(
        Station_3_Parsimonious_Prediction,
        schema=["ST3" + "_" + GanColumns[i] for i in range(len(GanColumns))],
    ).with_row_count("ST3_row_nr")

    # Station 4

    Station_4_Noise_Vector = np.random.randn(TotalSamples * _LatentDimension).reshape(
        TotalSamples, _LatentDimension, 1
    )
    Station_4_Parsimonious_Prediction = generator_scaler.inverse_transform(
        generator.predict(tf.constant(Station_4_Noise_Vector), verbose=0)
    )
    Station_4_Data_Frame = polars.from_numpy(
        Station_4_Parsimonious_Prediction,
        schema=["ST4" + "_" + GanColumns[i] for i in range(len(GanColumns))],
    ).with_row_count("ST4_row_nr")

    # Station 5

    Station_5_Noise_Vector = np.random.randn(TotalSamples * _LatentDimension).reshape(
        TotalSamples, _LatentDimension, 1
    )
    Station_5_Parsimonious_Prediction = generator_scaler.inverse_transform(
        generator.predict(tf.constant(Station_5_Noise_Vector), verbose=0)
    )
    Station_5_Data_Frame = polars.from_numpy(
        Station_5_Parsimonious_Prediction,
        schema=["ST5" + "_" + GanColumns[i] for i in range(len(GanColumns))],
    ).with_row_count("ST5_row_nr")

    # Station 6

    Station_6_Noise_Vector = np.random.randn(TotalSamples * _LatentDimension).reshape(
        TotalSamples, _LatentDimension, 1
    )
    Station_6_Parsimonious_Prediction = generator_scaler.inverse_transform(
        generator.predict(tf.constant(Station_6_Noise_Vector), verbose=0)
    )
    Station_6_Data_Frame = polars.from_numpy(
        Station_6_Parsimonious_Prediction,
        schema=["ST6" + "_" + GanColumns[i] for i in range(len(GanColumns))],
    ).with_row_count("ST6_row_nr")

    # Filter Station 1
    df_filt_st1 = Station_1_Data_Frame
    for key in Station_1_Airfoil_Constraints:
        FilterColumn = "ST1_" + key
        df_filt_st1 = df_filt_st1.filter(
            (polars.col(FilterColumn) >= Station_1_Airfoil_Constraints[key]["Lower"])
            & (polars.col(FilterColumn) <= Station_1_Airfoil_Constraints[key]["Upper"])
        )

    # Filter Station 2
    df_filt_st2 = Station_2_Data_Frame
    for key in Station_2_Airfoil_Constraints:
        FilterColumn = "ST2_" + key
        df_filt_st2 = df_filt_st2.filter(
            (polars.col(FilterColumn) >= Station_2_Airfoil_Constraints[key]["Lower"])
            & (polars.col(FilterColumn) <= Station_2_Airfoil_Constraints[key]["Upper"])
        )

    # Filter Station 3
    df_filt_st3 = Station_3_Data_Frame
    for key in Station_3_Airfoil_Constraints:
        FilterColumn = "ST3_" + key
        df_filt_st3 = df_filt_st3.filter(
            (polars.col(FilterColumn) >= Station_3_Airfoil_Constraints[key]["Lower"])
            & (polars.col(FilterColumn) <= Station_3_Airfoil_Constraints[key]["Upper"])
        )

    # Filter Station 4
    df_filt_st4 = Station_4_Data_Frame
    for key in Station_4_Airfoil_Constraints:
        FilterColumn = "ST4_" + key
        df_filt_st4 = df_filt_st4.filter(
            (polars.col(FilterColumn) >= Station_4_Airfoil_Constraints[key]["Lower"])
            & (polars.col(FilterColumn) <= Station_4_Airfoil_Constraints[key]["Upper"])
        )

    # Filter Station 5
    df_filt_st5 = Station_5_Data_Frame
    for key in Station_5_Airfoil_Constraints:
        FilterColumn = "ST5_" + key
        df_filt_st5 = df_filt_st5.filter(
            (polars.col(FilterColumn) >= Station_5_Airfoil_Constraints[key]["Lower"])
            & (polars.col(FilterColumn) <= Station_5_Airfoil_Constraints[key]["Upper"])
        )

    # Filter Station 6
    df_filt_st6 = Station_6_Data_Frame
    for key in Station_6_Airfoil_Constraints:
        FilterColumn = "ST6_" + key
        df_filt_st6 = df_filt_st6.filter(
            (polars.col(FilterColumn) >= Station_6_Airfoil_Constraints[key]["Lower"])
            & (polars.col(FilterColumn) <= Station_6_Airfoil_Constraints[key]["Upper"])
        )

    # Sort data frames

    df_filt_st1 = df_filt_st1.sort("ST1_MaxThickness", descending=True)

    df_filt_st2 = df_filt_st2.sort("ST2_MaxThickness", descending=True)

    df_filt_st3 = df_filt_st3.sort("ST3_MaxThickness", descending=True)

    df_filt_st4 = df_filt_st4.sort("ST4_MaxThickness", descending=True)

    df_filt_st5 = df_filt_st5.sort("ST5_MaxThickness", descending=True)

    df_filt_st6 = df_filt_st6.sort("ST6_MaxThickness", descending=True)

    # Combine data frames

    df_combined = polars.concat(
        [
            df_filt_st1,
            df_filt_st2,
            df_filt_st3,
            df_filt_st4,
            df_filt_st5,
            df_filt_st6,
        ],
        how="horizontal",
    )

    # Add thickness distribution check

    df_combined = df_combined.with_columns(
        polars.when(
            (polars.col("ST1_MaxThickness") >= polars.col("ST2_MaxThickness"))
            & (polars.col("ST2_MaxThickness") >= polars.col("ST3_MaxThickness"))
            & (polars.col("ST3_MaxThickness") >= polars.col("ST4_MaxThickness"))
            & (polars.col("ST4_MaxThickness") >= polars.col("ST5_MaxThickness"))
            & (polars.col("ST5_MaxThickness") >= polars.col("ST6_MaxThickness"))
        )
        .then(True)
        .otherwise(False)
        .alias("MaxThicknessDistributionAdherence")
    )

    MinSamples = min(
        df_filt_st1.shape,
        df_filt_st2.shape,
        df_filt_st3.shape,
        df_filt_st4.shape,
        df_filt_st5.shape,
        df_filt_st6.shape,
    )

    df_combined = df_combined.head(MinSamples[0])

    df_combined = df_combined.filter(
        polars.col("MaxThicknessDistributionAdherence") == True
    )

    # Extract numpy array noise vectors

    noise_vector_filt_st1 = Station_1_Noise_Vector[
        df_combined.select(polars.col("ST1_row_nr")).to_numpy().flatten()
    ]

    noise_vector_filt_st2 = Station_2_Noise_Vector[
        df_combined.select(polars.col("ST2_row_nr")).to_numpy().flatten()
    ]

    noise_vector_filt_st3 = Station_3_Noise_Vector[
        df_combined.select(polars.col("ST3_row_nr")).to_numpy().flatten()
    ]

    noise_vector_filt_st4 = Station_4_Noise_Vector[
        df_combined.select(polars.col("ST4_row_nr")).to_numpy().flatten()
    ]

    noise_vector_filt_st5 = Station_5_Noise_Vector[
        df_combined.select(polars.col("ST5_row_nr")).to_numpy().flatten()
    ]

    noise_vector_filt_st6 = Station_6_Noise_Vector[
        df_combined.select(polars.col("ST6_row_nr")).to_numpy().flatten()
    ]

    unified_noise_vector = np.hstack(
        [
            noise_vector_filt_st1,
            noise_vector_filt_st2,
            noise_vector_filt_st3,
            noise_vector_filt_st4,
            noise_vector_filt_st5,
            noise_vector_filt_st6,
        ]
    ).reshape(df_combined.shape[0], -1)

    final_array = []
    for sub_array in unified_noise_vector:
        C1 = np.random.uniform(
            low=Station_1_Chord_Bounds[0], high=Station_1_Chord_Bounds[1], size=1
        )
        C2 = np.random.uniform(
            low=Station_2_Chord_Bounds[0], high=Station_2_Chord_Bounds[1], size=1
        )
        C3 = np.random.uniform(
            low=Station_3_Chord_Bounds[0], high=Station_3_Chord_Bounds[1], size=1
        )
        C4 = np.random.uniform(
            low=Station_4_Chord_Bounds[0], high=Station_4_Chord_Bounds[1], size=1
        )
        C5 = np.random.uniform(
            low=Station_5_Chord_Bounds[0], high=Station_5_Chord_Bounds[1], size=1
        )
        C6 = np.random.uniform(
            low=Station_6_Chord_Bounds[0], high=Station_6_Chord_Bounds[1], size=1
        )
        L1 = np.random.uniform(
            low=Station_1_Wingspan_Increments_Bounds[0],
            high=Station_1_Wingspan_Increments_Bounds[1],
            size=1,
        )
        L2 = np.random.uniform(
            low=Station_2_Wingspan_Increments_Bounds[0],
            high=Station_2_Wingspan_Increments_Bounds[1],
            size=1,
        )
        L3 = np.random.uniform(
            low=Station_3_Wingspan_Increments_Bounds[0],
            high=Station_3_Wingspan_Increments_Bounds[1],
            size=1,
        )
        L4 = np.random.uniform(
            low=Station_4_Wingspan_Increments_Bounds[0],
            high=Station_4_Wingspan_Increments_Bounds[1],
            size=1,
        )
        L5 = np.random.uniform(
            low=Station_5_Wingspan_Increments_Bounds[0],
            high=Station_5_Wingspan_Increments_Bounds[1],
            size=1,
        )
        final_array.append(
            np.insert(
                sub_array.reshape(1, 60),
                obj=len(sub_array),
                values=[
                    C1[0],
                    C2[0],
                    C3[0],
                    C4[0],
                    C5[0],
                    C6[0],
                    L1[0],
                    L2[0],
                    L3[0],
                    L4[0],
                    L5[0],
                ],
            )
        )

    np.savetxt(
        Prelim_Generated_Candidate_Airfoil_Solutions,
        np.array(final_array).reshape(len(final_array), (10 * 6 + 6 + 5)),
    )

    return f"{MinSamples[0]} Candidate Samples Generated"


def dump_candidate_solution_genome(
    NoiseVector: np.array, ObjectiveFunctionValue: float, FilePath
):
    # Convert numpy array to string
    array_str = np.array2string(NoiseVector)

    # Create a combined string
    combined_str = f"{array_str}\n\n{ObjectiveFunctionValue}"

    # Write the combined string to a text file
    with open(FilePath, "w") as file:
        file.write(combined_str)


def initiate_empty_file(FilePath: str, init_val: float):
    # Write the combined string to a text file
    with open(FilePath, "w") as file:
        file.write(str(init_val))
