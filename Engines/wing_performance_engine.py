import numpy as np
import os
import math
import concurrent.futures
import scipy
import matplotlib.pyplot as plt
from Utilities.wing_performance_module_utilities import *
from config import (
    Planform_And_Airfoil_Coordinate_File,
    Optimisation_Objective_Function_Tracking_Path,
    Optimisation_Noise_Vector_Tracking_Path,
    Planform_Coordinate_File,
    Free_Stream_Speed_Base,
    Lift_Over_Drag_Base,
    Airfoil_Generation_Method,
    _RHO,
    _MU,
    _A,
    _WEIGHT,
    _BANKANGLE,
    _FLAP,
    _ALPHA,
    _WINGLET,
    Station_1_Airfoil_Coordinate_File,
    Station_2_Airfoil_Coordinate_File,
    Station_3_Airfoil_Coordinate_File,
    Station_4_Airfoil_Coordinate_File,
    Station_5_Airfoil_Coordinate_File,
    Station_6_Airfoil_Coordinate_File,
    V_Free_Stream,
    Fuselage_Performance_Data,
    Plot_Airfoil,
    Performance_Data_Path,
    Station_1_Airfoil_Constraints,
    Station_2_Airfoil_Constraints,
    Station_3_Airfoil_Constraints,
    Station_4_Airfoil_Constraints,
    Station_5_Airfoil_Constraints,
    Station_6_Airfoil_Constraints,
)
from Modules.airfoil_generation_module import generate_airfoil
from Modules.wing_planform_generation_module import generate_planform
from Utilities.optimisation_utilities import (
    partition_noise_vector,
    planform_penalty,
    get_airfoil_penalty,
    rectify_planform_sampling_error,
    dump_candidate_solution_genome,
    airfoil_thickness_distribution_penalty,
    airfoil_quality_penalty
)


def get_wing_performance(ga_instance, solution, solution_idx):

    (
        airfoil_noise_1,
        airfoil_noise_2,
        airfoil_noise_3,
        airfoil_noise_4,
        airfoil_noise_5,
        airfoil_noise_6,
        chords,
        increments,
    ) = partition_noise_vector(
        NoiseVector=solution, SelectedGenerationMethod=Airfoil_Generation_Method
    )

    # Rectify Planform Errors Due to MisSampling from Optimisation Algorithm

    chords, increments = rectify_planform_sampling_error(chords, increments)

    ## Station 1
    x_1, y_1, df_1 = generate_airfoil(
        NoiseVector=airfoil_noise_1,
        SelectedGenerationMethod=Airfoil_Generation_Method,
        AirfoilDumpFilePath=Station_1_Airfoil_Coordinate_File,
        AllowAirfoilPlotting=Plot_Airfoil,
    )
    st1_airfoil_penalty, _ = get_airfoil_penalty(
        parsimonious_vars=df_1,
        feature_list=list(Station_1_Airfoil_Constraints.keys()),
        constraint_dictionary=Station_1_Airfoil_Constraints,
    )
    ## Station 2
    x_2, y_2, df_2 = generate_airfoil(
        NoiseVector=airfoil_noise_2,
        SelectedGenerationMethod=Airfoil_Generation_Method,
        AirfoilDumpFilePath=Station_2_Airfoil_Coordinate_File,
        AllowAirfoilPlotting=Plot_Airfoil,
    )
    st2_airfoil_penalty, _ = get_airfoil_penalty(
        parsimonious_vars=df_2,
        feature_list=list(Station_2_Airfoil_Constraints.keys()),
        constraint_dictionary=Station_2_Airfoil_Constraints,
    )
    ## Station 3
    x_3, y_3, df_3 = generate_airfoil(
        NoiseVector=airfoil_noise_3,
        SelectedGenerationMethod=Airfoil_Generation_Method,
        AirfoilDumpFilePath=Station_3_Airfoil_Coordinate_File,
        AllowAirfoilPlotting=Plot_Airfoil,
    )
    st3_airfoil_penalty, _ = get_airfoil_penalty(
        parsimonious_vars=df_3,
        feature_list=list(Station_3_Airfoil_Constraints.keys()),
        constraint_dictionary=Station_3_Airfoil_Constraints,
    )
    ## Station 4
    x_4, y_4, df_4 = generate_airfoil(
        NoiseVector=airfoil_noise_4,
        SelectedGenerationMethod=Airfoil_Generation_Method,
        AirfoilDumpFilePath=Station_4_Airfoil_Coordinate_File,
        AllowAirfoilPlotting=Plot_Airfoil,
    )
    st4_airfoil_penalty, _ = get_airfoil_penalty(
        parsimonious_vars=df_4,
        feature_list=list(Station_4_Airfoil_Constraints.keys()),
        constraint_dictionary=Station_4_Airfoil_Constraints,
    )
    ## Station 5
    x_5, y_5, df_5 = generate_airfoil(
        NoiseVector=airfoil_noise_5,
        SelectedGenerationMethod=Airfoil_Generation_Method,
        AirfoilDumpFilePath=Station_5_Airfoil_Coordinate_File,
        AllowAirfoilPlotting=Plot_Airfoil,
    )
    st5_airfoil_penalty, _ = get_airfoil_penalty(
        parsimonious_vars=df_5,
        feature_list=list(Station_4_Airfoil_Constraints.keys()),
        constraint_dictionary=Station_5_Airfoil_Constraints,
    )
    ## Station 6
    x_6, y_6, df_6 = generate_airfoil(
        NoiseVector=airfoil_noise_6,
        SelectedGenerationMethod=Airfoil_Generation_Method,
        AirfoilDumpFilePath=Station_6_Airfoil_Coordinate_File,
        AllowAirfoilPlotting=Plot_Airfoil,
    )
    st6_airfoil_penalty, _ = get_airfoil_penalty(
        parsimonious_vars=df_6,
        feature_list=list(Station_6_Airfoil_Constraints.keys()),
        constraint_dictionary=Station_6_Airfoil_Constraints,
    )
    # Total Airfoil Penalty

    _Total_Airfoil_Penalty = (
        abs(
            st1_airfoil_penalty
            + st2_airfoil_penalty
            + st3_airfoil_penalty
            + st4_airfoil_penalty
            + st5_airfoil_penalty
            + st6_airfoil_penalty
        )
        * 1e3
    )

    # Total Thickness Distribution Penalty

    _AirfoilThicknessDistributionPenalty = airfoil_thickness_distribution_penalty(
        x=np.array(
            [
                df_1["MaxThickness"],
                df_2["MaxThickness"],
                df_3["MaxThickness"],
                df_4["MaxThickness"],
                df_5["MaxThickness"],
                df_6["MaxThickness"],
            ]
        ),
        weight=1e3,
    )


    _PlanformPenalty = planform_penalty(x=chords, weight=5_000)

    _LengthPenalty = np.abs((16 - (2 * np.sum(increments))))


    # Calculate Airfoil Quality Penalty

    airfoil_quality_penalty(X:np.array, Y: np.array, weight = 1e3)


    if (_PlanformPenalty>0) or (_LengthPenalty>0) or (_Total_Airfoil_Penalty>0) or (_AirfoilThicknessDistributionPenalty>0):
        return -1*(_PlanformPenalty+_LengthPenalty+_Total_Airfoil_Penalty+_AirfoilThicknessDistributionPenalty)

    else:

        # Generate the wing planform with the generated/ optimised coordinates

        generate_planform(Chords=chords, WingspanIncrements=increments)

        # Specify Objective Function Collector
        lift_to_drag_collector = []

        Area, Span, AR, WingData = GetWingData(Planform_Coordinate_File)

        def calculate_lift_drag(V):
            CD, CL = Wing_CD(
                Planform_Coordinate_File,
                Planform_And_Airfoil_Coordinate_File,
                V / 3.6,
                _RHO,
                _MU,
                _BANKANGLE,
                _WEIGHT,
                _FLAP,
                _ALPHA,
                _A,
                _WINGLET,
            )

            Drag = (
                Fuselage_Drag(Fuselage_Performance_Data, V / 3.6)
                + (0.5 * _RHO * ((V / 3.6) ** 2) * CD * Area * 2) * 1.05
            )
            return _WEIGHT / Drag

        with concurrent.futures.ThreadPoolExecutor() as executor:
            lift_to_drag_collector = list(executor.map(calculate_lift_drag, V_Free_Stream))

        err = l_over_d_objective_value(
            V_baseline=Free_Stream_Speed_Base,
            l_over_d_baseline=Lift_Over_Drag_Base,
            V_optimised=V_Free_Stream,
            l_over_d_optimised=np.array(lift_to_drag_collector).flatten(),
        )
        obj_func = (np.sum(err)) - (
            _PlanformPenalty
            + _LengthPenalty
            + _Total_Airfoil_Penalty
            + _AirfoilThicknessDistributionPenalty
        )

        performance_tracker = np.loadtxt(Optimisation_Objective_Function_Tracking_Path)

        objective_tracker = np.loadtxt("Data/OptimisationData/PenaltyTracking.dat")
        objective_tracker = np.append(
            objective_tracker,
            (
                _PlanformPenalty
                + _LengthPenalty
                + _Total_Airfoil_Penalty
                + _AirfoilThicknessDistributionPenalty
            ),
        )
        np.savetxt("Data/OptimisationData/PenaltyTracking.dat", objective_tracker)

        if obj_func > np.max(performance_tracker):
            performance_tracker = np.append(performance_tracker, obj_func)
            np.savetxt(Optimisation_Objective_Function_Tracking_Path, performance_tracker)

            #   Station 1 Airfoil Dump
            generate_airfoil(
                NoiseVector=airfoil_noise_1,
                SelectedGenerationMethod=Airfoil_Generation_Method,
                AirfoilDumpFilePath=Station_1_Airfoil_Coordinate_File.replace(
                    "AirfoilData", "OptimisationData"
                ),
                AllowAirfoilPlotting=Plot_Airfoil,
            )

            #   Station 2 Airfoil Dump
            generate_airfoil(
                NoiseVector=airfoil_noise_2,
                SelectedGenerationMethod=Airfoil_Generation_Method,
                AirfoilDumpFilePath=Station_2_Airfoil_Coordinate_File.replace(
                    "AirfoilData", "OptimisationData"
                ),
                AllowAirfoilPlotting=Plot_Airfoil,
            )

            #   Station 3 Airfoil Dump
            generate_airfoil(
                NoiseVector=airfoil_noise_3,
                SelectedGenerationMethod=Airfoil_Generation_Method,
                AirfoilDumpFilePath=Station_3_Airfoil_Coordinate_File.replace(
                    "AirfoilData", "OptimisationData"
                ),
                AllowAirfoilPlotting=Plot_Airfoil,
            )

            #   Station 4 Airfoil Dump
            generate_airfoil(
                NoiseVector=airfoil_noise_4,
                SelectedGenerationMethod=Airfoil_Generation_Method,
                AirfoilDumpFilePath=Station_4_Airfoil_Coordinate_File.replace(
                    "AirfoilData", "OptimisationData"
                ),
                AllowAirfoilPlotting=Plot_Airfoil,
            )

            #   Station 5 Airfoil Dump
            generate_airfoil(
                NoiseVector=airfoil_noise_5,
                SelectedGenerationMethod=Airfoil_Generation_Method,
                AirfoilDumpFilePath=Station_5_Airfoil_Coordinate_File.replace(
                    "AirfoilData", "OptimisationData"
                ),
                AllowAirfoilPlotting=Plot_Airfoil,
            )

            #   Station 6 Airfoil Dump
            generate_airfoil(
                NoiseVector=airfoil_noise_6,
                SelectedGenerationMethod=Airfoil_Generation_Method,
                AirfoilDumpFilePath=Station_6_Airfoil_Coordinate_File.replace(
                    "AirfoilData", "OptimisationData"
                ),
                AllowAirfoilPlotting=Plot_Airfoil,
            )

            dump_candidate_solution_genome(
                NoiseVector=solution,
                ObjectiveFunctionValue=obj_func,
                FilePath=Optimisation_Noise_Vector_Tracking_Path,
            )
            print("-----" * 17)
            print(f"Objective function value penalty: {np.round(obj_func)}")
            print(f"Planform penalty value penalty:   {np.round(_PlanformPenalty,2)}")
            print(f"Length penalty value penalty:     {np.round(_LengthPenalty,2)}")
            print(f"Airfoil shape penalty:    {np.round(_Total_Airfoil_Penalty,2)}")
            print(
                f"Airfoil thickness distribution penalty:    {np.round(_AirfoilThicknessDistributionPenalty,2)}"
            )

            print("-----" * 17)

        np.savetxt(Performance_Data_Path, lift_to_drag_collector)

        return obj_func

