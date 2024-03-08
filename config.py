import numpy as np

# Define Airfoil Generator Module to Use: [GAN, CST method]

Airfoil_Generation_Method = "GAN"

# Define Airfoil Performance Module to Use: [XFOIL, NeuralFoil, WindTunnel Data, PDLM]

Airfoil_Performance_Analysis_Module = "XFOIL"

# Define Optimisation Method: [GeneticAlgorithm, ...]

Optimisation_Method = "GeneticAlgorithm"

# Define Optimisation Algorithm Hyperparameters

GA_Hyperparameters = {"num_generations": 500, "num_parents_mating": 2}

# Define Plotting Configurations

Plot_Airfoil = True
Plot_Planfrom = True
Plot_Performance_Curves = True

# Define Optimisation Flight Speed Range

V_Free_Stream = [
    224.29,
    213.85,
    204.75,
    193.04,
    180.16,
    167.18,
    154.78,
    144.78,
    135.25,
    129.49,
    125.38,
    119.89,
    112.15,
    105.73,
    100.31,
    95.64,
    91.57,
    88.66,
]

# Define Planform Constraints

Constrain_Planform_Shape = True

# Define Chords Upper and Lower Bounds

Station_1_Chord_Bounds = (0.820000, 0.820000)
Station_2_Chord_Bounds = (0.800000, 0.800000)
Station_3_Chord_Bounds = (0.700000, 0.700000)
Station_4_Chord_Bounds = (0.535000, 0.535000)
Station_5_Chord_Bounds = (0.419000, 0.419000)
Station_6_Chord_Bounds = (0.210000, 0.210000)

# Define Wingspan Increments Upper and Lower Bounds

Station_1_Wingspan_Increments_Bounds = (2.200000, 2.200000)
Station_2_Wingspan_Increments_Bounds = (1.980000, 1.980000)
Station_3_Wingspan_Increments_Bounds = (1.800000, 1.800000)
Station_4_Wingspan_Increments_Bounds = (0.890000, 0.890000)
Station_5_Wingspan_Increments_Bounds = (0.670000, 0.670000)


# Define Performance Data File Path

Performance_Data_Path = "Data/PerformanceData/Lift_to_Drag_Performance.dat"

# Define Airfoil/ Planform Coordinate File Path (Can be Empty or Contain Coordinates)

Planform_And_Airfoil_Coordinate_File = "Data/PlanformData/Planform_Airfoil_Data.dat"

# Define Planform Coordinate File Path (Can be Empty or Contain Coordinates)

Planform_Coordinate_File = "Data/PlanformData/Planform_Data.dat"

# Define Station 1 Airfoil Coordinate File Path (Can be Empty or Contain Coordinates)

Station_1_Airfoil_Coordinate_File = "Data/AirfoilData/Station_1.dat"

# Define Station 2 Airfoil Coordinate File Path (Can be Empty or Contain Coordinates)

Station_2_Airfoil_Coordinate_File = "Data/AirfoilData/Station_2.dat"

# Define Station 3 Airfoil Coordinate File Path (Can be Empty or Contain Coordinates)

Station_3_Airfoil_Coordinate_File = "Data/AirfoilData/Station_3.dat"

# Define Station 4 Airfoil Coordinate File Path (Can be Empty or Contain Coordinates)

Station_4_Airfoil_Coordinate_File = "Data/AirfoilData/Station_4.dat"

# Define Station 5 Airfoil Coordinate File Path (Can be Empty or Contain Coordinates)

Station_5_Airfoil_Coordinate_File = "Data/AirfoilData/Station_5.dat"

# Define Station 6 Airfoil Coordinate File Path (Can be Empty or Contain Coordinates)

Station_6_Airfoil_Coordinate_File = "Data/AirfoilData/Station_6.dat"

# Define columns for the GAN Parsimonious Vector DataFrame Output

GanColumns = [
    "MaxCamber",
    "MaxCamberLocation",
    "MaxCamberRadius",
    "MaxThicknessLocation",
    "MaxThickness",
    "MaxThicknessTail90c",
    "MinBottomThickness",
    "MinBottomThicknessLocation",
    "MinBottomRadius",
    "MinTopThickness",
    "MinTopThicknessLocation",
    "MaxTopThickness",
    "MaxTopThicknessLocation",
    "MaxTopRadius",
    "MaxBottomThickness",
    "MaxBottomThicknessLocation",
    "TopSurfaceTailAngle",
    "BottomSurfaceTailAngle",
    "LECamberAngle",
    "TECamberAngle",
    "TopLERadius",
    "BottomLERadius",
    "MaxBottomRadius",
    "TEGap",
    "total_area",
    "x_centroid",
    "y_centroid",
]


# Define Station 1 Geometrical Constraints

Station_1_Airfoil_Constraints = {
    "MaxThickness": {"Lower": 0.130, "Upper": 0.225},
    "MaxThicknessLocation": {"Lower": 0.350, "Upper": 0.600},
    "MaxTopThickness": {"Lower": 0.080, "Upper": 0.225},
    "MaxTopThicknessLocation": {"Lower": 0.300, "Upper": 0.650},
    "MinBottomThickness": {"Lower": -0.050, "Upper": 0.000},
    "MaxCamberLocation": {"Lower": 0.350, "Upper": 0.650},
    "total_area": {"Lower": 0.100, "Upper": 0.135},
}

# Define Station 2 Geometrical Constraints

Station_2_Airfoil_Constraints = {
    "MaxThickness": {"Lower": 0.130, "Upper": 0.205},
    "MaxThicknessLocation": {"Lower": 0.350, "Upper": 0.600},
    "MaxTopThickness": {"Lower": 0.080, "Upper": 0.205},
    "MaxTopThicknessLocation": {"Lower": 0.300, "Upper": 0.650},
    "MinBottomThickness": {"Lower": -0.050, "Upper": 0.000},
    "MaxCamberLocation": {"Lower": 0.350, "Upper": 0.650},
    "total_area": {"Lower": 0.095, "Upper": 0.125},
}

# Define Station 3 Geometrical Constraints


Station_3_Airfoil_Constraints = {
    "MaxThickness": {"Lower": 0.130, "Upper": 0.170},
    "MaxThicknessLocation": {"Lower": 0.350, "Upper": 0.600},
    "MaxTopThickness": {"Lower": 0.080, "Upper": 0.170},
    "MaxTopThicknessLocation": {"Lower": 0.300, "Upper": 0.650},
    "MinBottomThickness": {"Lower": -0.050, "Upper": 0.000},
    "MaxCamberLocation": {"Lower": 0.350, "Upper": 0.650},
    "total_area": {"Lower": 0.085, "Upper": 0.110},
}


# Define Station 4 Geometrical Constraints


Station_4_Airfoil_Constraints = {
    "MaxThickness": {"Lower": 0.120, "Upper": 0.160},
    "MaxThicknessLocation": {"Lower": 0.350, "Upper": 0.600},
    "MaxTopThickness": {"Lower": 0.080, "Upper": 0.160},
    "MaxTopThicknessLocation": {"Lower": 0.300, "Upper": 0.650},
    "MinBottomThickness": {"Lower": -0.050, "Upper": 0.000},
    "MaxCamberLocation": {"Lower": 0.350, "Upper": 0.650},
    "total_area": {"Lower": 0.080, "Upper": 0.100},
}

# Define Station 5 Geometrical Constraints

Station_5_Airfoil_Constraints = {
    "MaxThickness": {"Lower": 0.110, "Upper": 0.150},
    "MaxThicknessLocation": {"Lower": 0.350, "Upper": 0.600},
    "MaxTopThickness": {"Lower": 0.080, "Upper": 0.150},
    "MaxTopThicknessLocation": {"Lower": 0.300, "Upper": 0.650},
    "MinBottomThickness": {"Lower": -0.050, "Upper": 0.000},
    "MaxCamberLocation": {"Lower": 0.350, "Upper": 0.650},
    "total_area": {"Lower": 0.080, "Upper": 0.100},
}

# Define Station 6 Geometrical Constraints

Station_6_Airfoil_Constraints = {
    "MaxThickness": {"Lower": 0.100, "Upper": 0.140},
    "MaxThicknessLocation": {"Lower": 0.350, "Upper": 0.600},
    "MaxTopThickness": {"Lower": 0.080, "Upper": 0.130},
    "MaxTopThicknessLocation": {"Lower": 0.300, "Upper": 0.650},
    "MinBottomThickness": {"Lower": -0.050, "Upper": 0.000},
    "MaxCamberLocation": {"Lower": 0.350, "Upper": 0.650},
    "total_area": {"Lower": 0.08, "Upper": 0.100},
}

# Define if New Constrained Candidate Solutions Need Be generated

Gen_New_Candidate_Solutions = True

# Define Number of Candidate Solutions

N_Candidate_Solutions = 5000

# Define Path to Generated Candidate Airfoil Solutions

Prelim_Generated_Candidate_Airfoil_Solutions = (
    f"Data/OptimisationData/{N_Candidate_Solutions}_Candidate_Solutions.dat"
)

# Define Path to Standard Atmosphere Tables

Standard_Atmosphere_Tables = "Data/UtilityData/Standard_atmosphere.txt"

# Define Path to Fuselage Performance Data

Fuselage_Performance_Data = "Data/UtilityData/JS3_Fuselage.dat"


# Define Baseline Design Free Stream Speeds
Free_Stream_Speed_Base = [
    224.29,
    213.85,
    204.75,
    193.04,
    180.16,
    167.18,
    154.78,
    144.78,
    135.25,
    129.49,
    125.38,
    119.89,
    112.15,
    105.73,
    100.31,
    95.64,
    91.57,
    88.66,
]


# Define Baseline Design Lift Over Drag Performance
Lift_Over_Drag_Base = [
    19.16206851231948,
    21.78412745328629,
    24.350339634024436,
    28.344567503802907,
    33.09276247215675,
    37.31954033213507,
    41.35029568644353,
    44.52595017885541,
    45.8305284310441,
    46.20997450839833,
    46.246717091675116,
    45.644305848880514,
    44.16428587875123,
    42.28289106960324,
    40.151238602928075,
    37.921538299028384,
    35.35219457480897,
    31.86488032707159,
]


# Define Flight Constants

_RHO = 1.225
_MU = 1.789e-5
_ALPHA = 5  # angle of attack (only for Lifting line solver) [deg]
_A = 2 * np.pi  # lift curve slope
_WEIGHT = 485 * 9.81  # [N]
_BANKANGLE = 0  # [deg]
_FLAP = 0  # [0 degrees is neutral or no flap]
Pressure_altitude = 0  # [m]
OAT = 15  # [Deg]
_WINGLET = 5  # [% contribution at low speeds]


# Specify the Parhs to ML/ DL Models and Scalers

# Define GAN-Generation Technique Model and Scaler Files

BLSTM_Model_Path = (
    "Models\\AirfoilGenerationModels\\MLModels\\pyfoilgen__blstm_model_6.h5"
)

BLSTM_Scaler_Path = "Models\\AirfoilGenerationModels\\Scalers\\pyfoilgen_scaler.joblib"

Generator_Modle_Path = (
    "Models\\AirfoilGenerationModels\\MLModels\\wgan_generator_0.9443675714351385.h5"
)

Generator_Scaler_Path = (
    "Models\\AirfoilGenerationModels\\Scalers\\wgan_generator_scaler.joblib"
)

# Define Performance Tracking File Path

Optimisation_Objective_Function_Tracking_Path = (
    "Data\\OptimisationData\\PerformanceTracking.dat"
)

# Define Performance Tracking File Path

Optimisation_Noise_Vector_Tracking_Path = (
    "Data\\OptimisationData\\NoiseVectorTracking.dat"
)
