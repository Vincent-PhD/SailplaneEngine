import numpy as np
from config import (
    Planform_And_Airfoil_Coordinate_File,
    Planform_Coordinate_File,
    Station_1_Airfoil_Coordinate_File,
    Station_2_Airfoil_Coordinate_File,
    Station_3_Airfoil_Coordinate_File,
    Station_4_Airfoil_Coordinate_File,
    Station_5_Airfoil_Coordinate_File,
    Station_6_Airfoil_Coordinate_File,
)


def generate_planform(Chords: list, WingspanIncrements: list):

    Station_1 = [
        Chords[0][0],
        Chords[0][1],
        WingspanIncrements[0][0],
        1.000000,
        1.000000,
    ]
    Station_2 = [
        Chords[0][1],
        Chords[0][2],
        WingspanIncrements[0][1],
        1.000000,
        1.000000,
    ]
    Station_3 = [
        Chords[0][2],
        Chords[0][3],
        WingspanIncrements[0][2],
        1.000000,
        1.000000,
    ]
    Station_4 = [
        Chords[0][3],
        Chords[0][4],
        WingspanIncrements[0][3],
        1.000000,
        1.000000,
    ]
    Station_5 = [
        Chords[0][4],
        Chords[0][5],
        WingspanIncrements[0][4],
        1.000000,
        1.000000,
    ]

    # Dump Planform Coordinate File
    np.savetxt(
        Planform_Coordinate_File,
        [Station_1, Station_2, Station_3, Station_4, Station_5],
        delimiter="   ",
    )

    ST1 = [Station_1_Airfoil_Coordinate_File, f"{Chords[0][0]}"]
    ST2 = [Station_2_Airfoil_Coordinate_File, f"{Chords[0][1]}"]
    ST3 = [Station_3_Airfoil_Coordinate_File, f"{Chords[0][2]}"]
    ST4 = [Station_4_Airfoil_Coordinate_File, f"{Chords[0][3]}"]
    ST5 = [Station_5_Airfoil_Coordinate_File, f"{Chords[0][4]}"]
    ST6 = [Station_6_Airfoil_Coordinate_File, f"{Chords[0][5]}"]

    data = [ST1, ST2, ST3, ST4, ST5, ST6]

    # Join the elements in each inner list with the delimiter and save as plain text
    formatted_data = ["   ".join(row) for row in data]

    # Dump Airfoil and Planform Coordinate File

    with open(Planform_And_Airfoil_Coordinate_File, "w") as file:
        file.write("\n".join(formatted_data))
