import matplotlib.pyplot as plt
import numpy as np
from config import (
    Plot_Airfoil,
    Plot_Planfrom,
    Plot_Performance_Curves,
    Planform_Coordinate_File,
    Lift_Over_Drag_Base,
    Free_Stream_Speed_Base,
    V_Free_Stream,
    Performance_Data_Path,
    Planform_And_Airfoil_Coordinate_File,
    Station_1_Airfoil_Coordinate_File,
    Station_2_Airfoil_Coordinate_File,
    Station_3_Airfoil_Coordinate_File,
    Station_4_Airfoil_Coordinate_File,
    Station_5_Airfoil_Coordinate_File,
    Station_6_Airfoil_Coordinate_File,
)
import os
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
import time
import multiprocessing

plt.rcParams["figure.figsize"] = (6, 3)


def generate_performance_plots(PerformanceFilePath):

    optimised_lift_to_drag_performance = np.loadtxt(PerformanceFilePath)
    plt.plot(
        V_Free_Stream,
        optimised_lift_to_drag_performance,
        label="Optimised",
        marker="o",
    )
    plt.plot(
        Free_Stream_Speed_Base,
        Lift_Over_Drag_Base,
        label="Baseline",
        marker="o",
    )
    plt.xlabel("Free stream speed [km/h]")
    plt.ylabel("Lift-to-drag ratio")
    plt.legend()
    plt.grid()
    plt.savefig(PerformanceFilePath.replace(".dat", ".png"))
    plt.pause(10)  # Pause to allow the plot to update
    plt.show(block=False)
    plt.clf()


def generate_and_dump_planform_plots(
    PlanformFilePath: str,
):

    wingspanincrements = np.insert(np.loadtxt(PlanformFilePath).T[2], 0, 0)
    l2 = np.cumsum(wingspanincrements)
    chord = np.insert(
        np.loadtxt(PlanformFilePath).T[0],
        5,
        np.loadtxt(PlanformFilePath).T[1][-1],
    )
    plt.plot(l2, chord, color="darkblue")
    plt.plot(-l2, chord, color="darkblue")
    plt.plot(
        [np.max(l2) for i in range(0, 50)],
        np.linspace(0, np.min(chord), 50),
        color="darkblue",
    )
    plt.plot(
        [np.min(-l2) for i in range(0, 50)],
        np.linspace(0, np.min(chord), 50),
        color="darkblue",
    )
    plt.plot(
        np.linspace(np.min(-l2), np.max(l2), 50),
        [0 for i in range(0, 50)],
        color="darkblue",
    )
    plt.axis("equal")
    plt.grid()
    plt.savefig(PlanformFilePath.replace(".dat", ".png"))
    plt.pause(10)  # Pause to allow the plot to update
    plt.show(block=False)
    plt.clf()

    return


def generate_and_dump_airfoil_plot(AirfoilFilePath: str, Xcoords, Ycoords):
    plt.figure()
    plt.plot(Xcoords, Ycoords)
    plt.grid()
    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(AirfoilFilePath)
    plt.savefig(AirfoilFilePath.replace(".dat", ".jpg"))
    plt.pause(10)  # Pause to allow the plot to update
    plt.show(block=False)
    plt.clf()


def generate_airfoil_plots(AirfoilFilePath):
    generate_and_dump_airfoil_plot(
        AirfoilFilePath=AirfoilFilePath,
        Xcoords=np.loadtxt(AirfoilFilePath).T[0],
        Ycoords=np.loadtxt(AirfoilFilePath).T[1],
    )


def generate_planform_plots(PlanformFilePath):
    generate_and_dump_planform_plots(
        PlanformFilePath=PlanformFilePath,
    )


def generate_plots_wrapper(coordinate_file):
    if "Airfoil" in coordinate_file:
        generate_airfoil_plots(coordinate_file)
    elif "Planform" in coordinate_file:
        generate_planform_plots(coordinate_file)
    elif "Performance" in coordinate_file:
        generate_performance_plots(coordinate_file)

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
while True:
    # Station 1
    x = np.loadtxt(Station_1_Airfoil_Coordinate_File).T[0]
    y = np.loadtxt(Station_1_Airfoil_Coordinate_File).T[1]
    plt.gca().cla() # optionally clear axes
    plt.plot(x, y, c="r")
    plt.title("Station 1")
    plt.draw()
    plt.pause(0.1)
    plt.show(
    block=True
    ) # block=True lets the window stay open at the end of the animation.