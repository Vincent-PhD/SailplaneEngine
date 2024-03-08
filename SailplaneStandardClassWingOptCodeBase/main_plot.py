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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
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


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Add this line

    while True:
        if Plot_Airfoil:
            # List of data files
            data_files = [
                Station_1_Airfoil_Coordinate_File,
                Station_2_Airfoil_Coordinate_File,
                Station_3_Airfoil_Coordinate_File,
                Station_4_Airfoil_Coordinate_File,
                Station_5_Airfoil_Coordinate_File,
                Station_6_Airfoil_Coordinate_File,
                # Planform_And_Airfoil_Coordinate_File,
            ]

            # Create processes for each airfoil plot generation
            processes = []
            for file in data_files:
                process = multiprocessing.Process(
                    target=generate_plots_wrapper,
                    args=(file,),
                )
                processes.append(process)
                process.start()

            # Wait for all processes to complete
            for process in processes:
                process.join()


# def create_plot():
#     if Plot_Performance_Curves:
#         # Run Code
#         generate_performance_plots(PerformanceFilePath=Performance_Data_Path)

#     if Plot_Airfoil:
#         # Run Code
#         generate_airfoil_plots()  # Initial generation of plots

#     if Plot_Planfrom:
#         # Run Code
#         generate_planform_plots(
#             PlanformFilePath=Planform_Coordinate_File
#         )  # Initial generation of plots
#     # Load images
#     airfoil_folder_path = "./Data/AirfoilData/"
#     image_paths = (
#         [
#             airfoil_folder_path + f
#             for f in os.listdir(airfoil_folder_path)
#             if f.endswith(".dat")
#         ]
#         + [Planform_Coordinate_File]
#         + [Performance_Data_Path]
#     )
#     image_paths = [f.replace(".dat", ".png") for f in image_paths]
#     images = [mpimg.imread(path) for path in image_paths]

#     # Create subplots
#     fig, axes = plt.subplots(4, 2, figsize=(12, 10))

#     # Plot images in subplots
#     for i, ax in enumerate(axes.flat):
#         if i < len(images):
#             ax.imshow(images[i])
#             ax.axis("off")
#         else:
#             ax.axis("off")  # Turn off empty subplots

#     # Adjust layou
#     plt.draw()
#     plt.pause(10)
#     plt.cla()
#     plt.clf()


# def monitor_file(file_path, callback):
#     # Get the initial modification time of the file
#     last_modified = os.path.getmtime(file_path)

#     while True:
#         # Check if the file has been modified
#         current_modified = os.path.getmtime(file_path)
#         if current_modified > last_modified:
#             # File has been updated, trigger the callback function
#             last_modified = current_modified
#             callback()

#         # Wait for a short duration before checking again
#         time.sleep(0.1)  # Adjust the interval as needed


# monitor_file(Planform_And_Airfoil_Coordinate_File, create_plot)
