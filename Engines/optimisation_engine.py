from Engines.wing_performance_engine import get_wing_performance
import numpy as np
from geneticalgorithm import geneticalgorithm as gan
import pygad
from config import (
    N_Candidate_Solutions,
    Prelim_Generated_Candidate_Airfoil_Solutions,
    GA_Hyperparameters,
    Gen_New_Candidate_Solutions,
    # Constrain_Planform_Shape,
    # Station_1_Chord_Bounds,
    # Station_2_Chord_Bounds,
    # Station_3_Chord_Bounds,
    # Station_4_Chord_Bounds,
    # Station_5_Chord_Bounds,
    # Station_6_Chord_Bounds,
    # Station_1_Wingspan_Increments_Bounds,
    # Station_2_Wingspan_Increments_Bounds,
    # Station_3_Wingspan_Increments_Bounds,
    # Station_4_Wingspan_Increments_Bounds,
    # Station_5_Wingspan_Increments_Bounds,
)
from Utilities.optimisation_utilities import (
    generate_constrained_population,
    initiate_empty_file,
)


def define_optimisation_problem(
    SelectedOptimisationMethod: str, SelectedAirfoilGenerationMethod: str
):

    initiate_empty_file(
        FilePath="Data\\OptimisationData\\NoiseVectorTracking.dat",
        init_val=-1.000000000000000000e07,
    )
    initiate_empty_file(
        FilePath="Data\\OptimisationData\\PerformanceTracking.dat",
        init_val=-1.000000000000000000e07,
    )
    initiate_empty_file(
        FilePath="Data\\OptimisationData\\PenaltyTracking.dat",
        init_val=-1.000000000000000000e07,
    )

    if SelectedOptimisationMethod == "GeneticAlgorithm":
        if Gen_New_Candidate_Solutions:
            generate_constrained_population(TotalSamples=N_Candidate_Solutions)
            Candidate_Solutions = np.loadtxt(
                Prelim_Generated_Candidate_Airfoil_Solutions
            )
        else:
            Candidate_Solutions = np.loadtxt(
                "Data\\OptimisationData\\100_Candidate_Solutions.dat"
            )

        # To prepare the initial population, there are 2 ways:
        # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
        # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.

        # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
        ga_instance = pygad.GA(
            num_generations=GA_Hyperparameters["num_generations"],
            num_parents_mating=GA_Hyperparameters["num_parents_mating"],
            fitness_func=get_wing_performance,
            initial_population=Candidate_Solutions,
            crossover_type="uniform",
        )

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
        ga_instance.plot_fitness()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")

        if ga_instance.best_solution_generation != -1:
            print(
                f"Best fitness value reached after {ga_instance.best_solution_generation} generations."
            )

        # Saving the GA instance.
        filename = "Data\OptimisationData\genetic"  # The filename to which the instance is saved. The name is without extension.
        ga_instance.save(filename=filename)

        # Loading the saved GA instance.
        loaded_ga_instance = pygad.load(filename=filename)
        loaded_ga_instance.plot_fitness()

    # if (
    #     SelectedOptimisationMethod == "GeneticAlgorithm"
    #     and SelectedAirfoilGenerationMethod == "CST"
    # ):

    #     lw = -1
    #     uw = 1

    #     if Constrain_Planform_Shape:

    #         varbound = np.array(
    #             [(lw, uw)] * 102
    #             + [
    #                 Station_1_Chord_Bounds,
    #                 Station_2_Chord_Bounds,
    #                 Station_3_Chord_Bounds,
    #                 Station_4_Chord_Bounds,
    #                 Station_5_Chord_Bounds,
    #                 Station_6_Chord_Bounds,
    #             ]
    #             + [
    #                 Station_1_Wingspan_Increments_Bounds,
    #                 Station_2_Wingspan_Increments_Bounds,
    #                 Station_3_Wingspan_Increments_Bounds,
    #                 Station_4_Wingspan_Increments_Bounds,
    #                 Station_5_Wingspan_Increments_Bounds,
    #             ]
    #         )

    #     else:

    #         varbound = np.array(
    #             [(lw, uw)] * 102 + [(0.20, 0.95)] * 6 + [(1.00, 2.50)] * 5
    #         )

    #     vartype = np.array(np.array([["real"]] * 113))

    #     model = gan(
    #         function=get_wing_performance,
    #         dimension=113,
    #         variable_type_mixed=vartype,
    #         variable_boundaries=varbound,
    #         algorithm_parameters=algorithm_param,
    #         function_timeout=60 * 10,
    #     )

    #     model.run()

    # elif (
    #     SelectedOptimisationMethod == "GeneticAlgorithm"
    #     and SelectedAirfoilGenerationMethod == "GAN"
    # ):

    #     lw = -1
    #     uw = 1

    #     if Constrain_Planform_Shape:

    #         varbound = np.array(
    #             [(lw, uw)] * 60
    #             + [
    #                 Station_1_Chord_Bounds,
    #                 Station_2_Chord_Bounds,
    #                 Station_3_Chord_Bounds,
    #                 Station_4_Chord_Bounds,
    #                 Station_5_Chord_Bounds,
    #                 Station_6_Chord_Bounds,
    #             ]
    #             + [
    #                 Station_1_Wingspan_Increments_Bounds,
    #                 Station_2_Wingspan_Increments_Bounds,
    #                 Station_3_Wingspan_Increments_Bounds,
    #                 Station_4_Wingspan_Increments_Bounds,
    #                 Station_5_Wingspan_Increments_Bounds,
    #             ]
    #         )

    #     else:

    #         varbound = np.array(
    #             [(lw, uw)] * 60 + [(0.20, 0.95)] * 6 + [(1.00, 2.50)] * 5
    #         )

    #     vartype = np.array([["real"]] * 71)

    #     model = gan(
    #         function=get_wing_performance,
    #         dimension=71,
    #         variable_type_mixed=vartype,
    #         variable_boundaries=varbound,
    #         algorithm_parameters=algorithm_param,
    #         function_timeout=60 * 10,
    #     )
    #     model.run()
    else:
        RuntimeError
