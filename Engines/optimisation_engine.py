from Engines.wing_performance_engine import get_wing_performance
import numpy as np
from geneticalgorithm import geneticalgorithm as gan
import pygad
import scipy
from config import (
    N_Candidate_Solutions,
    Prelim_Generated_Candidate_Airfoil_Solutions,
    GA_Hyperparameters,
    Gen_New_Candidate_Solutions,
    # Constrain_Planform_Shape,
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
from Utilities.optimisation_utilities import (
    generate_constrained_population,
    initiate_empty_file,
)


def define_optimisation_problem(
    SelectedOptimisationMethod: str, SelectedAirfoilGenerationMethod: str, SelectedOptimisationEngine: str
):

    initiate_empty_file(
        FilePath="Data/OptimisationData/NoiseVectorTracking.dat",
        init_val=-1.000000000000000000e07,
    )
    
    initiate_empty_file(
        FilePath="Data/OptimisationData/PerformanceTracking.dat",
        init_val=-1.000000000000000000e07,
    )
    initiate_empty_file(
        FilePath="Data/OptimisationData/PenaltyTracking.dat",
        init_val=-1.000000000000000000e07,
    )


    initiate_empty_file(
        FilePath="Data/OptimisationData/AirfoilThicknessDistributionPenaltyTracking.dat",
        init_val=-1.000000000000000000e07,
    )

    initiate_empty_file(
        FilePath="Data/OptimisationData/AirfoilShapePenaltyTracking.dat",
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
                "Data/OptimisationData/100_Candidate_Solutions.dat"
            )

        if SelectedOptimisationEngine== "PyGad":

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

        elif SelectedOptimisationEngine== "Scipy":

            varbound = np.array(
                [(-1.5, 1.5)] * 60
                + [
                    Station_1_Chord_Bounds,
                    Station_2_Chord_Bounds,
                    Station_3_Chord_Bounds,
                    Station_4_Chord_Bounds,
                    Station_5_Chord_Bounds,
                    Station_6_Chord_Bounds,
                ]
                + [
                    Station_1_Wingspan_Increments_Bounds,
                    Station_2_Wingspan_Increments_Bounds,
                    Station_3_Wingspan_Increments_Bounds,
                    Station_4_Wingspan_Increments_Bounds,
                    Station_5_Wingspan_Increments_Bounds,
                ]
            )
            scipy.optimize.differential_evolution(get_wing_performance, 
                                                  varbound, 
                                                  args=(), 
                                                  strategy='best1bin', 
                                                  maxiter=1000, 
                                                  tol=0.001, 
                                                  mutation=(0.5, 1), 
                                                  recombination=0.7, 
                                                  seed=None, 
                                                  callback=None, 
                                                  disp=True, 
                                                  polish=True, 
                                                  init=Candidate_Solutions, 
                                                  atol=0, 
                                                  updating='immediate', 
                                                  workers=-1, 
                                                  constraints=(), 
                                                  x0=None, 
                                                  integrality=None, 
                                                  vectorized=False)

        
        else:
            RuntimeError


    else:
        RuntimeError
