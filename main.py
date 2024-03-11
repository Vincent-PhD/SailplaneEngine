from config import Optimisation_Method, Airfoil_Generation_Method, Optimisation_Engine
from Engines.optimisation_engine import define_optimisation_problem


define_optimisation_problem(
    SelectedOptimisationMethod=Optimisation_Method,
    SelectedAirfoilGenerationMethod=Airfoil_Generation_Method,
    SelectedOptimisationEngine=Optimisation_Engine
)
