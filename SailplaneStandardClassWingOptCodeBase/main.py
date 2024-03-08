from config import Optimisation_Method, Airfoil_Generation_Method
from Engines.optimisation_engine import define_optimisation_problem


define_optimisation_problem(
    SelectedOptimisationMethod=Optimisation_Method,
    SelectedAirfoilGenerationMethod=Airfoil_Generation_Method,
)
