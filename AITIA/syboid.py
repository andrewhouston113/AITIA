import numpy as np
from sklearn.neighbors import BallTree
from AITIA.complexity_measures import F1, N1
from pymop.problem import Problem
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skewnorm
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skewnorm
import warnings
warnings.filterwarnings('ignore')


class SyBoid:
    """
    A class for generating synthetic data using a genetic algorithm-based approach.

    This class allows you to configure and run a genetic algorithm to optimize synthetic data generation parameters based on F1 Score and N1 Score criteria. It can also mimic classes, data types, or the entire dataset as needed.

    Attributes:
    F1_Score (float): The F1 score used in the synthetic data generation.
    N1_Score (float): The N1 score used in the synthetic data generation.
    X (ndarray): The input data if available. (Either provide X or NumberOfBoids and Dimensions)
    y (ndarray): The target data if available. (Either provide y or Classes)
    NumberOfBoids (int): The number of boids to generate synthetic data for.
    Dimensions (int): The number of dimensions for the synthetic data.
    Classes (int): The number of unique classes in the synthetic data.
    Mimic_Classes (bool): Flag to indicate whether to mimic classes.
    Mimic_DataTypes (bool): Flag to indicate whether to mimic data types.
    Mimic_Dataset (bool): Flag to indicate whether to mimic the entire dataset.
    return_opt_stats (bool): Flag to indicate whether to return optimization statistics.

    Methods:
    - Generate_Data: Runs the genetic algorithm to generate synthetic data.
    - return_best_dataset(): Generates the best synthetic dataset using the optimized parameters.

    Usage Example:
    # Create a SyBoid instance with desired parameters
    syboid = SyBoid(F1_Score=0.8, N1_Score=0.2, X=input_data, Mimic_Classes=True, return_opt_stats=True)
    
    # Generate synthetic data
    synthetic_data, target = syboid.return_best_dataset()
    """

    def __init__(self, F1_Score, N1_Score, X=None, y=None, NumberOfBoids=None, Dimensions=None, Classes=None, 
                    Mimic_Classes=False, Mimic_DataTypes=False, Mimic_Dataset=False, return_opt_stats = False):
        
        # Check if either X or both NumberOfBoids and Dimensions are provided, and either y or Classes are provided.
        if ((X is None) and ((NumberOfBoids is None) or (Dimensions is None))) or ((y is None) and (Classes is None)):
            raise ValueError("You must provide either X or both NumberOfBoids and Dimensions, and either y or Classes.")
        
        # Check if the flag for dataset mimicry is set and ensure that both X and y are provided.
        if (Mimic_Dataset and ((X is None) or (y is None))):
            raise ValueError("You must provide X and y to mimic the dataset.")

        # Check if the flag for data type mimicry is set and ensure that X is provided.
        if (Mimic_DataTypes and (X is None)):
            raise ValueError("You must provide X to mimic data types.")
        
        # Check if the flag for class mimicry is set and ensure that y is provided.
        if (Mimic_Classes and (y is None)):
            raise ValueError("You must provide y to mimic classes.")

        # If X is provided, store the number of boids, dimensions, and X data.
        if X is not None:
            self.NumberOfBoids = X.shape[0]
            self.Dimensions = X.shape[1]
            self.X = X
        else:
            # If X is not provided, store the number of boids and dimensions from the provided arguments.
            self.NumberOfBoids = NumberOfBoids
            self.Dimensions = Dimensions
            self.X = X

        # If y is provided, calculate the number of unique classes, and store y data.
        if y is not None:
            self.Classes = np.unique(y).shape[0]
            self.y = y
        else:
            # If y is not provided, store the Classes value from the provided arguments.
            self.Classes = Classes
            self.y = y

        # Store the F1_Score and N1_Score values.
        self.F1_Score = F1_Score
        self.N1_Score = N1_Score

        # Store the flags for mimicry settings.
        self.Mimic_Classes = Mimic_Classes
        self.Mimic_DataTypes = Mimic_DataTypes
        self.Mimic_Dataset = Mimic_Dataset

        # Store the return settings
        self.return_opt_stats = return_opt_stats

    def Generate_Data(self, pop_size=40, n_gen=10):
        """
        Runs the genetic algorithm to generate synthetic data.

        Parameters:
        pop_size (int): Population size for the genetic algorithm. Default is 40.
        n_gen (int): Maximum number of generations to optimise for. Default is 10.

        Returns:
        pop (list): The final population of synthetic data.
        stats (object): The optimization statistics.
        """
        def main(seed=None):
            random.seed(seed)  # Set the random seed for reproducibility

            # Initialize statistics object for recording data
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"

            pop = toolbox.population(n=MU)  # Initialize the population

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Compile statistics about the population
            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(invalid_ind), **record)

            hist_best = 1
            hist_best_gen = 0
            # Begin the generational process
            for gen in range(1, NGEN):
                offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Calculate best, average, and worst fitness for this generation
                fits = [np.array(ind.fitness.values) for ind in pop if ind.fitness.valid]
                best_fit = np.nanmin(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(fits)))
                avg_fit = np.nanmean(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(fits)))
                worst_fit = np.nanmax(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(fits)))

                # Print and update historical best fitness
                print("Gen: %.0f, Best: %.3f, Avg: %.3f, Worst: %.3f" % (gen,best_fit,avg_fit,worst_fit))
                if best_fit < hist_best:
                    hist_best = best_fit
                    hist_best_gen = gen
                elif (gen - hist_best_gen) > 4:
                    return pop, fits

                if best_fit < 0.015:
                    return pop, fits

                # Select the next generation population from parents and offspring
                pop = toolbox.select(pop + offspring, MU)

                # Compile statistics about the new population
                record = stats.compile(pop)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
            return pop, fits

        # Problem definition
        NOBJ = 2
        NDIM = 8
        P = 12
        BOUND_LOW, BOUND_UP = 0.0, 1  # Define lower and upper bounds for variables
        problem = SyntheticDataGenerator(F1_Score=self.F1_Score, N1_Score=self.N1_Score,
                                        X=self.X, y=self.y, NumberOfBoids=self.NumberOfBoids, Dimensions=self.Dimensions,
                                        Classes=self.Classes,
                                        Mimic_Classes=self.Mimic_Classes, Mimic_DataTypes=self.Mimic_DataTypes,
                                        Mimic_Dataset=self.Mimic_Dataset)

        # Algorithm parameters
        MU = pop_size  # Population size
        NGEN = n_gen  # Number of generations
        CXPB = 0.95  # Crossover probability
        MUTPB = 0.01  # Mutation probability

        # Create uniform reference point for NSGA-III
        ref_points = tools.uniform_reference_points(NOBJ, P)

        # Create classes for individuals and fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox initialization for genetic algorithm
        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        toolbox = base.Toolbox()
        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        pop, stats = main()  # Run the genetic algorithm

        # Find the best individual in the final population
        self.best_x = pop[np.argmin(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(stats))[0])]

        if self.return_opt_stats:
            return pop, stats  # Return the final population and statistics

    def return_best_dataset(self):
        """
        Generates the best synthetic dataset using the optimized parameters.

        Returns:
        positions (ndarray) : The final synthetic data positions.
        target (ndarray): The target values corresponding to the synthetic data positions.
        """
        x = np.copy(self.best_x)  # Create a copy of the best parameters found by the genetic algorithm

        # Define lower and upper bounds for specific parameters
        ll = np.array([0, 0, 0, 0, 0, 0, -100, -100])
        ul = np.array([1, 1, 1, 0.50, 1, 0.50, 100, 100])

        # Rescale certain parameters using the specified bounds
        x[3] = ll[3] + (ul[3] * x[3])
        x[5] = ll[5] + (ul[5] * x[5])
        x[6] = ll[6] + (ul[6] * x[6])
        x[7] = ll[7] + (ul[7] * x[7])

        # Simulation Parameters
        timesteps = 15  # Number of simulation time steps

        # Initialize X and y data based on configuration settings
        if self.Mimic_Dataset:
            positions = np.zeros((self.NumberOfBoids, self.Dimensions, timesteps))
            positions[:, :, 0] = self.X
            target = self.y
        elif self.Mimic_Classes:
            # Initialize positions with random values if mimicking classes
            np.random.seed(0)
            positions = np.zeros((self.NumberOfBoids, self.Dimensions, timesteps))
            positions[:, :, 0] = np.random.uniform(0, 1, (self.NumberOfBoids, self.Dimensions))
            target = self.y
        else:
            # Initialize positions with random values if not mimicking classes
            np.random.seed(0)
            positions = np.zeros((self.NumberOfBoids, self.Dimensions, timesteps))
            positions[:, :, 0] = np.random.uniform(0, 1, (self.NumberOfBoids, self.Dimensions))
            target = np.random.randint(0, self.Classes, self.NumberOfBoids)

        if self.Mimic_DataTypes:
            # Determine which features have correct discrete data types
            correct_discrete = np.sum(self.X, axis=0) % 1 == 0

        # Initialize obedience values using a skewed normal distribution
        ObedienceValues = skewnorm.rvs(x[6], size=self.NumberOfBoids, random_state=42)
        ObedienceValues = 1 - ((ObedienceValues - np.min(ObedienceValues)) / (np.max(ObedienceValues) - np.min(ObedienceValues)))

        # Initialize Feature Importance using a skewed normal distribution
        FeatureImportance = skewnorm.rvs(x[7], loc=1, scale=0.5, size=self.Dimensions, random_state=42)
        FeatureImportance[FeatureImportance < 0] = 0
        FeatureImportance[FeatureImportance > 1] = 1

        # Simulate boids' movements over time
        for i in range(1, timesteps):
            positions[:, :, i] = _boid(positions[:, :, i - 1], target, x[0], x[1], x[2], x[3], x[4], x[5], FeatureImportance, ObedienceValues)

        if self.Mimic_DataTypes:
            # If mimicking data types, round the final positions to the nearest integer for discrete features
            for i in range(correct_discrete.shape[0]):
                if correct_discrete[i]:
                    X_fake = MinMaxScaler().fit_transform(positions[:, i, -1].reshape(-1, 1))
                    ul = max(self.X[:, i])
                    ll = min(self.X[:, i])
                    scaled = ll + (ul * X_fake).reshape(1, -1)
                    positions[:, i, -1] = np.round(scaled, 0)

        return positions[:, :, -1], target  # Return the final positions and target values



class SyntheticDataGenerator(Problem):
    """
    A class for generating synthetic data and evaluating the generated dataset against the pre-defined specification.

    Parameters:
    F1_Score: The target F1 score used for simulation evaluation.
    N1_Score: The target N1 score used for simulation evaluation.
    X: The input data (if provided).
    y: The target data (if provided).
    NumberOfBoids: The number of boids (if 'X' is not provided).
    Dimensions: The dimensions of the data (if 'X' is not provided).
    Classes: The number of classes (if 'y' is not provided).
    Mimic_Classes: A flag indicating whether to mimic classes during simulation. Default is false.
    Mimic_DataTypes: A flag indicating whether to mimic data types during simulation. Default is false.
    Mimic_Dataset: A flag indicating whether to mimic the entire dataset during simulation. Default is false.

    Methods:
    _evaluate(x, out): Simulate the system based on input parameters 'x' and return F1 and N1 scores.

    Attributes:
    - F1_Score: The target F1 score used for evaluation.
    - N1_Score: The target N1 score used for evaluation.
    - X: The input data (if provided).
    - y: The target data (if provided).
    - NumberOfBoids: The number of boids in the simulation.
    - Dimensions: The dimensions of the data or positions in the simulation.
    - Classes: The number of classes in the dataset.
    - Mimic_Classes: A flag indicating whether class mimicry is enabled.
    - Mimic_DataTypes: A flag indicating whether data type mimicry is enabled.
    - Mimic_Dataset: A flag indicating whether dataset mimicry is enabled.

    Note: This class is designed for the DEAP framework and should be extended and customized as needed.
    """
    def __init__(self, F1_Score, N1_Score, X=None, y=None, NumberOfBoids=None, Dimensions=None, Classes=None, 
                    Mimic_Classes=False, Mimic_DataTypes=False, Mimic_Dataset=False):

        # If X is provided, store the number of boids, dimensions, and X data.
        if X is not None:
            self.NumberOfBoids = X.shape[0]
            self.Dimensions = X.shape[1]
            self.X = X
        else:
            # If X is not provided, store the number of boids and dimensions from the provided arguments.
            self.NumberOfBoids = NumberOfBoids
            self.Dimensions = Dimensions

        # If y is provided, calculate the number of unique classes, and store y data.
        if y is not None:
            self.Classes = np.unique(y).shape[0]
            self.y = y
        else:
            # If y is not provided, store the Classes value from the provided arguments.
            self.Classes = Classes

        # Store the F1_Score and N1_Score values.
        self.F1_Score = F1_Score
        self.N1_Score = N1_Score

        # Store the flags for mimicry settings.
        self.Mimic_Classes = Mimic_Classes
        self.Mimic_DataTypes = Mimic_DataTypes
        self.Mimic_Dataset = Mimic_Dataset
        
        super().__init__(n_var=8, 
                         n_obj=2, 
                         n_constr=0, 
                         xl=np.array([0, 0, 0, 0, 0, 0, -100, -100]), 
                         xu=np.array([1, 1, 1, 0.50, 1, 0.50, 100, 100]), 
                         evaluation_of="auto")

    def _evaluate(self, x, out):
        """
        Evaluate the simulation based on input parameters and return F1 and N1 scores.

        Parameters:
        x: A 1D array of input parameters for the simulation.
        out: A dictionary to store the evaluation results.
        
        Returns:
        The function does not return a value but updates the 'out' dictionary with the F1 and N1 scores.
        """
        # Extract the first element of the input array 'x' (assuming it's a 1D array).
        x = x[0]

        # Define lower and upper bounds for specific elements in 'x'.
        ll = np.array([0, 0, 0, 0, 0, 0, -100, -100])
        ul = np.array([1, 1, 1, 0.50, 1, 0.50, 100, 100])

        # Apply scaling to selected elements in 'x' using the bounds.
        x[3] = ll[3] + (ul[3] * x[3])
        x[5] = ll[5] + (ul[5] * x[5])
        x[6] = ll[6] + (ul[6] * x[6])
        x[7] = ll[7] + (ul[7] * x[7])

        # Simulation Parameters
        timesteps = 15

        # Initialize X and y for the simulation based on various conditions.
        if self.Mimic_Dataset:
            # Use provided X and y for the simulation.
            positions = np.zeros((self.NumberOfBoids, self.Dimensions, timesteps))
            positions[:, :, 0] = self.X
            target = self.y
        elif self.Mimic_Classes:
            # Generate random positions and use provided y for the simulation.
            np.random.seed(0)
            positions = np.zeros((self.NumberOfBoids, self.Dimensions, timesteps))
            positions[:, :, 0] = np.random.uniform(0, 1, (self.NumberOfBoids, self.Dimensions))
            target = self.y
        else:
            # Generate random positions and random target values for the simulation.
            np.random.seed(0)
            positions = np.zeros((self.NumberOfBoids, self.Dimensions, timesteps))
            positions[:, :, 0] = np.random.uniform(0, 1, (self.NumberOfBoids, self.Dimensions))
            target = np.random.randint(0, self.Classes, self.NumberOfBoids)

        if self.Mimic_DataTypes:
            # Identify discrete features in the X data and apply scaling.
            correct_discrete = np.sum(self.X, axis=0) % 1 == 0

        # Initialize obedience values using a random distribution.
        ObedienceValues = skewnorm.rvs(x[6], size=self.NumberOfBoids, random_state=42)
        ObedienceValues = 1 - ((ObedienceValues - np.min(ObedienceValues)) / ((np.max(ObedienceValues)) - np.min(ObedienceValues)))

        # Initialize Feature Importance using a random distribution.
        FeatureImportance = skewnorm.rvs(x[7], loc=1, scale=0.5, size=self.Dimensions, random_state=42)
        FeatureImportance[FeatureImportance < 0] = 0
        FeatureImportance[FeatureImportance > 1] = 1

        try:
            # Simulate boids for multiple timesteps.
            for i in range(1, timesteps):
                positions[:, :, i] = _boid(positions[:, :, i-1], target, x[0], x[1], x[2], x[3], x[4], x[5], FeatureImportance, ObedienceValues)

            if self.Mimic_DataTypes:
                # If mimicking data types, adjust positions based on scaling.
                for i in range(correct_discrete.shape[0]):
                    if correct_discrete[i]:
                        X_fake = MinMaxScaler().fit_transform(positions[:, i, -1].reshape(-1, 1))
                        ul = max(self.X[:, i])
                        ll = min(self.X[:, i])
                        scaled = ll + (ul * X_fake).reshape(1, -1)
                        positions[:, i, -1] = np.round(scaled, 0)

            # Evaluate the simulation by calculating F1 and N1 scores.
            f1_ = F1(positions[:, :, -1], target)
            n1_ = N1(positions[:, :, -1], target)
            f1 = abs(self.F1_Score - f1_)
            n1 = abs(self.N1_Score - n1_)

            if np.nan in [f1, n1]:
                # Handle the case where NaN values are encountered.
                f1 = 1
                n1 = 1

        except:
            # Handle exceptions that may occur during the simulation.
            f1 = 1
            n1 = 1

        # Store the F1 and N1 scores in the 'out' dictionary.
        out["F"] = np.column_stack([f1, n1])


def _boid(flock_pos, target, FieldOfView, WeightCenterOfMass, 
         WeightSeparation, MinSeparation, WeightAvoidance, 
         AvoidanceTolerance, FeatureImportance, ObedienceValues):
    """
    Simulate the flocking behavior of a group of boids.

    Args:
    flock_pos (numpy.ndarray): An array representing the positions of boids.
    target (numpy.ndarray): An array specifying the target class for each boid.
    FieldOfView (float): The field of view angle for each boid.
    WeightCenterOfMass (float): Weight of the center of mass rule.
    WeightSeparation (float): Weight of the separation rule.
    MinSeparation (float): Minimum separation distance for the separation rule.
    WeightAvoidance (float): Weight of the avoidance rule.
    AvoidanceTolerance (float): Tolerance distance for the avoidance rule.
    FeatureImportance (float): Weight for the overall importance of rules.
    ObedienceValues (numpy.ndarray): An array of obedience values for each boid.

    Returns:
    numpy.ndarray: An updated array of boid positions after applying flocking rules.
    """
    def _get_same_class_idx(inst,boid):
      return inst[np.where(target[inst]==boid)[0]]

    def _get_diff_class_idx(inst,boid):
      return inst[np.where(target[inst]!=boid)[0]]

    def _get_same_class_dist(inst,dist,boid):
      return dist[np.where(target[inst]==boid)[0]]

    def _get_diff_class_dist(inst,dist,boid):
      return dist[np.where(target[inst]!=boid)[0]]

    tree = BallTree(flock_pos, leaf_size=10)
    inst_dist, inst_idx = tree.query(flock_pos, k=int(flock_pos.shape[0]*FieldOfView))
    inst_idx = inst_idx[:,1:]
    inst_dist = inst_dist[:,1:]

    same_class_idx = list(map(_get_same_class_idx, inst_idx, target))
    diff_class_idx = list(map(_get_diff_class_idx, inst_idx, target))
    same_class_dist = list(map(_get_same_class_dist, inst_idx, inst_dist, target))
    diff_class_dist = list(map(_get_diff_class_dist, inst_idx, inst_dist, target))

    for boid in range(flock_pos.shape[0]):
        # Rule 1: Cohesion
        rule_1 = (np.mean(flock_pos[same_class_idx[boid],:],axis=0) - flock_pos[boid,:])*((FeatureImportance*WeightCenterOfMass)*ObedienceValues[boid])

        # Rule 2: Seperation
        idx = np.where(same_class_dist[boid]<MinSeparation)[0]
        differences = flock_pos[same_class_idx[boid][idx],:] - flock_pos[boid,:]
        distances = same_class_dist[boid][idx]
        rule_2 = np.zeros(flock_pos.shape[1]) - sum((differences/distances[:,np.newaxis])/distances[:,np.newaxis])
        rule_2 = rule_2 * ((FeatureImportance*WeightSeparation)*ObedienceValues[boid])

        # Rule 3: Avoidance
        idx = np.where(diff_class_dist[boid]<AvoidanceTolerance)[0]
        differences = flock_pos[diff_class_idx[boid][idx],:] - flock_pos[boid,:]
        distances = diff_class_dist[boid][idx]
        rule_3 = np.zeros(flock_pos.shape[1]) - sum((differences/distances[:,np.newaxis])/distances[:,np.newaxis])
        rule_3 = rule_3 * ((FeatureImportance*WeightAvoidance)*ObedienceValues[boid])

        flock_vel = rule_1 + rule_2 + rule_3
        flock_pos[boid,:] = flock_pos[boid,:] + flock_vel
    return flock_pos