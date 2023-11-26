import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from zepid.graphics import EffectMeasurePlot
from AITIA.heuristics import KNeighbors, DisjunctSize, DisjunctClass, ClassLikelihood, HeursiticsCalculator
from AITIA.utils import generate_points_around_x
from AITIA.syboid import SyBoid
from AITIA.complexity_measures import F1, N1
import skfuzzy as fuzz
import itertools
import random


class MisclassificationExplainer:
    """
    MisclassificationExplainer class is designed to explain misclassifications made by a predictive model 
    through the analysis of heuristic scores and the generation of various evaluation metrics.

    Parameters:
        - n_neighbors (int, optional): Number of neighbors for the K-nearest neighbors model.
        - max_depth (int, optional): Maximum depth for the DisjunctClass model.
        - balanced (bool, optional): Whether to use balanced classes in the DisjunctClass model.

    Attributes:
        - KDN (KNeighbors): K-nearest neighbors model for calculating disagreement scores.
        - DS (DisjunctSize): DisjunctSize model for calculating DisjunctSize scores.
        - DCP (DisjunctClass): DisjunctClass model for calculating DisjunctClassProbability scores.
        - CLD (ClassLikelihood): ClassLikelihood model for calculating ClassLikelihoodDifference scores.
        - results (dict): Dictionary containing the results of the explanation.

    Methods:
        - fit(X, y, categorical_idx=[]): Fit the internal models to the input data.
        - calculate_heuristics(X, y): Calculate heuristic scores using the fitted models.
        - explain(model, X, y): Explain misclassifications using heuristic scores and logistic regression.
        - plot_results(): Generate plots displaying odds ratios, ROC curves, and Precision-Recall curves.

    Example Usage:
        explainer = MisclassificationExplainer(n_neighbors=10, max_depth=4, balanced=False)
        explainer.fit(X_train, y_train, categorical_idx=[2, 5, 7])
        explainer.explain(model, X_test, y_test)
        explainer.plot_results()
    """

    def __init__(self, n_neighbors=10, max_depth=4, balanced=False):
        self.KDN = KNeighbors(n_neighbors)
        self.DS = DisjunctSize()
        self.DCP = DisjunctClass(max_depth=max_depth, balanced=balanced)
        self.CLD = ClassLikelihood()

    def fit(self, X, y, categorical_idx=[]):
        """
        Fit the internal models to the input data.

        Parameters:
            - X (array-like): Feature matrix.
            - y (array-like): Target labels.
            - categorical_idx (list, optional): List of indices for categorical features.

        Returns:
            None
        """
        
        self.KDN.fit(X,y)
        self.DS.fit(X,y)
        self.DCP.fit(X,y)
        self.CLD.fit(X, y, categorical_idx=categorical_idx)
    
    def calculate_heuristics(self, X, y):
        """
        Calculate heuristic scores using the fitted models.

        Parameters:
            - X (array-like): Feature matrix.
            - y (array-like): Target labels.

        Returns:
            dict: Dictionary containing heuristic scores.
        """

        KDN_score = self.KDN.calculate_disagreement(X,y)
        DS_score = self.DS.calculate(X)
        DCP_score = self.DCP.calculate_percentage(X,y)
        CLD_score = self.CLD.calculate_class_likelihood_difference(X,y)
        
        return {'KDN_scores': KDN_score,
                'DS_scores': DS_score,
                'DCP_scores': DCP_score,
                'CLD_scores': CLD_score}

    def explain(self, model, X, y):
        """
        Explain misclassifications using heuristic scores.

        Parameters:
            - model: Predictive model with a `predict` method.
            - X (array-like): Feature matrix.
            - y (array-like): Target labels.

        Returns:
            None
        """

        # Make predictions using the provided model
        y_pred = model.predict(X)

        # Identify misclassifications
        misclassifications = y_pred != y

        # Calculate heuristic scores using the implemented method
        scores = self.calculate_heuristics(X, y)

        results = {}
        for heuristic, heuristic_scores in scores.items():
            # Standardize heuristic scores using z-score normalization
            heuristic_scores_ = stats.zscore(heuristic_scores)
            # Add a constant term for logistic regression
            heuristic_scores_ = sm.add_constant(heuristic_scores_)
            
            # Fit logistic regression model to predict misclassifications using heuristic scores
            res = sm.Logit(misclassifications, heuristic_scores_).fit()

            # Calculate Area Under the Receiver Operating Characteristic (AUROC) and Area Under the Precision-Recall Curve (AUPRC) score
            auroc = roc_auc_score(misclassifications, heuristic_scores)
            auprc = average_precision_score(misclassifications, heuristic_scores)

            # Generate ROC and Precision-Recall curve
            fpr, tpr, _ = roc_curve(misclassifications, heuristic_scores)
            precision, recall, _ = precision_recall_curve(misclassifications, heuristic_scores)

            # Store results for the current heuristic
            results[heuristic] = {
                'odds_ratio': np.exp(res.params[1]).round(3),
                'confidence_interval': np.exp(res.conf_int()[1,:]).round(3),
                'pvalue': res.pvalues[1].round(3),
                'auroc': auroc.round(3),
                'auprc': auprc.round(3),
                'roc_curve': {'fpr': list(fpr), 'tpr': list(tpr)},
                'precision_recall_curve': {'precision': list(precision), 'recall': list(recall)}
            }
        
        self.results = results

    def plot_results(self):
        """
        Generate plots displaying odds ratios, ROC curves, and Precision-Recall curves.

        Parameters:
            None

        Returns:
            None
        """

        # Produce forest plot for the odds ratios
        p = EffectMeasurePlot(label=[k for k in self.results.keys()], 
                              effect_measure=[v['odds_ratio'] for v in self.results.values()],
                            lcl=[v['confidence_interval'][0] for v in self.results.values()], 
                            ucl=[v['confidence_interval'][1] for v in self.results.values()])

        p.labels(effectmeasure='OR')
        p.colors(pointshape="D")
        ax=p.plot(figsize=(15,5), 
                  t_adjuster=0.09, 
                  max_value=int(np.ceil(max([v['confidence_interval'][1] for v in self.results.values()])*1.1)), 
                  min_value=int(0))
        ax.set_title('A) Odd Ratios')
        
        # Produce the ROC curves
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        legend = []
        for k, v in self.results.items():
            axes[0].plot(v['roc_curve']['fpr'], v['roc_curve']['tpr'], alpha=1)
            legend.append(f'{k}, AUROC = {np.round(v["auroc"], 2)}')
        
        axes[0].legend(legend, loc='lower right')
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.2)
        axes[0].set_xlim([-0.01, 1.01])  # 'xlim' and 'ylim' instead of 'set_xlim' and 'set_ylim'
        axes[0].set_ylim([-0.01, 1.01])
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_title('B) AUROC')

        # Produce the Precision-Recall curves
        legend = []
        for k, v in self.results.items():
            axes[1].plot(v['precision_recall_curve']['recall'], v['precision_recall_curve']['precision'], alpha=1)
            legend.append(f'{k}, AUPRC = {np.round(v["auprc"], 2)}')

        axes[1].set_xlim([-0.01, 1.01])  # 'xlim' and 'ylim' instead of 'set_xlim' and 'set_ylim'
        axes[1].set_ylim([-0.01, 1.01])
        axes[1].set_ylabel('Precision')
        axes[1].set_xlabel('Recall')
        axes[1].legend(legend, loc='upper right')
        axes[1].set_title('C) AUPRC')


class CompetencyAnalysis:
    """
    Class for analyzing the competency of a machine learning model across synthetic datasets.

    Parameters:
        - n_datasets (int, optional): Number of synthetic datasets to generate (default is 100).
        - max_distance (float, optional): Maximum distance for generating synthetic datasets around the actual dataset complexity (default is 0.4).
        - pop_size (int, optional): Population size for generating synthetic datasets (default is 40).
        - n_gen (int, optional): Number of generations for generating synthetic datasets (default is 10).

    Attributes:
        - n_datasets (int): Number of synthetic datasets to generate.
        - max_distance (float): Maximum distance for generating synthetic datasets around the actual dataset complexity.
        - pop_size (int): Population size for generating synthetic datasets.
        - n_gen (int): Number of generations for generating synthetic datasets.
        - f1_score (float): F1 score of the actual dataset.
        - n1_score (float): N1 score of the actual dataset.
        - datasets (dict): Dictionary containing information about generated synthetic datasets.

    Methods:
        - prepare_analysis(X, y): Prepare synthetic datasets for analysis.
        - evaluate_competency(model, cv=5, scoring='roc_auc'): Evaluate the competency of a model across synthetic datasets using cross-validation.
        - visualise_results(): Visualize the results using a heatmap and scatter plot.

    Example Usage:
        analysis = CompetencyAnalysis(n_datasets=100, max_distance=0.4, pop_size=40, n_gen=10)
        analysis.prepare_analysis(X_train, y_train)
        analysis.evaluate_competency(model, cv=5, scoring='roc_auc')
        analysis.visualise_results()
    """

    def __init__(self, n_datasets=100, max_distance=0.4, pop_size=40, n_gen=10):
        self.n_datasets = n_datasets
        self.max_distance = max_distance
        self.pop_size = pop_size
        self.n_gen = n_gen

    def prepare_analysis(self, X, y):
        """
        Prepare synthetic datasets for analysis.

        Parameters:
            - X (array-like): Input features of the actual dataset.
            - y (array-like): True labels of the actual dataset.
        """

        # calculate F1 and N1 of dataset
        self.f1_score = F1(X, y)
        self.n1_score = N1(X, y)

        # generate points about actual dataset complexity
        generated_points = generate_points_around_x(self.f1_score, self.n1_score, self.n_datasets, self.max_distance)

        # generate synthetic dataset
        datasets = {}
        for i, point in tqdm(enumerate(generated_points), desc="Generating synthetic datasets", total=len(generated_points)):

            syboid = SyBoid(F1_Score=point[0], 
                        N1_Score=point[1], 
                        X=X, 
                        y=y, 
                        Mimic_Classes=True, 
                        Mimic_DataTypes=False,
                        Mimic_Dataset=False)

            syboid.Generate_Data(pop_size=self.pop_size,n_gen=self.n_gen)

            X_, y_ = syboid.return_best_dataset()

            datasets[i] = {'F1': F1(X_, y_),
                        'N1': N1(X_, y_),
                        'X': X_,
                        'y': y_}
        
        self.datasets = datasets
    
    def evaluate_competency(self, model, cv=5, scoring='roc_auc'):
        """
        Evaluate the competency of a model across synthetic datasets using cross-validation.

        Parameters:
            - model: Machine learning model.
            - cv (int, optional): Number of cross-validation folds (default is 5).
            - scoring (str, optional): Evaluation metric for cross-validation (default is 'roc_auc').
        """

        for v in self.datasets.values():
            try:
                # calculate cross_val_score for a given dataset
                v['score'] = np.mean(cross_val_score(model, v['X'], v['y'], cv=cv, scoring=scoring))
            except Exception as e:
                print(f"Error processing entry {v}: {e}")
                del v # Remove the entry from the datasets dictionary
    
    def visualise_results(ca):
        """
        Visualize the results using a heatmap and scatter plot.
        """

        # Extract F1, N1, and accuracy scores from the datasets
        f1_scores = [v['F1'] for v in ca.datasets.values()]
        n1_scores = [v['N1'] for v in ca.datasets.values()]
        accuracy_scores = [v['score'] for v in ca.datasets.values()]

        # Set up a grid for the heatmap
        f1_range = np.linspace(min(f1_scores), max(f1_scores), 100)
        n1_range = np.linspace(min(n1_scores), max(n1_scores), 100)
        f1_mesh, n1_mesh = np.meshgrid(f1_range, n1_range)

        # Use k-nearest neighbors regression to predict accuracy at each grid point
        knn_regressor = KNeighborsRegressor(n_neighbors=20, weights='distance')
        knn_regressor.fit(np.column_stack((f1_scores, n1_scores)), accuracy_scores)

        # Predict the accuracy at each grid point
        accuracy_grid = knn_regressor.predict(np.column_stack((f1_mesh.ravel(), n1_mesh.ravel())))

        # Reshape the predicted accuracy to the shape of the meshgrid
        accuracy_grid = accuracy_grid.reshape(f1_mesh.shape)

        # Create a larger figure
        plt.figure(figsize=(14, 8))

        # Create a heatmap using the predicted accuracy
        plt.imshow(accuracy_grid, cmap='viridis', extent=(min(f1_scores), max(f1_scores), min(n1_scores), max(n1_scores)))

        # Plot actual data points on top
        plt.scatter(f1_scores, n1_scores, c=accuracy_scores, s=40, edgecolors='k', linewidths=1, marker='o', label='Synthetic Datasets')
        plt.scatter(ca.f1_score, ca.n1_score, color='red', edgecolors='k', linewidths=1, marker='s', label='Original Specification')

        # Customize the appearance of the plot
        plt.colorbar(label='Weighted Classifier Score')
        plt.xlabel('F1 Score')
        plt.ylabel('N1 Score')
        plt.legend()
        plt.show()

    
class ConceptDriftAnalysis:
    """
    Class for analyzing concept drift using synthetic data.

    Parameters:
        - X (array-like): Input features of the original dataset.
        - y (array-like): True labels of the original dataset.
        - f1_target (float, optional): Target F1 score for generating synthetic data (default is 1).
        - n1_target (float, optional): Target N1 score for generating synthetic data (default is 1).
        - pop_size (int, optional): Population size for generating synthetic data (default is 40).
        - n_gen (int, optional): Number of generations for generating synthetic data (default is 10).

    Attributes:
        - pop_size (int): Population size for generating synthetic data.
        - n_gen (int): Number of generations for generating synthetic data.
        - f1_target (float): Target F1 score for generating synthetic data.
        - n1_target (float): Target N1 score for generating synthetic data.
        - X (array-like): Input features of the original dataset.
        - y (array-like): True labels of the original dataset.
        - X_ (array-like): Best synthetic data generated by SyBoid.
        - y_ (array-like): Corresponding labels for the best synthetic data.

    Methods:
        - prepare_analysis(): Prepare synthetic data for analysis using SyBoid.
        - evaluate_concept_drift(model, n_splits, scoring='accuracy'): Evaluate a model with synthetic data using k-fold cross-validation.

    Example Usage:
        analysis = ConceptDriftAnalysis(X_train, y_train, f1_target=0.8, n1_target=0.9, pop_size=50, n_gen=15)
        analysis.prepare_analysis()
        results = analysis.evaluate_concept_drift(model, n_splits=5, scoring='roc_auc')
    """

    def __init__(self, X, y, f1_target=1, n1_target=1, pop_size=40, n_gen=10):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.f1_target = f1_target
        self.n1_target = n1_target
        self.X = X
        self.y = y

        self.prepare_analysis()

    def prepare_analysis(self):
        """
        Prepare synthetic data for analysis using SyBoid.
        """

        # Generate synthetic data using SyBoid
        syboid = SyBoid(
            F1_Score=self.f1_target,
            N1_Score=self.n1_target,
            X=self.X,
            y=self.y,
            Mimic_Classes=True,
            Mimic_DataTypes=True,
            Mimic_Dataset=True,
        )
        
        syboid.Generate_Data(pop_size=self.pop_size, n_gen=self.n_gen)

        # Get the best synthetic dataset from SyBoid
        self.X_, self.y_ = syboid.return_best_dataset()

    def evaluate_concept_drift(self, model, n_splits, scoring='accuracy'):
        """
        Evaluate a model with synthetic data using k-fold cross-validation.

        Parameters:
        - model: Machine learning model
        - n_splits: Number of stratified folds for cross-validation
        - evaluation_metric: String representing the evaluation metric ('accuracy', 'roc_auc', 'precision', 'recall', 'f1'), default is 'accuracy'

        Returns:
        - results: Dictionary containing performance metrics for different synthetic percentages
        """

        # Create stratified k-fold object
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Split the synthetic data into folds
        splits = [test_index for _, test_index in skf.split(self.X_, self.y_)]

        results = {}

        for n_folds in range(n_splits + 1):
            
            f1_scores = []
            n1_scores = []
            performance = []

            if n_folds > 0:
                # Iterate over combinations of folds
                folds_included = itertools.combinations(range(n_splits), r=n_folds)

                for folds in folds_included:
                    idx = np.hstack([splits[f] for f in folds])

                    X = np.concatenate([self.X, self.X_[idx]])
                    y = np.concatenate([self.y, self.y_[idx]])

                    f1_scores.append(F1(X, y))
                    n1_scores.append(N1(X, y))

                    # Cross-validate the model
                    score = cross_val_score(model, X, y, cv=skf, scoring=scoring)
                    performance.append(np.mean(score))

            else:
                f1_scores.append(F1(self.X, self.y))
                n1_scores.append(N1(self.X, self.y))

                # Cross-validate the model with the original data
                score = cross_val_score(model, self.X, self.y, cv=skf, scoring=scoring)
                performance.append(np.mean(score))

            try:
                synthetic_percentage = self.X.shape[0] / (self.X.shape[0] + idx.shape[0])
            except NameError:
                synthetic_percentage = 0

            # Store results for different synthetic percentages
            results[synthetic_percentage] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'mean_n1': np.mean(n1_scores),
                'std_n1': np.std(n1_scores),
                'mean_score': np.mean(performance),
                'std_score': np.std(performance)
            }
            
        return results
    
def derive_abstention_threshold(X, y, uncertainty_estimator, scoring='roc_auc', target_score=0.8, n_splits=5, random_state=42):
    """
    Finds the mean optimal abstention threshold that achieves the target evaluation metric using cross-validation.

    Parameters:
    - X: Input features
    - y: True labels
    - uncertainty_estimator: An instance of UncertaintyEstimator
    - scoring: Evaluation metric to use for determining the optimal threshold (default is 'roc_auc')
    - target_score: Target score to achieve with the chosen evaluation metric (default is 0.8)
    - n_splits: Number of cross-validation folds (default is 5)
    - random_state: Random seed for reproducibility (default is 42)

    Returns:
    - mean_optimal_threshold: The mean optimal abstention threshold that achieves the target metric over the folds
    """

    # Define a dictionary of evaluation metrics
    evaluation_metrics = {
        'accuracy': {
            'function': accuracy_score,
            'requires_probabilities': False
        },
        'precision': {
            'function': precision_score,
            'requires_probabilities': False
        },
        'recall': {
            'function': recall_score,
            'requires_probabilities': False
        },
        'f1': {
            'function': f1_score,
            'requires_probabilities': False
        },
        'roc_auc': {
            'function': roc_auc_score,
            'requires_probabilities': True
        },
    }

    # Create StratifiedKFold object for cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mean_optimal_threshold = 0.0

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit uncertainty estimator on training data
        uncertainty_estimator.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred, y_prob, uncertainty = uncertainty_estimator.predict(X_val)

        # Iterate through threshold values from 99 to 0
        for threshold_ in range(99, -1, -1):
            threshold = threshold_ / 100.0

            # Check if all classes are present in the subset below the threshold
            if all([c in y_val[uncertainty < threshold] for c in np.unique(y_val)]):

                # Calculate the score based on the specified evaluation metric
                if evaluation_metrics[scoring]['requires_probabilities']:
                    score = evaluation_metrics[scoring]['function'](y_val[uncertainty < threshold], y_prob[uncertainty < threshold, 1])
                else:
                    score = evaluation_metrics[scoring]['function'](y_val[uncertainty < threshold], y_pred[uncertainty < threshold])

                # If the score meets or exceeds the target, update the mean threshold and break the loop
                if score >= target_score:
                    mean_optimal_threshold += threshold
                    break

    # Calculate the mean optimal threshold over all folds
    mean_optimal_threshold /= n_splits

    return mean_optimal_threshold

def calculate_shapley_value(X_train, f, x, j, M=100):
    """
    Calculate the Shapley value for a specific feature j using a random subset of other features.

    Parameters:
    - X_train (np.ndarray): Training dataset used to sample instances.
    - f: Uncertainty Explaination object
    - x (np.ndarray): Instance(s) for which the Shapley value is calculated.
    - j (int): Index of the feature for which the Shapley value is calculated.
    - M (int): Number of iterations to estimate the Shapley value (default is 100).

    Returns:
    - float: Shapley value for the specified feature.
    """
    # Check if x has the correct shape
    if x.shape[0] != 1:
        raise ValueError("Input instance x must have shape (1, n).")

    # Get the number of features in the instance x
    n_features = x.shape[1]
    
    # Initialize an empty list to store marginal contributions
    marginal_contributions = []
    
    # Create a list of feature indices, excluding the feature of interest (j)
    feature_idxs = list(range(n_features))
    feature_idxs.remove(j)

    # Perform M iterations to estimate Shapley value
    for _ in range(M):
        # Sample a random index to get a random instance from X_train
        random_idx = random.randint(0, len(X_train) - 1)
        z = X_train[random_idx]
        
        # Randomly select a subset of features for the positive side of the Shapley value
        x_idx = random.sample(feature_idxs, min(max(int(0.2 * n_features), random.choice(feature_idxs)), int(0.8 * n_features)))

        # Determine the complement set for the negative side of the Shapley value
        z_idx = [idx for idx in feature_idxs if idx not in x_idx]

        # Construct two new instances by modifying the features
        x_plus_j = np.array([x[0, i] if i in x_idx + [j] else z[i] for i in range(n_features)])
        x_minus_j = np.array([z[i] if i in z_idx + [j] else x[0, i] for i in range(n_features)])

        # Calculate the marginal contribution for the current iteration
        marginal_contribution = f.predict_(x_plus_j.reshape(1, -1))[0] - \
                                f.predict_(x_minus_j.reshape(1, -1))[0]
        
        # Append the marginal contribution to the list
        marginal_contributions.append(marginal_contribution)

    # Calculate the average Shapley value over all iterations
    phi_j_x = sum(marginal_contributions) / len(marginal_contributions)

    return phi_j_x

def explain_features(X_train, f, x, M=100):
    """
    Calculate Shapley values for all features for all instances in x.

    Parameters:
    - X_train (np.ndarray): Training dataset used to sample instances.
    - f: Uncertainty Explanation object.
    - x (np.ndarray): Instance(s) for which the Shapley values are calculated.
    - M (int): Number of iterations to estimate the Shapley values (default is 100).

    Returns:
    - list of lists: Shapley values for all features for each instance in x.
    """
    shapley_values_for_all_instances = []

    # Iterate over instances in x
    for instance in tqdm(x, desc="Explaining Instances", unit="instance"):
        shapley_values_for_instance = []

        # Get the number of features in the instance
        n_features = len(instance)

        # Calculate Shapley value for each feature
        for j in range(n_features):
            shapley_value = calculate_shapley_value(X_train, f, instance.reshape(1,-1), j, M)
            shapley_values_for_instance.append(shapley_value)

        shapley_values_for_all_instances.append(shapley_values_for_instance)

    return np.array(shapley_values_for_all_instances)

class MetaUncertaintyEstimator:
    """
    MetaUncertaintyEstimator class for estimating uncertainty using heuristics and fuzzy clustering.

    Parameters:
    - f: A fitted uncertainty explanation object with weights and cluster centers.

    Methods:
    - predict_(x): Predicts the misclassifications risk based on the provided instance and the meta-model's properties.
    """
    def __init__(self, f):
        self.f = f

    def predict_(self, x):
        """
        Generate predictions and assess misclassification risk for input data.

        Parameters:
        X (array-like): Input data for prediction.

        Returns:
        misclassifications_risk (array-like): Risk assessment of misclassifications for the input data.
        """
        ## Weight heuristics by the determined weights.
        x = x * self.f.weights

        # Predict cluster memberships and other information using the fuzzy clustering system.
        # This step is necessary for risk assessment.
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                np.transpose(x), self.f.cntr, 2, error=0.005, maxiter=100, init=None)

        # Calculate misclassifications risk based on cluster memberships and cluster-specific heuristics.
        misclassifications_risk = self.f._weighted_average(u, self.f.Cluster_IH_mean)
        return misclassifications_risk
    
def explain_meta_features(f, X_train, y_train, x, M=100):
    """
    Explain the meta-features of an instance using Shapley values.

    Parameters:
    - f: Uncertainty Explanation object.
    - X_train (np.ndarray): Training dataset.
    - y_train (np.ndarray): Target values corresponding to the training dataset.
    - x (np.ndarray): Instance(s) for which the meta-feature Shapley values are calculated.
    - M (int): Number of iterations to estimate the Shapley values (default is 100).

    Returns:
    - list: Shapley values for the meta-features.
    """
    # Initialize HeuristicsCalculator and fit it to the training data
    hc = HeursiticsCalculator()
    hc.fit(X_train, y_train)

    # Calculate heuristics for the given instance(s)
    heuristics = hc.calculate(x)

    # Initialize MetaUncertaintyEstimator using the provided Uncertainty Estimator
    mue = MetaUncertaintyEstimator(f)

    # Explain the meta-features contribution to the degree of uncertainty using the explain_features function
    shap_values = explain_features(f.heuristics, mue, heuristics, M=M)

    return heuristics, np.array(shap_values)