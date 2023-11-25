from AITIA.heuristics import KNeighbors, DisjunctSize, DisjunctClass, ClassLikelihood
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from AITIA.syboid import SyBoid
from AITIA.complexity_measures import F1, N1

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

    def __init__(self, n_datasets=100, max_distance=0.4, pop_size=40, n_gen=10):
        self.n_datasets = n_datasets
        self.max_distance = max_distance
        self.pop_size = pop_size
        self.n_gen = n_gen

    def prepare_analysis(self, X, y):
        # calculate F1 and N1 of dataset
        self.f1_score = F1(X, y)
        self.n1_score = N1(X, y)

        # generate points about actual dataset complexity
        generated_points = self._generate_points_around_x()

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
        
        for v in self.datasets.values():
            try:
                # calculate cross_val_score for a given dataset
                v['score'] = np.mean(cross_val_score(model, v['X'], v['y'], cv=cv, scoring=scoring))
            except Exception as e:
                print(f"Error processing entry {v}: {e}")
                del v # Remove the entry from the datasets dictionary
    
    def visualise_results(ca):
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

    def _generate_points_around_x(self):
        """
        Generate a series of points around the central point (F1, N1).

        Parameters:
        - F1: Float representing the central point along the F1 axis.
        - N1: Float representing the central point along the N1 axis.
        - num_points: Number of points to generate.
        - max_distance: Maximum distance from the central point.

        Returns:
        - Array of generated points.
        """
        mean = np.array([self.f1_score, self.n1_score])
        cov_matrix = np.eye(2)  # Identity matrix as the covariance matrix

        # Generate points using a 2D Gaussian distribution
        points = np.random.multivariate_normal(mean, cov_matrix, self.n_datasets)

        # Scale the points based on the maximum distance
        scaled_points = (self.max_distance * (points - np.mean(points, axis=0)) / np.max(np.abs(points - np.mean(points, axis=0)), axis=0))+mean

        scaled_points = np.clip(scaled_points, 0, 1)

        return scaled_points
    
    
class ConceptDriftAnalysis:
    
    def __init__(self, X, y, f1_target=1, n1_target=1, pop_size=40, n_gen=10):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.f1_target = f1_target
        self.n1_target = n1_target
        self.X = X
        self.y = y

        self.prepare_analysis()


    def prepare_analysis(self):

        # drifted dataset
        syboid = SyBoid(F1_Score=self.f1_target, 
                             N1_Score=self.n1_target, 
                             X=self.X, 
                             y=self.y, 
                             Mimic_Classes=True, 
                             Mimic_DataTypes=True,
                             Mimic_Dataset=True)
        
        syboid.Generate_Data(pop_size=self.pop_size, n_gen=self.n_gen)

        self.X_, self.y_ = syboid.return_best_dataset()


    def evaluate_model_with_synthetic_data(self, model, n_splits, scoring='accuracy'):
        """
        Evaluate a model with synthetic data using k-fold cross-validation.

        Parameters:
        - model: Machine learning model
        - n_splits: Number of stratified folds for cross-validation
        - evaluation_metric: String representing the evaluation metric ('accuracy', 'roc_auc', 'precision', 'recall', 'f1'), default is 'accuracy'

        Returns:
        - performance_scores: List of performance scores for each n
        """

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        splits = []
        for _, test_index in skf.split(self.X_, self.y_):
            splits.append(test_index)

        results = {}

        for n_folds in range(n_splits + 1):
            
            f1_scores = []
            n1_scores = []
            performance = []

            if n_folds > 0:
                folds_included = itertools.combinations(range(n_splits), r=n_folds)

                for folds in folds_included:
                    idx = np.hstack([splits[f] for f in folds])

                    X = np.concatenate([self.X, self.X_[idx]])
                    y = np.concatenate([self.y, self.y_[idx]])    

                    f1_scores.append(F1(X, y))
                    n1_scores.append(N1(X, y))

                    score = cross_val_score(model, X, y, cv = skf, scoring = scoring)
                    performance.append(np.mean(score))

            else:
                f1_scores.append(F1(self.X, self.y))
                n1_scores.append(N1(self.X, self.y))

                score = cross_val_score(model, self.X, self.y, cv = skf, scoring = scoring)
                performance.append(np.mean(score))

            try:
                synthetic_percentage = self.X.shape[0]/(self.X.shape[0]+idx.shape[0])
            except NameError:
                synthetic_percentage = 0

            results[synthetic_percentage] = {
                                            'mean_f1': np.mean(f1_scores),
                                            'std_f1': np.std(f1_scores),
                                            'mean_n1': np.mean(n1_scores),
                                            'std_n1': np.std(n1_scores),
                                            'mean_score': np.mean(performance),
                                            'std_score': np.std(performance)
                                            }
            
        return results