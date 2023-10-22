import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial import distance
from scipy.stats import norm
from itertools import combinations
from AITIA.utils import extract_decision_tree, diversity_degree

class DisjunctSize:
    """
    DisjunctSize is a class for calculating normalized disjunct sizes for new instances based on a fitted decision tree.

    Attributes:
    X (array-like or None): Placeholder for the dataset.
    decision_tree (dict or None): Placeholder for the extracted decision tree structure.
    largest_leaf (int): A variable to store the size of the largest leaf in the decision tree.

    Methods:
    - fit(X, y): Fit a DecisionTreeClassifier to the provided dataset and extract its structure.
    - calculate(X): Calculate normalized disjunct sizes for new instances based on the fitted decision tree.
    - _find_largest_instances_count(dict): Recursively finds the largest value associated with the key 'instances_count' in a nested dictionary.
    - _get_leaf_size(node, instance): Helper method to determine the leaf size for a given instance.

    Example Usage:
    >>> disjunct_size = DisjunctSize()
    >>> disjunct_size.fit(X_train, y_train)
    >>> sizes = disjunct_size.calculate(X_test)
    """

    def __init__(self):
        self.decision_tree = None
        self.largest_leaf = 0

    def fit(self, X, y):
        """
        Fit a DecisionTreeClassifier to the provided dataset and extract its structure.
        
        Parameters:
        X (array-like): The dataset to fit the decision tree to.
        y (array-like): The target labels corresponding to the dataset.

        Returns:
        None
        """
        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Check if y is a supported data type
        if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError("y must be a NumPy array, pandas Series, or pandas DataFrame.")

        # Convert X and y to NumPy arrays if they are DataFrames or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Change type of the dataset X to a float32 array
        X = X.astype(np.float32)
        
        # Create and train a DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        # Extract and store the decision tree in a dictionary format
        self.decision_tree = extract_decision_tree(clf.tree_, X, y, node=0, depth=0)
        self.largest_leaf = self._find_largest_instances_count(self.decision_tree)

    def calculate(self, X):
        """
        Calculate normalized disjunct sizes for a list of new instances based on the previously fitted decision tree.

        Parameters:
        X (array-like): List of new instances for which disjunct sizes need to be calculated.

        Returns:
        list: A list of normalized disjunct sizes for each input instance in X.
        """
        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Store the dataset X as a float32 array
        X = X.astype(np.float32)

        # Initialize a list to store the normalized disjunct sizes for new instances
        normalised_disjunct_size = []
        
        for instance in X:
            # Calculate the disjunct size for the instance and normalize it
            disjunct_size = self._get_leaf_size(self.decision_tree, instance)
            normalised_disjunct_size.append(disjunct_size / self.largest_leaf)

        return normalised_disjunct_size
    
    def _find_largest_instances_count(self, dictionary):
        """
        Recursively finds the largest value associated with the key 'instances_count' in a nested dictionary.

        Parameters:
        dictionary (dict): The input dictionary to search for 'instances_count' values.

        Returns:
        int or None: The largest 'instances_count' value found in the dictionary or its sub-dictionaries. Returns None if no 'instances_count' values are found.
        """
        if isinstance(dictionary, dict):
            instances_counts = [v for k, v in dictionary.items() if k == 'instances_count']
            sub_instances_counts = [self._find_largest_instances_count(v) for v in dictionary.values() if isinstance(v, dict)]
            all_counts = instances_counts + sub_instances_counts
            if all_counts:
                return max(all_counts)
            else:
                return 0
        return 0

    def _get_leaf_size(self, node, instance):
        """
        Recursively determine the size of a leaf node for a given instance in the decision tree structure.

        Parameters:
        node (dict): The node within the decision tree structure.
        instance (array-like): The instance for which the leaf size is calculated.

        Returns:
        int: The size (number of instances) of the leaf node that corresponds to the given instance.
        """
        # Recursive function to determine the leaf size for a given instance
        if "decision" in node:
            feature_name = node["decision"]["feature_name"]
            threshold = node["decision"]["threshold"]
            if instance[feature_name] <= threshold:
                return self._get_leaf_size(node["left"], instance)
            else:
                return self._get_leaf_size(node["right"], instance)
        else:
            # If the node is a leaf, return its instance count
            return node["instances_count"]
        

class DisjunctClass:
    """
    DisjunctClass is a class for calculating percentage of instances in the same leaf node that share the same class as a new instances based on a fitted decision tree
      and the diversity of instances in the same leaf node as a new instances.

    Attributes:
    decision_tree (dict or None): Placeholder for the extracted decision tree structure.

    Methods:
    - fit(X, y): Fit a DecisionTreeClassifier to the provided dataset and extract its structure.
    - calculate_percentage(X, y): Calculate disjunct class percentages for new instances based on the fitted decision tree.
    - calculate_diversity(X): Calculate disjunct class diversity for new instances based on the fitted decision tree.
    - _get_leaf_percentage(node, instance, instance_class): Helper method to determine the percentage of instances in the same leaf node that share the same class as a new instance.
    - _get_leaf_diversity(node, instance): Helper method to determine the diversity of instances in the same leaf node as a new instance.

    Example Usage:
    >>> disjunct_class = DisjunctSize()
    >>> disjunct_class.fit(X_train, y_train, max_depth=4)
    >>> percentage = disjunct_class.calculate_percentage(X_test, y_test)
    >>> diversity = disjunct_size.calculate_diversity(X_test)
    """

    def __init__(self, max_depth=4, balanced=False):
        self.decision_tree = None
        self.n_classes = None
        self.max_depth = max_depth
        self.balanced = balanced

    def fit(self, X, y):
        """
        Fit a DecisionTreeClassifier to the provided dataset and extract its structure.
        
        Parameters:
        X (array-like): The dataset to fit the decision tree to.
        y (array-like): The target labels corresponding to the dataset.
        max_depth (int): The maximum depth the decision tree can fit to
        balanced (bool): Whether to balance the class_weights of when fitting the decision tree.

        Returns:
        None
        """
        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Check if y is a supported data type
        if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError("y must be a NumPy array, pandas Series, or pandas DataFrame.")

        # Convert X and y to NumPy arrays if they are DataFrames or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        #Calculate and store the number of unique classes
        self.n_classes = len(np.unique(y))
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Change type of the dataset X to a float32 array
        X = X.astype(np.float32)
        
        # Create and train a DecisionTreeClassifier balancing the class weights if specified
        if self.balanced:
            clf = DecisionTreeClassifier(max_depth=self.max_depth, class_weight='balanced')
        else:
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(X, y)

        # Extract and store the decision tree in a dictionary format
        self.decision_tree = extract_decision_tree(clf.tree_, X, y, node=0, depth=0)

    def calculate_percentage(self, X, y):
        """
        Calculate disjunct class percentages for a list of new instances based on the previously fitted decision tree.

        Parameters:
        X (array-like): List of new instances for which disjunct class percentages need to be calculated.
        y (array-like): The target labels corresponding to the dataset.

        Returns:
        list: A list of disjunct class percentages for each input instance in X.
        """
        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Store the dataset X as a float32 array
        X = X.astype(np.float32)

        # Initialize a list to store the normalized disjunct sizes for new instances
        disjunct_class_percentages = []
        
        for instance, instance_class in zip(X, y):
            # Calculate the disjunct class percentages for the instances
            disjunct_class_percentages.append(self._get_leaf_percentage(self.decision_tree, instance, instance_class))

        return disjunct_class_percentages
    
    def calculate_diversity(self, X):
        """
        Calculate disjunct class diversity for a list of new instances based on the previously fitted decision tree.

        Parameters:
        X (array-like): List of new instances for which disjunct class diversity need to be calculated.

        Returns:
        list: A list of disjunct class diversity for each input instance in X.
        """
        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Store the dataset X as a float32 array
        X = X.astype(np.float32)

        # Initialize a list to store the normalized disjunct sizes for new instances
        disjunct_class_diversities = []
        
        for instance in X:
            # Calculate the disjunct class percentages for the instances
            disjunct_class_diversities.append(self._get_leaf_diversity(self.decision_tree, instance))

        return disjunct_class_diversities
    
    def _get_leaf_percentage(self, node, instance, instance_class):
        """
        Recursively determine the percentages of instances in a leaf node with the same class label of a given instance in the decision tree structure.

        Parameters:
        node (dict): The node within the decision tree structure.
        instance (array-like): The instance for which the leaf size is calculated.
        instance_class (any): The target label corresponding to the instance.

        Returns:
        float: The percentage of the leaf node that corresponds to the given instance, with the same class label of the given instance.
        """
        # Recursive function to determine the leaf size for a given instance
        if "decision" in node:
            feature_name = node["decision"]["feature_name"]
            threshold = node["decision"]["threshold"]
            if instance[feature_name] <= threshold:
                return self._get_leaf_percentage(node["left"], instance, instance_class)
            else:
                return self._get_leaf_percentage(node["right"], instance, instance_class)
        else:
            # If the node is a leaf, return its disjunct class percentage
            return node["instances_by_class"][instance_class]/node["instances_count"]

    
    def _get_leaf_diversity(self, node, instance):
        """
        Recursively determine the diversity of instances in a leaf node a given instance in the decision tree structure.

        Parameters:
        node (dict): The node within the decision tree structure.
        instance (array-like): The instance for which the leaf size is calculated.

        Returns:
        float: The diversity of the leaf node that corresponds to the given instance, with the same class label of the given instance.
        """
        # Recursive function to determine the leaf size for a given instance
        if "decision" in node:
            feature_name = node["decision"]["feature_name"]
            threshold = node["decision"]["threshold"]
            if instance[feature_name] <= threshold:
                return self._get_leaf_diversity(node["left"], instance)
            else:
                return self._get_leaf_diversity(node["right"], instance)
        else:
            # If the node is a leaf, return its diversity
            node_labels = [x for x, count in enumerate(node["instances_by_class"].values()) for _ in range(count)]
            return diversity_degree(node_labels, self.n_classes)
            

class KNeighbors:
    """
    KNeighbors is a class for calculating the disagreeing neighbors percentage and diverse neighbors score for a set of instances using k-Nearest Neighbors.
    
    Attributes:
    y: NumPy array, labels for training data.
    n_neighbors: int, the number of neighbors to consider in k-Nearest Neighbors.

    Methods:
    - fit(self, X, y, n_neighbors=5): Fits a k-Nearest Neighbors classifier to the training data.
    - calculate_disagreement(self, X, y): Calculates the disagreeing neighbors percentage for a set of instances.
    - calculate_diversity(self, X, y): Calculates the diverse neighbors score for a set of instances.
    - _get_disagreeing_neighbors_percentage(self, instance, instance_class): Calculates the disagreeing neighbors percentage for a single instance.
    - _get_diverse_neighbors_score(self, instance): Calculates the diverse neighbors score for a single instance.
    
    Example usage:
    >>> KNeigh = KNeighbors()
    >>> KNeigh.fit(X_train, y_train, n_neighbors=5)
    >>> kdn_score = KNeigh.calculate_disagreement(X_test, y_test)
    >>> kdivn_score = KNeigh.calculate_diversity(X_test)
    """
    def __init__(self, n_neighbors):
        self.y = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit a KNeighborsClassifier to the provided dataset.
        
        Parameters:
        X (array-like): The dataset to fit the nearest neighbors classifier to.
        y (array-like): The target labels corresponding to the dataset.
        n_neighbors (int): The number of neighbours the classifier considers when determining the nearest neighbours.

        Returns:
        None
        """

        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Check if y is a supported data type
        if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError("y must be a NumPy array, pandas Series, or pandas DataFrame.")

        # Convert X and y to NumPy arrays if they are DataFrames or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        # Store y for future use
        self.y = y    

        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Fit a NearestNeighbours algorithm to X
        nn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.nearest_neighbors = nn.fit(X, y)

    def calculate_disagreement(self, X, y):
        """
        Calculate the disagreeing neighbors percentages for a list of new instances based on the previously fitted nearest neighbours classifier.

        Parameters:
        X (array-like): List of new instances for which disagreeing neighbors percentages need to be calculated.
        y (array-like): The target labels corresponding to the dataset.

        Returns:
        list: A list of disagreeing neighbors percentages for each input instance in X.
        """

        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize a list to store the disagreeing neighbors scores for new instances
        disagreeing_neighbors = []
        
        for instance, instance_class in zip(X, y):
            # Calculate the disagreeing neighbors scores for the instances
            disagreeing_neighbors.append(self._get_disagreeing_neighbors_percentage(instance.reshape(1, -1), instance_class))

        return disagreeing_neighbors
    
    def calculate_diversity(self, X):
        """
        Calculate the diverse neighbors score for a list of new instances based on the previously fitted nearest neighbours classifier.

        Parameters:
        X (array-like): List of new instances for which diverse neighbors score need to be calculated.
        y (array-like): The target labels corresponding to the dataset.

        Returns:
        list: A list of diverse neighbors score for each input instance in X.
        """

        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize a list to store the diverse neighbors scores for new instances
        diverse_neighbors = []
        
        for instance in X:
            # Calculate the diverse neighbors scores for the instances
            diverse_neighbors.append(self._get_diverse_neighbors_score(instance.reshape(1, -1)))

        return diverse_neighbors
    
    def _get_disagreeing_neighbors_percentage(self, instance, instance_class):
        """
        Calculate the disagreeing neighbors percentage for a single instance.

        Parameters:
        instance (array-like): The instance for which the disagreeing neighbors percentage is calculated.
        instance_class (any): The target label corresponding to the instance.

        Returns:
        float: The percentage of neighbors whose class is not the same as 'instance_class'.
        """

        # Find the indices of the k-nearest neighbors of 'instance'
        neighbors_idx = self.nearest_neighbors.kneighbors(instance, return_distance=False)
        
        # Compare the class labels of neighbors with the true class label of 'instance'
        nn_classes = [1 if self.y[idx] != instance_class else 0 for idx in neighbors_idx[0]]
        
        # Calculate the disagreeing neighbors percentage
        percentage = sum(nn_classes) / len(nn_classes)
        
        return percentage
    
    def _get_diverse_neighbors_score(self, instance):
        """
        Calculate the diverse neighbors score for a single instance.

        Parameters:
        instance (array-like): The instance for which the diverse neighbors score is calculated.

        Returns:
        float: The diverse neighbors score for the instance.
        """

        # Find the indices of the k-nearest neighbors of 'instance'
        neighbors_idx = self.nearest_neighbors.kneighbors(instance, return_distance=False)
        
        # Compare the class labels of neighbors with the true class label of 'instance'
        nn_classes = [self.y[idx] for idx in neighbors_idx[0]]
        
        # Calculate the diverse neighbors score
        diversity_score = diversity_degree(nn_classes, len(np.unique(self.y)))
        
        return diversity_score


class ClassLikelihood:
    """
    ClassLikelihood is a class for calculating class likelihood differences, evidence conflict, and evidence volume for new instances based on a fitted dataset.

    Attributes:
    data (dict or None): Placeholder for statistics computed from the training dataset.
    classes (array-like or None): Placeholder for unique class labels in the training dataset.

    Methods:
    - fit(X, y, categorical_idx=[]): Fit the class likelihood model to the provided dataset.
    - calculate_class_likelihood_difference(X, y): Calculate class likelihood differences for a list of new instances based on the training set statistics.
    - calculate_evidence_conflict(X, return_evidence_volume=False): Calculate evidence conflict for a list of new instances based on the training set statistics.
    - _class_stats(X, y, categorical_idx): Compute statistics for the training dataset, including class counts and feature statistics.
    - _class_likelihood(instance, target_class): Calculate the class likelihood for a given instance and a specific target class.
    - _class_likelihood_difference(instance, instance_class): Calculate the class likelihood difference for a given instance and its actual class.
    - _evidence_conflict(instance, return_evidence_volume=False): Calculate evidence conflict for a given instance.
    - _evidence_volume(instance, target_class): Calculate evidence volume for an instance belonging to a certain class.

    Example Usage:
    >>> clf = ClassLikelihood()
    >>> clf.fit(X_train, y_train)
    >>> likelihood_diff = clf.calculate_class_likelihood_difference(X_test, y_test)
    >>> evidence_conflict, evidence_volume = clf.calculate_evidence_conflict(X_test, return_evidence_volume=True)
    """

    def __init__(self):
        self.data = None
        self.classes = None

    def fit(self, X, y, categorical_idx=[]):
        """
        Fit the Likelihood class with the dataset.

        Parameters:
        X (array-like): A 2D array (MxN) representing instances with features.
        y (array-like): An array containing the class labels corresponding to each instance in X.

        Returns:
        None
        """

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of instances.")
        
        # Store the data description in a dictionary format
        self.data = self._class_stats(X, y, categorical_idx)
        self.classes = np.unique(y)

    def calculate_class_likelihood_difference(self, X, y):
        """
        Calculate the class likelihood difference for a list of new instances based on the training set statistics.

        Parameters:
        X (array-like): List of new instances for which class likelihood differences need to be calculated.
        y (array-like): The target labels corresponding to the dataset.

        Returns:
        list: A list of class likelihood differences for each input instance in X.
        """

        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize a list to store the likelihood differences for new instances
        likelihood_difference = []
        
        for instance, instance_class in zip(X, y):
            # Calculate the likelihood differences for the instances
            likelihood_difference.append(self._class_likelihood_difference(instance, instance_class))

        return likelihood_difference
    
    def calculate_evidence_conflict(self, X, return_evidence_volume = False):
        """
        Calculate evidence conflict for a list of new instances based on the training set statistics.

        Parameters:
        X (array-like): List of new instances for which evidence conflict need to be calculated.

        Returns:
        list: A list of evidence conflict for each input instance in X (if return_evidence_volume is True).
        """

        # Check if X is a supported data type
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("X must be a NumPy array, pandas DataFrame, or pandas Series.")

        # Convert X to a NumPy array if it's a DataFrame or Series
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize lists to store evidence conflict and evidence volume
        evidence_conflict = []
        evidence_volume = []

        for instance in X:
            # Calculate evidence conflict for the current instance
            ec, ev = self._evidence_conflict(instance)
            
            # Append evidence conflict and volume to the list
            evidence_conflict.append(ec)
            evidence_volume.append(ev)
        
        if return_evidence_volume:
            return evidence_conflict, evidence_volume
        else:
            return evidence_conflict
    
    def _class_stats(self, X, y, categorical_idx):
        """
        Calculate class-specific statistics for features in the input data.

        Parameters:
        X (numpy.ndarray): The input data with shape (n_samples, n_features) where n_samples is the number of samples,
            and n_features is the number of features.

        y (numpy.ndarray): The class labels corresponding to each sample in the input data 'X'. It should have shape (n_samples,).

        categorical_idx (list): A list of indices representing categorical features in the input data 'X'.

        Returns:
        dict: A dictionary containing class-specific statistics for each feature in the input data 'X'.
        """
        # Get the unique class labels from the 'y' array
        num_classes = np.unique(y)
        # Get the number of features in the input data 'X'
        num_features = X.shape[1]

        # Calculate the importance of each feature
        if categorical_idx:
            mutual_info = mutual_info_classif(X, y, discrete_features=categorical_idx).reshape(-1, 1).flatten()
        else:
            mutual_info = mutual_info_classif(X, y).reshape(-1, 1).flatten()

        # Offset mutual information by the lowest value (plus a negligible amount) to account for zeros and negative numbers
        feat_importance = mutual_info + (min(mutual_info) + 1)

        # Create an empty dictionary to store the class statistics
        class_dict = {}

        for feature in range(num_features):
            # Check if the current feature is in the list of categorical features
            if feature in categorical_idx:
                feature_dict = {'type': 'categorical', 'importance': feat_importance[feature], 'counts': {}}
                all_categories = np.unique(X[:, feature])
                
                for class_val in num_classes:
                    x_class = X[y == class_val, feature]
                    
                    # Calculate the counts of unique categories and store in a dictionary
                    unique, counts = np.unique(x_class, return_counts=True)
                    class_counts = dict(zip(unique, counts))
                    
                    # Ensure that every possible category is included, even if count is 0
                    for category in all_categories:
                        if category not in class_counts:
                            class_counts[category] = 0
                    
                    feature_dict['counts'][class_val] = class_counts
                    
                class_dict[feature] = feature_dict
            else:
                # If it's continuous, create a dictionary to store mean and standard deviation
                feature_dict = {'type': 'continuous', 'importance': feat_importance[feature], 'mean': {}, 'std': {}}
                
                for class_val in num_classes:
                    x_class = X[y == class_val, feature]
                    
                    # Calculate the mean and standard deviation and store in the dictionary
                    feature_dict['mean'][class_val] = np.mean(x_class)
                    feature_dict['std'][class_val] = np.std(x_class)
                
                class_dict[feature] = feature_dict

        return class_dict
    
    def _class_likelihood(self, instance, target_class):
        """
        Calculate the class likelihood for an instance belonging to a certain class.

        Parameters:
        instance (list or array): A 1-D array or list representing the instance for which to calculate the class likelihood.
        target_class (str): The class label for which to calculate the likelihood.

        Returns:
        float: The class likelihood for the given instance and class.
        """

        likelihood = 1.0
        for idx, feature in self.data.items():
            if feature['type'] == 'continuous':
                likelihood *= norm.cdf(instance[idx], loc=feature['mean'][target_class], scale=feature['std'][target_class])
            else:
                if instance[idx] not in self.data[idx]['counts'][target_class]:
                    raise ValueError(f"Category {instance[idx]} not found in training set for feature {idx}")
                class_total = 0
                for class_val in self.data[idx]['counts'].keys():
                    class_total += self.data[idx]['counts'][class_val][instance[idx]]
                likelihood *= self.data[idx]['counts'][target_class][instance[idx]]/class_total

        return likelihood
    
    def _class_likelihood_difference(self, instance, instance_class):
        """
        Calculate the class likelihood difference for an instance belonging to a certain class.

        Parameters:
        instance (list or array): A 1-D array or list representing the instance for which to calculate the class likelihood difference.
        instance_class (any): The target label corresponding to the instance.

        Returns:
        float: The class likelihood difference for the given instance and class.
        """
        # Calculate class likelihood for the instance's actual class
        likelihood_actual = self._class_likelihood(instance, instance_class)
        
        # Calculate class likelihood for all other classes
        likelihood_other = [self._class_likelihood(instance, class_label) for class_label in self.classes if class_label != instance_class]

        # Calculate the difference between the actual class likelihood and the maximum likelihood of other classes
        likelihood_difference = likelihood_actual - max(likelihood_other)
        
        return likelihood_difference

    def _evidence_conflict(self, instance):
        """
        Calculate evidence conflict for a given instance.

        Parameters:
        instance (list or array): A 1-D array or list representing the instance for which to calculate the evidence conflict.
        return_evidence_volume (bool, optional): Whether to return the evidence volume. Default is False.

        Returns:
        float: The evidence conflict score for the given instance.
        float (optional): The evidence volume score for the given instance if return_evidence_volume is True.
        """

        # Calculate evidence volume for each class
        evidence_volume_per_class = [self._evidence_volume(instance, class_label) for class_label in self.classes]

        # Find the maximum evidence volume among classes
        evidence_volume = max(evidence_volume_per_class)

        # Calculate the total evidence across all classes
        total_evidence = sum(evidence_volume_per_class)
        
        # Calculate the percentage share of evidence volume for each class
        evidence_share_per_class = [int((ev / total_evidence) * 100) for ev in evidence_volume_per_class]

        # Create a list of evidence labels based on the percentage share
        evidence_labels = [x for x, count in enumerate(evidence_share_per_class) for _ in range(count)]

        # Calculate evidence conflict using diversity degree
        evidence_conflict = diversity_degree(evidence_labels, len(np.unique(self.classes)))

        return evidence_conflict, evidence_volume

    def _evidence_volume(self, instance, target_class):
        """
        Calculate evidence volume for an instance belonging to a certain class.

        Parameters:
        instance (list or array): A 1-D array or list representing the instance for which to calculate the evidence volume.
        target_class (str): The class label for which to calculate evidence volume.

        Returns:
        list: The normalized evidence volume for the given instance and class for each feature.
        """

        # Initialize evidence
        evidence = 1

        # Iterate over features in the data
        for idx, feature in self.data.items():
            if feature['type'] == 'continuous':
                # Calculate evidence for continuous features
                evidence *= (norm.cdf(instance[idx], loc=feature['mean'][target_class], scale=feature['std'][target_class])) * feature['importance']
            else:
                # Calculate evidence for categorical features
                if instance[idx] not in self.data[idx]['counts'][target_class]:
                    raise ValueError(f"Category {instance[idx]} not found in the training set for feature {idx}")

                # Calculate evidence using category counts
                class_total = 0
                for class_val in self.data[idx]['counts'].keys():
                    class_total += self.data[idx]['counts'][class_val][instance[idx]]
                evidence *= (self.data[idx]['counts'][target_class][instance[idx]] / class_total) * feature['importance']

        return evidence
    

class RDOS:
    """
    RDOS (Relative Density Outlier Score) is a class for calculating the outlierness of an instance using relative density-based methods.
    
    Parameters:
    n_neighbors (int, optional): The number of neighbors to consider for density estimation. Default is 10.
    h (float, optional): The smoothing parameter for the Gaussian kernel used in density estimation. Default is 1.
    
    Attributes:
    n_neighbors (int): The number of neighbors to consider for density estimation.
    h (float): The smoothing parameter for the Gaussian kernel.
    X_train (numpy.ndarray): The training data used to fit the model.
    nearest_neighbors (NearestNeighbors): The NearestNeighbors model trained on the training data.
    training_neighborhood (dict): Neighborhood information for each data point in the training set.
    distance_matrix (numpy.ndarray): The distance matrix between data points in the training set.
    training_rdos (dict): Relative Density-based Outlier Scores (rdos) for data points in the training set.
    scaler (MinMaxScaler): A MinMaxScaler fitted to the training rdos values for scaling.
    
    Methods:
    fit(X): Fit the model with training data and perform necessary preprocessing. 
    calculate(X): Calculate the scaled relative density outlier score for a given instance or set of instances. 
    _get_training_neighborhood(nbrs, dists): Create the neighborhood dictionary for each data point in the training set based on nearest neighbors and distance information.
    _get_testing_neighborhood(X, nbrs): Create the neighborhood dictionary for each new instance in the set based on nearest neighbors and distance information from the training.
    _calculate_rdos(dictionary): Calculate Relative Density Outlier Scores (rdos) for each instance in the input dictionary.

    Example Usage:
    >>> rdos = RDOS()
    >>> rdos.fit(X_train)
    >>> outlierness = rdos.calculate(X_test)
    """
    def __init__(self, n_neighbors = 10, h = 1):
        
        if not isinstance(n_neighbors, int):
            raise TypeError("n_neighbors should be an integer")

        self.n_neighbors = n_neighbors
        self.h = h
        self.X_train = None
        self.nearest_neighbors = None
        self.training_neighbourhood = None
        self.distance_matrix = None
        self.training_rdos = None
        self.scaler = None

    def fit(self, X):
        """
        Fit the model with training data and perform necessary preprocessing.

        Parameters:
        X (array-like or DataFrame): The training data, where each row represents a data point.

        Raises:
        TypeError: If X is not a NumPy array or compatible data structure.
        ValueError: If X is not a 2D array.
        """
        # Check if X is a Pandas DataFrame, and convert it to a NumPy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Check if X is a NumPy array or a compatible data structure
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a NumPy array")

        # Set the training data for the instance
        self.X_train = X

        # Check if X is a 2D array
        if X.ndim != 2:
            raise ValueError("X should be a 2D array")
        
        # Set the training data for the instance
        self.X_train = X
        
        # Calculate the distance matrix between the data points in X
        self.distance_matrix = distance.cdist(X, X)
        
        # Initialize a NearestNeighbors object with a specified number of neighbors
        NN = NearestNeighbors(n_neighbors=self.n_neighbors)
        
        # Fit the NearestNeighbors model with the training data X
        self.nearest_neighbors = NN.fit(X)

        # Calculate the distances and indices of the k-nearest neighbors for each data point in X
        dists, nbrs = self.nearest_neighbors.kneighbors(X, n_neighbors=self.n_neighbors+1)
        
        # Exclude the first neighbor
        nbrs = nbrs[:, 1:]
        dists = dists[:, 1:]

        # Get the training neighborhood using the calculated neighbor indices and distances
        self.training_neighborhood = self._get_training_neighborhood(nbrs, dists)
        
        # Calculate the relative distances of the training neighborhood
        self.training_neighborhood = self._calculate_rdos(self.training_neighborhood)
        
        # Initialize a MinMaxScaler and fit it to the relative distances of the training neighborhood
        self.scaler = MinMaxScaler().fit(
            np.array([v['rdos'] for v in self.training_neighborhood.values()]).reshape(-1, 1)
        )

    def calculate(self, X, transform = True):
        """
        Calculate the scaled relative distances (rdos) for a given data point or set of data points.

        Parameters:
        X (array-like or DataFrame): The data point(s) for which to calculate rdos.
        transform (bool): Whether to MinMaxScale the scores. Default is True.

        Returns:
        list: A list of scaled relative distances (rdos) for the input data point(s).
        """
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Find the indices of the k-nearest neighbors of the input data X
        _, nbrs = self.nearest_neighbors.kneighbors(X, n_neighbors=self.n_neighbors)
        
        # Get the testing neighborhood using the neighbor indices
        test_neighborhood = self._get_testing_neighborhood(X, nbrs)
        
        # Calculate the relative density outlier score for the test neighborhood
        if transform:
            rdos = [self.scaler.transform(np.array(v['rdos']).reshape(-1, 1))[0][0] for v in self._calculate_rdos(test_neighborhood).values()]
        else:
            rdos = [v['rdos'] for v in self._calculate_rdos(test_neighborhood).values()]
        
        return rdos


    def _get_training_neighborhood(self, nbrs, dists):
        """
        Create the neighborhood dictionary for each data point in the training set based on nearest neighbors and distance information.

        Parameters:
        nbrs (numpy.ndarray): A 2D array where each row contains indices of k-nearest neighbors for a data point.
        dists (numpy.ndarray): A 2D array where each row contains distances to k-nearest neighbors.

        Returns:
        dict: A dictionary where each key represents a data point's index and the corresponding value is a dictionary
            containing information about its neighborhood.
        """
        # Initialize dictionaries to store neighborhood information
        shared_neighbors = {}
        reverse_neighbors = {}
        neighborhood = {}
        px = {}

        # Create a set for each data point's neighbors
        nbrs_set = [set(nbrs[i, :]) for i in range(nbrs.shape[0])]

        for idx in range(nbrs.shape[0]):
            # Calculate shared neighbors (common neighbors with other data points)
            shared_neighbors[idx] = [i for i, s in enumerate(nbrs_set) if (idx != i) and (s.intersection(nbrs_set[idx]))]
            # Calculate reverse neighbors (data points that have the current point as a neighbor)
            reverse_neighbors[idx] = [i for i, s in enumerate(nbrs_set) if (idx != i) and (idx in s)]
            # Combine neighbors, shared neighbors, and reverse neighbors to form the neighborhood
            neighborhood[idx] = list(nbrs_set[idx] | set(shared_neighbors[idx]) | set(reverse_neighbors[idx]))

            # Calculate the Gaussian kernel density estimation ('px') for the current data point
            Kgaussian = 1 / ((2 * np.pi) ** (self.X_train.shape[1] / 2)) * np.exp(-((self.distance_matrix[idx, neighborhood[idx]]) / (2 * self.h ** 2)))
            px[idx] = (1 / (len(neighborhood[idx]) + 1)) * np.sum((1 / (self.h ** self.X_train.shape[1])) * Kgaussian)

        # Create a dictionary to store neighborhood information for all data points
        neighbourhoods = {
            idx: {
                'nbrs': nbrs[idx, :],
                'cut-off_dist': dists[idx, -1],
                'shared_neighbors': shared_neighbors[idx],
                'reverse_neighbors': reverse_neighbors[idx],
                'neighborhood': neighborhood[idx],
                'px': px[idx]
            }
            for idx in range(nbrs.shape[0])
        }

        return neighbourhoods
    

    def _get_testing_neighborhood(self, X, nbrs):
        """
        Create the neighborhood dictionary for each new data point in the set based on nearest neighbors and distance information from the training.

        Parameters:
        nbrs (numpy.ndarray): A 2D array where each row contains indices of k-nearest neighbors for a data point.
        dists (numpy.ndarray): A 2D array where each row contains distances to k-nearest neighbors.

        Returns:
        dict: A dictionary where each key represents a data point's index and the corresponding value is a dictionary
            containing information about its neighborhood.
        """
        # Initialize dictionaries to store neighborhood information
        shared_neighbors = {}
        reverse_neighbors = {}
        neighborhood = {}
        px = {}

        # Create a set for each instance's neighbors and get the training instances neighbours and max_dist values
        nbrs_set = [set(nbrs[i, :]) for i in range(X.shape[0])]
        training_nbrs_set = [set(v['nbrs']) for v in self.training_neighborhood.values()]
        max_dist = [v['cut-off_dist'] for v in self.training_neighborhood.values()]

        for idx in range(nbrs.shape[0]):
            # Calculate shared neighbors (common neighbors with other data points)
            shared_neighbors[idx] = [i for i, s in enumerate(training_nbrs_set) if s.intersection(nbrs_set[idx])]
           
            # Calculate distance of the current instance from each instance in the training set
            dist_mat = distance.cdist(X[[idx],:], self.X_train)[0]
            
            # Calculate reverse neighbors (data points that have the current point as a neighbor)
            reverse_neighbors[idx] = [i for i, d in enumerate(zip(dist_mat, max_dist)) if d[0] < d[1]]
            neighborhood[idx] = list(nbrs_set[idx] | set(shared_neighbors[idx]) | set(reverse_neighbors[idx]))
            
            # Calculate the Gaussian kernel density estimation ('px') for the current data point
            Kgaussian = 1 / ((2 * np.pi) ** (self.X_train.shape[1] / 2)) * np.exp(-((dist_mat[neighborhood[idx]]) / (2 * self.h ** 2)))
            px[idx] = (1 / (len(neighborhood[idx]) + 1)) * np.sum((1 / (self.h ** self.X_train.shape[1])) * Kgaussian)

        # Create a dictionary to store neighborhood information for all data points
        neighbourhoods = {
            idx: {
                'nbrs': nbrs[idx, :],
                'shared_neighbors': shared_neighbors[idx],
                'reverse_neighbors': reverse_neighbors[idx],
                'neighborhood': neighborhood[idx],
                'px': px[idx]
            }
            for idx in range(X.shape[0])
        }

        return neighbourhoods


    def _calculate_rdos(self, dictionary):
        """
        Calculate Relative Distance-based Outlier Scores (rdos) for each data point in the input dictionary.

        Parameters:
        dictionary (dict): A dictionary containing neighborhood information for data points.

        Returns:
        dict: A modified version of the input dictionary with 'rdos' values calculated for each data point.
        """
        for v in dictionary.values():
            # Calculate rdos for each data point in the neighborhood
            rdos = sum([self.training_neighborhood[idx]['px'] for idx in v['neighborhood']]) / (len(v['neighborhood']) * v['px'])
            v['rdos'] = rdos

        return dictionary
    

class ClassLevelRDOS:
    """
    Class for calculating outlierness diversity based on Class-Level RDOS.

    Attributes:
    n_neighbors (int): Number of neighbors for RDOS calculation.
    h (float): A parameter 'h' for RDOS calculation.
    rdos_calculators (dict): A dictionary to store RDOS calculators for each class.

    Methods:
    - fit(X, y): Fit the ClassLevelRDOS model with training data.
    - calculate(X): Calculate outlierness diversity for input data.
    Example Usage:
    >>> cl_rdos = ClassLevelRDOS()
    >>> cl_rdos.fit(X_train, y_train)
    >>> class_level_rdos_scores = cl_rdos.calculate(X_test)
    """
    def __init__(self, n_neighbors=10, h=1):
        self.n_neighbors = n_neighbors
        self.h = h
        self.rdos_calculators = None

    def fit(self, X, y):
        """
        Fit the an RDOS class for each class in the training data.

        Parameters:
        - X (array-like or DataFrame): Training data with features.
        - y (array-like): Training data labels.
        """
        # Check if X is a Pandas DataFrame, and convert it to a NumPy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Check if X is a NumPy array or a compatible data structure
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a NumPy array")
        
        # Check if X is a 2D array
        if X.ndim != 2:
            raise ValueError("X should be a 2D array")
        
        # Initialize a dictionary to hold RDOS calculators for each class
        rdos_classes = {}

        for class_id, _ in enumerate(np.unique(y)):
            # Create an RDOS calculator for the class with specified parameters
            rdos_classes[class_id] = RDOS(n_neighbors=self.n_neighbors, h=self.h)
            # Fit the RDOS calculator with instances belonging to the current class
            rdos_classes[class_id].fit(X[y == class_id, :])

        # Store the RDOS calculators in the class instance
        self.rdos_calculators = rdos_classes

    def calculate(self, X):
        """
        Calculate outlierness diversity for input data.

        Parameters:
        - X (array-like or DataFrame): Input data with features.

        Returns:
        - outlierness_diversity (list): A list of outlierness diversity values for each data point in 'X'.
        """
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = []
        # Calculate RDOS scores for each class and store them in 'scores'
        for v in self.rdos_calculators.values():
            scores.append(v.calculate(X, transform=True))

        outlierness_diversity = []
        for idx in range(X.shape[0]): 
            # Calculate the total outlierness for the point
            total_outlierness = sum([score[idx] for score in scores]) 
            
            # Calculate the percentage share of outlierness for each class
            outlierness_share_per_class = [int((score[idx] / total_outlierness) * 100) for score in scores]

            # Create a list of outlierness labels based on the percentage share
            evidence_labels = [x for x, count in enumerate(outlierness_share_per_class) for _ in range(count)]

            # Calculate outlierness diversity using diversity degree
            outlierness_diversity.append(diversity_degree(evidence_labels, len(self.rdos_calculators.keys())))

        return outlierness_diversity
    

class HyperplaneDistance:
    """
    A class for fitting Support Vector Machine (SVM) models for OVO classification and calculating normalized
    minimum distances of instances from a decision boundary.

    Attributes:
    kernel (str): The kernel function used for SVM models (default is 'linear').
    class_weight (bool): Whether to balance the class weights for the SVM models (default is False).
    svms (dict): A dictionary containing trained SVM models for different class combinations.
    scaler (MinMaxScaler): A MinMaxScaler used to normalize the distances.

    Methods:
    - fit(X, y): Fit SVM models to the input data for binary classification and calculate normalized distances.
    - calculate(X): Calculate normalized distances of instances from SVM decision boundaries.
    - _get_distances(X): Helper method to calculate distances from decision boundaries.

    Example Usage:
    >>> clf = HyperplaneDistance(kernel='linear', balanced=True)
    >>> clf.fit(X_train, y_train)
    >>> distances = clf.calculate(X_test)

    """

    def __init__(self, kernel = 'linear', balanced = False):
        self.balanced = balanced
        self.kernel = kernel
        self.svms = None
        self.scaler = None

    def fit(self, X, y):
        """
        Fit SVM models to the input data for binary OVO classification.

        Parameters:
        X (array-like): The input feature data. It can be a Pandas DataFrame or a NumPy array.
        y (array-like): The target labels for binary classification.

        Raises:
        TypeError: If X is not a NumPy array or a compatible data structure.
        ValueError: If X is not a 2D array.
        """
        # Check if X is a Pandas DataFrame, and convert it to a NumPy array if needed.
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Check if X is a NumPy array or a compatible data structure.
        if not isinstance(X, np.ndarray):
            raise TypeError("X should be a NumPy array")

        # Check if X is a 2D array.
        if X.ndim != 2:
            raise ValueError("X should be a 2D array")

        # Initialize a dictionary to hold SVMs for each class.
        svm_combos = {}
        c = list(combinations(np.unique(y), 2))

        for class_id, c in enumerate(c):
            idx = np.where((y == c[0]) | (y == c[1]))[0]

            # Create an SVM model with the specified kernel and class weight.
            if self.balanced:
                svm_combos[class_id] = svm.SVC(kernel=self.kernel, class_weight='balanced')
            else:
                svm_combos[class_id] = svm.SVC(kernel=self.kernel)

            
            # Fit the SVM model to the instances corresponding to the current class combination.
            svm_combos[class_id].fit(X[idx, :], y[idx])

        self.svms = svm_combos  # Store the trained SVMs

        # Calculate distances from decision boundaries for training data.
        training_distances = self._get_distances(X)

        # Create and fit a MinMaxScaler to normalize the distances.
        self.scaler = MinMaxScaler().fit(training_distances.reshape(-1, 1))

    def calculate(self, X):
        """
        Calculate normalized distances of instances from SVM decision boundaries.

        Parameters:
        X (array-like): The input feature data for which distances are to be calculated. It should be a 2D array.

        Returns:
        distances (list): A list of normalized distances for each instance.
        """
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate distances from the decision boundaries of SVM models.
        distances = self._get_distances(X)

        # Calculate the decision function for the input data and normalize using the previously fitted MinMaxScaler and return them as a list.
        return list(self.scaler.transform(distances.reshape(-1, 1)).flatten())

    def _get_distances(self, X):
        """
        Calculate the minimum normalized distances of instances from SVM decision boundaries.

        Parameters:
        X (array-like): The input feature data for which distances are to be calculated.

        Returns:
        distances (array): An array of minimum normalized distances for each instance.
        """
        scores = []
        for v in self.svms.values():
            # Normalize the decision function by dividing it by the L2 norm of the SVM coefficients.
            scores.append(v.decision_function(X)/np.linalg.norm(v.coef_))

        # Stack the scores vertically and find the minimum value along each column (instance).
        return np.min(np.vstack(scores),axis=0)

class HeursiticsCalculator:

    def __init__(self, kneighbors_n=5, max_depth=4, balanced=False, rdos_neighbors=5):
        self.KDN = KNeighbors(n_neighbors=kneighbors_n)
        self.DS = DisjunctSize()
        self.DCD = DisjunctClass(max_depth=max_depth, balanced=balanced)
        self.HD = HyperplaneDistance(balanced=balanced)
        self.OL = RDOS(n_neighbors=rdos_neighbors)
        self.CL_OL = ClassLevelRDOS(n_neighbors=rdos_neighbors)
        self.EC = ClassLikelihood()

    def fit(self,X,y):
        self.KDN.fit(X,y)
        self.DS.fit(X,y)
        self.DCD.fit(X,y)
        self.HD.fit(X,y)
        self.OL.fit(X)
        self.CL_OL.fit(X,y)
        self.EC.fit(X,y)

    def calculate(self,X):
        KDN_score = self.KDN.calculate_diversity(X)
        DS_score = self.DS.calculate(X)
        DCD_score = self.DCD.calculate_diversity(X)
        HD_score = self.HD.calculate(X)
        OL_score = self.OL.calculate(X)
        CL_OL_score = self.CL_OL.calculate(X)
        EC_score = self.EC.calculate_evidence_conflict(X)

        return np.transpose(np.vstack((KDN_score, DS_score, DCD_score, HD_score, OL_score, CL_OL_score, EC_score)))