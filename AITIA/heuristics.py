from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from AITIA.utils import extract_decision_tree
from scipy.stats import norm
import numpy as np
import pandas as pd

class DisjunctSize:
    """
    DisjunctSize is a class for calculating normalized disjunct sizes for new instances based on a fitted decision tree.

    Attributes:
    X (array-like or None): Placeholder for the dataset.
    decision_tree (dict or None): Placeholder for the extracted decision tree structure.
    largest_leaf (int): A variable to store the size of the largest leaf in the decision tree.

    Methods:
    - fit(X, y): Fit a DecisionTreeClassifier to the provided dataset and extract its structure.
    - calculate(X_new): Calculate normalized disjunct sizes for new instances based on the fitted decision tree.
    - extract_decision_tree(tree, node=0, depth=0): Helper method to extract the decision tree structure.
    - get_leaf_size(node, instance): Helper method to determine the leaf size for a given instance.

    Example Usage:
    >>> disjunct_size = DisjunctSize()
    >>> disjunct_size.fit(X_train, y_train)
    >>> sizes = disjunct_size.calculate(X_new)
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
            X = X.reshape(-1, 1)

        # Change type of the dataset X to a float32 array
        X = X.astype(np.float32)
        
        # Create and train a DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        # Extract and store the decision tree in a dictionary format
        self.decision_tree = extract_decision_tree(clf.tree_, X, y, node=0, depth=0)
        self.largest_leaf = self.find_largest_instances_count(self.decision_tree)

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
            X = X.reshape(-1, 1)

        # Store the dataset X as a float32 array
        X = X.astype(np.float32)

        # Initialize a list to store the normalized disjunct sizes for new instances
        normalised_disjunct_size = []
        
        for instance in X:
            # Calculate the disjunct size for the instance and normalize it
            disjunct_size = self.get_leaf_size(self.decision_tree, instance)
            normalised_disjunct_size.append(disjunct_size / self.largest_leaf)

        return normalised_disjunct_size
    
    def find_largest_instances_count(self, dictionary):
        """
        Recursively finds the largest value associated with the key 'instances_count' in a nested dictionary.

        Parameters:
        dictionary (dict): The input dictionary to search for 'instances_count' values.

        Returns:
        int or None: The largest 'instances_count' value found in the dictionary or its sub-dictionaries. Returns None if no 'instances_count' values are found.
        """
        if isinstance(dictionary, dict):
            instances_counts = [v for k, v in dictionary.items() if k == 'instances_count']
            sub_instances_counts = [self.find_largest_instances_count(v) for v in dictionary.values() if isinstance(v, dict)]
            all_counts = instances_counts + sub_instances_counts
            if all_counts:
                return max(all_counts)
            else:
                return 0
        return 0

    def get_leaf_size(self, node, instance):
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
                return self.get_leaf_size(node["left"], instance)
            else:
                return self.get_leaf_size(node["right"], instance)
        else:
            # If the node is a leaf, return its instance count
            return node["instances_count"]
        
class DisjunctClassPercentage:
    """
    DisjunctClassPercentage is a class for calculating percentage of instances in the same leaf node that share the same class as a new instances based on a fitted decision tree.

    Attributes:
    decision_tree (dict or None): Placeholder for the extracted decision tree structure.

    Methods:
    - fit(X, y): Fit a DecisionTreeClassifier to the provided dataset and extract its structure.
    - calculate(X_new): Calculate disjunct class percentages for new instances based on the fitted decision tree.
    - get_leaf_percentage(node, instance): Helper method to determine the percentage of instances in the same leaf node that share the same class as a new instance.

    Example Usage:
    >>> disjunct_size = DisjunctSize()
    >>> disjunct_size.fit(X_train, y_train)
    >>> sizes = disjunct_size.calculate(X_new)
    """

    def __init__(self):
        self.decision_tree = None

    def fit(self, X, y, max_depth=4, balanced=False):
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
        
        # Ensure X is 2-dimensional in shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Change type of the dataset X to a float32 array
        X = X.astype(np.float32)
        
        # Create and train a DecisionTreeClassifier balancing the class weights if specified
        if balanced:
            clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')
        else:
            clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X, y)

        # Extract and store the decision tree in a dictionary format
        self.decision_tree = extract_decision_tree(clf.tree_, X, y, node=0, depth=0)

    def calculate(self, X, y):
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
            X = X.reshape(-1, 1)

        # Store the dataset X as a float32 array
        X = X.astype(np.float32)

        # Initialize a list to store the normalized disjunct sizes for new instances
        disjunct_class_percentages = []
        
        for instance, instance_class in zip(X, y):
            # Calculate the disjunct class percentages for the instances
            disjunct_class_percentages.append(self.get_leaf_percentage(self.decision_tree, instance, instance_class))

        return disjunct_class_percentages
    
    def get_leaf_percentage(self, node, instance, instance_class):
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
                return self.get_leaf_percentage(node["left"], instance, instance_class)
            else:
                return self.get_leaf_percentage(node["right"], instance, instance_class)
        else:
            # If the node is a leaf, return its instance count
            if instance_class in node["instances_by_class"]:
                return node["instances_by_class"][instance_class]/node["instances_count"]
            else:
                return 0
            
class KDisagreeingNeighbors:
    """
    KDisagreeingNeighbors is a class for calculating the disagreeing neighbors percentage for a set of instances using k-Nearest Neighbors.
    
    Attributes:
    y: NumPy array, labels for training data.
    n_neighbors: int, the number of neighbors to consider in k-Nearest Neighbors.

    Methods:
    - fit(self, X, y, n_neighbors=5): Fits a k-Nearest Neighbors classifier to the training data.
    - calculate(self, X, y): Calculates the disagreeing neighbors percentage for a set of instances.
    - get_disagreeing_neighbors_percentage(self, instance, instance_class): Calculates the disagreeing neighbors percentage for a single instance.
    
    Example usage:
    >>> knn = KDisagreeingNeighbors()
    >>> knn.fit(train_X, train_y, n_neighbors=5)
    >>> results = knn.calculate(test_X, test_y)
    """
    def __init__(self):
        self.y = None
        self.n_neighbors = None

    def fit(self, X, y, n_neighbors=5):
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
            X = X.reshape(-1, 1)
        
        # Fit a NearestNeighbours algorithm to X
        nn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.nearest_neighbors = nn.fit(X, y)

    def calculate(self, X, y):
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
            X = X.reshape(-1, 1)

        # Initialize a list to store the disagreeing neighbors scores for new instances
        disagreeing_neighbors = []
        
        for instance, instance_class in zip(X, y):
            # Calculate the disagreeing neighbors scores for the instances
            disagreeing_neighbors.append(self.get_disagreeing_neighbors_percentage(instance.reshape(1, -1), instance_class))

        return disagreeing_neighbors
    
    def get_disagreeing_neighbors_percentage(self, instance, instance_class):
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


class ClassLikelihoodDifference:
    """
    ClassLikelihoodDifference is a class for calculating class likelihood differences for new instances based on a fitted dataset.

    Attributes:
    data (dict or None): Placeholder for statistics computed from the training dataset.
    classes (array-like or None): Placeholder for unique class labels in the training dataset.

    Methods:
    - fit(X, y, categorical_idx=[]): Fit the class likelihood model to the provided dataset.
    - calculate(X, y): Calculate class likelihood differences for a list of new instances based on the training set statistics.
    - class_stats(X, y, categorical_idx): Compute statistics for the training dataset, including class counts and feature statistics.
    - class_likelihood(instance, target_class): Calculate the class likelihood for a given instance and a specific target class.
    - class_likelihood_difference(instance, instance_class): Calculate the class likelihood difference for a given instance and its actual class.

    Example Usage:
    >>> clf = ClassLikelihoodDifference()
    >>> clf.fit(X_train, y_train)
    >>> likelihood_diff = clf.calculate(X_new, y_new)
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
        self.data = self.class_stats(X, y, categorical_idx)
        self.classes = np.unique(y)

    def calculate(self, X, y):
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
            X = X.reshape(-1, 1)

        # Initialize a list to store the likelihood_differences for new instances
        likelihood_difference = []
        
        for instance, instance_class in zip(X, y):
            # Calculate the disagreeing neighbors scores for the instances
            likelihood_difference.append(self.class_likelihood_difference(instance, instance_class))

        return likelihood_difference
    
    def class_stats(self, X, y, categorical_idx):
        # Get the unique class labels from the 'y' array
        num_classes = np.unique(y)
        # Get the number of features in the input data 'X'
        num_features = X.shape[1]

        # Create an empty dictionary to store the class statistics
        class_dict = {}

        for feature in range(num_features):
            # Check if the current feature is in the list of categorical features
            if feature in categorical_idx:
                feature_dict = {'type': 'categorical', 'counts': {}}
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
                feature_dict = {'type': 'continuous', 'mean': {}, 'std': {}}
                
                for class_val in num_classes:
                    x_class = X[y == class_val, feature]
                    
                    # Calculate the mean and standard deviation and store in the dictionary
                    feature_dict['mean'][class_val] = np.mean(x_class)
                    feature_dict['std'][class_val] = np.std(x_class)
                
                class_dict[feature] = feature_dict

        return class_dict
    
    def class_likelihood(self, instance, target_class):
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
    
    def class_likelihood_difference(self, instance, instance_class):
        """
        Calculate the class likelihood difference for an instance belonging to a certain class.

        Parameters:
        instance (list or array): A 1-D array or list representing the instance for which to calculate the class likelihood difference.
        instance_class (any): The target label corresponding to the instance.

        Returns:
        float: The class likelihood difference for the given instance and class.
        """
        # Calculate class likelihood for the instance's actual class
        likelihood_actual = self.class_likelihood(instance, instance_class)
        
        # Calculate class likelihood for all other classes
        likelihood_other = [self.class_likelihood(instance, class_label) for class_label in self.classes if class_label != instance_class]
        print(likelihood_actual, likelihood_other)
        # Calculate the difference between the actual class likelihood and the maximum likelihood of other classes
        likelihood_difference = likelihood_actual - max(likelihood_other)
        
        return likelihood_difference