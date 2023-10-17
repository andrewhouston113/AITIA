from sklearn.tree import DecisionTreeClassifier
from AITIA.utils import extract_decision_tree
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

        Returns:
        int: The size (number of instances) of the leaf node that corresponds to the given instance.
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
            print(node, instance_class)
            if instance_class in node["instances_by_class"]:
                return node["instances_by_class"][instance_class]/node["instances_count"]
            else:
                return 0