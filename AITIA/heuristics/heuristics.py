from sklearn.tree import DecisionTreeClassifier
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
        self.X = None
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

        # Store the dataset X as a float32 array
        self.X = X.astype(np.float32)
        
        # Create and train a DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        # Extract and store the decision tree in a dictionary format
        self.decision_tree = self.extract_decision_tree(clf.tree_, node=0, depth=0)

        # Remove X from saved variables
        self.X = None

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
        
    def extract_decision_tree(self, tree, node=0, depth=0):
        """
        Recursively extract the structure of a decision tree and return it as a dictionary.

        Parameters:
        tree (sklearn.tree._tree.Tree): The decision tree object to extract the structure from.
        node (int, optional): The current node in the tree. Defaults to the root node (0).
        depth (int, optional): The depth of the current node in the tree. Defaults to 0.

        Returns:
        dict: A dictionary representing the structure of the decision tree.
        """
        # Initialize an empty dictionary to store tree information
        tree_info = {}

        # Check if the current node is a leaf node
        if tree.children_left[node] == tree.children_right[node]:
            # If it's a leaf, store the depth and count of instances in the leaf
            leaf_node = node
            instances_at_leaf = sum([1 for i in range(len(self.X)) if tree.apply(self.X[i].reshape(1, -1))[0] == leaf_node])
            tree_info["depth"] = depth
            tree_info["instances_count"] = instances_at_leaf
            
            # Update the largest_leaf if the current leaf is the largest encountered so far
            if instances_at_leaf > self.largest_leaf:
                self.largest_leaf = instances_at_leaf
        else:
            # If it's not a leaf, store the decision condition
            feature_name = tree.feature[node]
            threshold = tree.threshold[node]
            left_node = tree.children_left[node]
            right_node = tree.children_right[node]

            tree_info["depth"] = depth
            tree_info["decision"] = {"feature_name": feature_name, "threshold": threshold}

            # Recursively traverse left and right subtrees
            tree_info["left"] = self.extract_decision_tree(tree, left_node, depth + 1)
            tree_info["right"] = self.extract_decision_tree(tree, right_node, depth + 1)

        return tree_info

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