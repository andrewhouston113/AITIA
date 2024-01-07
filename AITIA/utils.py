import numpy as np
from sklearn.model_selection import StratifiedKFold

def extract_decision_tree(tree, X, y, node=0, depth=0):
        """
        Recursively extract the structure of a decision tree and return it as a dictionary.

        Parameters:
        tree (sklearn.tree._tree.Tree): The decision tree object to extract the structure from.
        X (array-like): The dataset the decision tree was fitted to.
        y (array-like): The target labels corresponding to the dataset.
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
            instances_at_leaf = sum([1 for i in range(len(X)) if tree.apply(X[i].reshape(1, -1))[0] == leaf_node])
            tree_info["depth"] = depth
            tree_info["instances_count"] = instances_at_leaf
            
            # Initialize a dictionary to store the count of instances for each unique y value in the leaf
            instances_by_class = {}
            unique_classes = set(y)  # Get all unique classes from y
            for class_label in unique_classes:
                instances_by_class[class_label] = 0  # Initialize counts to 0

            for i in range(len(X)):
                if tree.apply(X[i].reshape(1, -1))[0] == leaf_node:
                    y_value = y[i]
                    instances_by_class[y_value] += 1

            tree_info["instances_by_class"] = instances_by_class

        else:
            # If it's not a leaf, store the decision condition
            feature_name = tree.feature[node]
            threshold = tree.threshold[node]
            left_node = tree.children_left[node]
            right_node = tree.children_right[node]

            tree_info["depth"] = depth
            tree_info["decision"] = {"feature_name": feature_name, "threshold": threshold}

            # Recursively traverse left and right subtrees
            tree_info["left"] = extract_decision_tree(tree, X, y, left_node, depth + 1)
            tree_info["right"] = extract_decision_tree(tree, X, y, right_node, depth + 1)

        return tree_info

def diversity_degree(data, n_classes):
    """
    Calculate the diversity degree of a dataset.

    Args:
    data (array-like): The input data for which diversity is calculated.
    n_classes (int): The expected number of unique classes or categories in the data.

    Returns:
    float: The diversity degree score, ranging from 0 to 1. 
           0 indicates perfect diversity, 1 means no diversity.
    """
    # Calculate the total number of data points
    N = len(data)
    
    # Calculate the count of each unique element in the data
    _, n = np.unique(data, return_counts=True)
    
    # Ensure 'n' has at least 'n_classes' elements to avoid division by zero issues
    n = np.append(n, [0] * (n_classes - len(n)))
    
    # Calculate the Likelihood Ratio Imbalance Degree (LRID)
    LRID = np.nansum([(nc * np.log(N / (n_classes * nc)) if nc > 0 else 0) for nc in n])
    
    # Initialize an array for the null diversity
    null_diversity = np.zeros(n_classes)
    null_diversity[0] = N
    
    # Calculate the Imbalance Degree summation for null diversity
    null_diversity_LRID = np.nansum([(nc * np.log(N / (n_classes * nc)) if nc > 0 else 0) for nc in null_diversity])
    
    # Calculate and return the diversity degree score
    if LRID == 0:
        # If LRID is exactly 0, return 1 (perfect diversity)
        return 1
    else:
        # If LRID is not 0, compute diversity degree between 0 and 1
        return 1 - (LRID / null_diversity_LRID)

def generate_points_around_x(f1_score, n1_score, n_datasets, max_distance):
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
    mean = np.array([f1_score, n1_score])
    cov_matrix = np.eye(2)  # Identity matrix as the covariance matrix

    # Generate points using a 2D Gaussian distribution
    points = np.random.multivariate_normal(mean, cov_matrix, n_datasets)

    # Scale the points based on the maximum distance
    scaled_points = (max_distance * (points - np.mean(points, axis=0)) / np.max(np.abs(points - np.mean(points, axis=0)), axis=0))+mean

    scaled_points = np.clip(scaled_points, 0, 1)

    return scaled_points
