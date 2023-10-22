import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gower
from scipy.sparse.csgraph import minimum_spanning_tree

def F1(X, y):
    """
    Calculate the F1 score, a measure the overlap between the values of the features and takes the 
    value of the largest discriminant ratio among all the available features.

    Parameters:
    X (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels associated with the data.

    Returns:
    float: The F1 score.
    """
    # Check if X is a Pandas DataFrame, and convert it to a NumPy array if needed.
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Check if X is a NumPy array or a compatible data structure.
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be a NumPy array")
    
    # Initialize empty lists to store numerator and denominator values
    num = []
    den = []
    
    for j in range(np.unique(y).shape[0]):
        # Calculate the numerator for label 'j' and append to 'num'
        num.append(_numerator(X, y, j))
        
        # Calculate the denominator for label 'j' and append to 'den'
        den.append(_denominator(X, y, j))

    # Calculate the intermediate result by summing the 'num' and 'den' arrays and handling NaN values
    aux = np.nansum(np.vstack(num), axis=0) / np.nansum(np.vstack(den), axis=0)
    
    # Find the maximum value in 'aux'
    aux = np.nanmax(aux)  # Originally commented out
    
    # Scale aux to [0 1]
    F1 = 1 / (aux + 1)
    
    return F1  # Was originally np.nanmean(F1)

def N1(X, y):
    """
    Calculate the N1 score, a measure of raction of borderline points (N1) computes the percentage of 
    vertexes incident to edges connecting examples of opposite classes in a Minimum Spanning Tree (MST).

    Parameters:
    X (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels associated with the data.

    Returns:
    float: The N1 score.
    """
    # Check if X is a Pandas DataFrame, and convert it to a NumPy array if needed.
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Check if X is a NumPy array or a compatible data structure.
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be a NumPy array")
    
    # Scale the data 'X' to the range [0, 1]
    X_ = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    # Calculate the Gower distance matrix for the scaled data
    dist_m = np.triu(gower.gower_matrix(X_), k=1)
    
    # Construct the minimum spanning tree (MST) from the distance matrix
    mst = minimum_spanning_tree(dist_m)
    
    # Extract the edges of the MST
    node_i, node_j = np.where(mst.toarray() > 0)
    
    # Identify edges with different classes
    diff_cls = y[node_i] != y[node_j]
    
    # Calculate the number of unique nodes with different classes
    aux = len(np.unique(np.concatenate([node_i[diff_cls], node_j[diff_cls]])))
    
    # Calculate the N1 score as 'aux' divided by the total number of samples in 'X'
    N1 = aux / X.shape[0]
    
    return N1

def _branch(X, y, j):
    """
    Extract the subset of data in 'X' where the labels match the given label 'j'.
    
    Parameters:
    X (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels associated with the data.
    j (int): The label for which the subset of data is to be extracted.
    
    Returns:
    numpy.ndarray: A subset of the input data 'X' where the labels match 'j'.
    """
    # Extract the subset of data in 'X' where the labels match the given label 'j'
    return X[np.where(y == j)[0], :]

def _numerator(X, y, j):
    """
    Calculate the numerator term for label 'j' based on the mean squared difference.

    Parameters:
    X (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels associated with the data.
    j (int): The label for which the numerator is to be calculated.

    Returns:
    numpy.ndarray: The numerator term for label 'j' based on the mean squared difference.
    """
    # Get the subset of data for label 'j' using the '_branch' function
    tmp = _branch(X, y, j)
    
    # Calculate the numerator term based on the mean squared difference
    aux = tmp.shape[0] * (np.mean(tmp, axis=0) - np.mean(X, axis=0))**2
    
    return aux

def _denominator(X, y, j):
    """
    Calculate the denominator term for label 'j' based on the sum of squared differences.

    Parameters:
    X (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels associated with the data.
    j (int): The label for which the denominator is to be calculated.

    Returns:
    numpy.ndarray: The denominator term for label 'j' based on the sum of squared differences.
    """
    # Get the subset of data for label 'j' using the '_branch' function
    tmp = _branch(X, y, j)
    
    # Calculate the denominator term based on the sum of squared differences
    aux = np.sum((tmp - np.mean(tmp, axis=0))**2, axis=0)
    
    return aux