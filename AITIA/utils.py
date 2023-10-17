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
            
            # Store the count of instances for each unique y value in the leaf
            instances_by_class = {}
            for i in range(len(X)):
                if tree.apply(X[i].reshape(1, -1))[0] == leaf_node:
                    y_value = y[i]
                    if y_value in instances_by_class:
                        instances_by_class[y_value] += 1
                    else:
                        instances_by_class[y_value] = 1

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