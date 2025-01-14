import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, val_class=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.val_class = val_class

class Desicion_tree():
    def __init__(self, min_samples_split=2, max_depth=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def make_tree(self, dataset, depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        samples, features = X.shape
        
        if samples >= self.min_samples_split and (self.max_depth is None or depth <= self.max_depth):
            best_split = self.find_best_split(dataset, samples, features)
            if best_split and best_split.get("info_gain", 0) > 0:
                left_subtree = self.make_tree(best_split["dataset_left"], depth + 1)
                right_subtree = self.make_tree(best_split["dataset_right"], depth + 1)
                return Node(best_split["feature"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])

        leaf_val = self.calculate_leaf_val(Y)
        return Node(val_class=leaf_val)

    def find_best_split(self, dataset, samples, features):
        best_split = {}
        max_info_gain = float("-inf")

        for feature in range(features):
            feature_vals = dataset[:, feature]
            
            valid_indices = ~np.isnan(feature_vals)
            feature_vals_valid = feature_vals[valid_indices]
            dataset_valid = dataset[valid_indices]
            
            nan_indices = np.isnan(feature_vals)
            dataset_nan = dataset[nan_indices]
            
            unique_thresholds = np.unique(feature_vals_valid)

            for threshold in unique_thresholds:
                dataset_left_valid, dataset_right_valid = self.split(dataset_valid, feature, threshold)
                
                if len(dataset_left_valid) > 0 and len(dataset_right_valid) > 0:

                    Y, left_Y, right_Y = dataset[:, -1], dataset_left_valid[:, -1], dataset_right_valid[:, -1]

                    curr_info_gain = self.info_gain(Y, left_Y, right_Y)

                    if curr_info_gain > max_info_gain:
                        best_split["feature"] = feature
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = np.concatenate((dataset_left_valid, dataset_nan), axis=0)
                        best_split["dataset_right"] = dataset_right_valid
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, dataset, feature, threshold):
        dataset_left = np.array([row for row in dataset if row[feature] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature] > threshold])
        return dataset_left, dataset_right

    def info_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        p_cls = np.array([len(y[y == cls]) / len(y) for cls in class_labels])
        p_cls = p_cls[p_cls > 0]
        return -np.sum(p_cls * np.log2(p_cls))
    
    def calculate_leaf_val(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y.values.reshape(-1, 1)), axis=1)
        self.root = self.make_tree(dataset)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X.values]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.val_class != None: return tree.val_class
        feature_val = x[tree.feature]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def print_tree(self, node, features, indent=""):
         if node.val_class is not None:
             print(indent + f"Класс: {node.val_class}")
         else:
             feature_name = features[node.feature]
             print(indent + f"Признак:'{feature_name}' <= {node.threshold}")
             self.print_tree(node.left, features, indent + "  ")
             print(indent + f"Признак: '{feature_name}' > {node.threshold}")
             self.print_tree(node.right, features, indent + "  ")
