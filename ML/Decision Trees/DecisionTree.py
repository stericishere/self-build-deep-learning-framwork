"""
Decision Tree is a type of supervised learning algorithm 
that is used for classification and regression tasks.
It works by recursively splitting the data 
into subsets based on the random features.
"""
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None,
                 threshold=None, 
                 left=None, 
                 right=None, 
                 *,
                 value=None):
        self.feature_index = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = None
        if value is not None:
            self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split: int=2, max_depth: int=100, n_features: int=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = 0
        self.root = None

    def fit(self, X : np.ndarray, y : np.ndarray):
        self.root = self._grow_tree(X, y)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        
    def _grow_tree(self, X : np.ndarray, y : np.ndarray, depth : int=0) -> Node:
        n_sample, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        
        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_sample < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        
        
        
        # find the best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # create child nodes
        left_idxs, right_idxs = self._split(X, y, X[:, best_feat], best_thresh)
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left_child, right_child)
        
    def _best_split(self, X : np.ndarray, y : np.ndarray, feat_idxs : np.ndarray) -> tuple:
        """
        Find the best split for the data
        Args:
            X (np.ndarray): The data
            y (np.ndarray): The target variable
            feat_idxs (np.ndarray): The feature indices to consider
        Returns:
            tuple: The best feature index and threshold
        """
        best_info_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx, in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # calculate the information gain
                info_gain = self._information_gain(y, X_column, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold
    
    def _information_gain(self, y : np.ndarray, X_column : np.ndarray, threshold : float) -> float:
        """
        Calculate the information gain of a split
        Args:
            y (np.ndarray): The target variable
            X_column (np.ndarray): The feature column
            threshold (float): The threshold to split the data
        Returns:
            float: The information gain of the split
        """
        # calculate the parent entropy
        parent_entropy = self._entropy(y)
        
        # split the data
        left_split, right_split = self._split(X_column, y, threshold)
        
        if len(left_split) == 0 and len(right_split) == 0:
            return 0
        
        # calculate the entropy of the left and right splits
        n = len(y)
        left_entropy = self._entropy(y[left_split])
        right_entropy = self._entropy(y[right_split])
        
        child_entropy = (len(left_split) / n) * left_entropy + \
                        (len(right_split) / n) * right_entropy
        
        # calculate the information gain
        info_gain = parent_entropy - child_entropy
        
        return info_gain
    
    def _split(self, X : np.ndarray, y : np.ndarray, feat_idx : int, threshold : float) -> tuple:
        """
        Split the data into left and right based on the threshold
        Args:
            X (np.ndarray): The data
            y (np.ndarray): The target variable
            feat_idx (int): The feature index to split on
            threshold (float): The threshold to split the data
        Returns:
        """
        left_idxs = np.argwhere(X[:, feat_idx] <= threshold).flatten()
        right_idxs = np.argwhere(X[:, feat_idx] > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y : np.ndarray) -> float:
        """
        Calculate the entropy of the target variable
        Args:
            y (np.ndarray): The target variable
        Returns:
            float: The entropy of the target variable
        """
        # calculate the entropy of the target variable
        # entropy = -sum(p * log2(p))
        # p is the probability of the label
        # log2(p) is the log of the probability
        # sum is the sum of the entropy of the labels
        hist = np.bincount(y)
        ps = hist / len(y)
        # if p is 0, then log2(p) is -inf, so we need to handle that
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _entropy_by_split(self, left : np.ndarray, right : np.ndarray, len_parent : int) -> float:
        """
        Calculate the entropy of the split
        """
        p = len(left) / len_parent
        q = len(right) / len_parent
        return p * self._entropy(left) + q * self._entropy(right)
        
        
    def _most_common_label(self, y : np.ndarray) -> int:
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        # Counter({'apple': 7, 'orange': 4, 'banana': 2})
        # counter.most_common(n) returns a list of top n tuples, 
        # counter.most_common(1) = [('apple', 7)]
        # most_common(1)[0] = ('apple', 7)
        # most_common(1)[0][0] = 'apple'
        return most_common
    
    def _best_split(self, X : np.ndarray, y : np.ndarray) -> tuple:
        best_split = {}
        best_info_gain = 0
        
    def predict(self, X : np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x : np.ndarray, node : Node) -> int:
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)