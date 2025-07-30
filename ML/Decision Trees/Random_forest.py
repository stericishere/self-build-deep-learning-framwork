from DecisionTree import DecisionTree # Assuming this is a correct implementation
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees: int=10, min_samples_split: int=2, max_depth: int=100, n_features: int=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
    
    def _bootstrap_samples(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Creates a bootstrap sample by sampling data points with replacement."""
        n_samples = X.shape[0]
        # Generate random indices with replacement
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the Random Forest on bootstrap samples of the data."""
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth,
                                n_features=self.n_features)
            
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict_classification(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X using a majority vote."""
        # Get predictions from all trees. Shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to shape (n_samples, n_trees) 
        # so each row is one sample's predictions
        tree_preds = tree_preds.T
        # Use the to get the most common label for each row (sample)
        y_pred = np.array([self._most_common_label(preds) for preds in tree_preds])
        return y_pred
    
    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values for samples in X by averaging tree predictions."""
        # Get predictions from all trees. Shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # np.mean with axis=0 calculates the mean down the columns (for each tree)
        return np.mean(tree_preds, axis=0)
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """Finds the most frequent label in an array of predictions."""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common