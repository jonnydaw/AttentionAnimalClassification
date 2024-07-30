# Import necessary libraries
import numpy as np
import pandas as pd
from abstract_tuned_models import DataFormating

# https://www.youtube.com/watch?v=kFwe2ZZU7yw&ab_channel=AssemblyAI
# NOT ORIGINAL CODE

class Node:
    def __init__(this, feature = None, threshold = None, left= None, right= None, value= None):
        this.feature = feature
        this.threshold = threshold
        this.left = left
        this.right = right
        # this.gain = gain
        this.value = value
    
    def leaf_node(this):
         return this.value is not None

class DecisionTree:
   
    def __init__(this, min_sample_split = 2, max_depth = 50, n_features = None):
        this.root = None
        this.min_samples_split = min_sample_split
        this.max_depth = max_depth
        this.n_features = n_features

    def fit(this, X,y ):    
        this.n_features = X.shape[1] if not this.n_features else min(X.shape[1], this.n_features)
        #print(this.n_features)
        this.root = this.grow_tree(X,y) 

    def grow_tree(this, X,y,depth = 0):
        num_samples, num_features = X.shape
        n_labels = len(np.unique(y))

        if(depth >= this.max_depth or n_labels == 1 or num_samples < this.min_samples_split):
            val = this.most_common_label(y)
            return Node(value=val)
        
        feature_indexes = np.random.choice(num_features, this.n_features,replace=False)
        best_feature, best_threshold = this.best_split(X,y,feature_indexes)

        left_idxs, right_idxs = this.split(X[:,best_feature], best_threshold)
        left = this.grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = this.grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feature, best_threshold,left,right)

    def most_common_label(this,nums):
        majority = -1_000_000
        count = 0
        for num in nums:
            if(count == 0):
                majority = num
                count += 1
            elif(num == majority):
                count += 1
            else:
                count -= 1
        return majority
    
    def best_split(this, X, y, feature_indexes):
        best_gain = -1_000_000
        split_index,split_threshold = None,None
        for idx in feature_indexes:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = this.information_gain(X_column, y, threshold)
                if(gain > best_gain):
                    best_gain = gain
                    split_index = idx
                    split_threshold = threshold
        return split_index, split_threshold

    def information_gain(this,X_col, y, threshold):
        p_entropy = this.entropy(y)
        left_idx, right_idx = this.split(X_col, threshold)
        if(len(left_idx) == 0 or len(right_idx) == 0 ):
            return 0
        num = len(y)
        num_left, num_right = len(left_idx), len(right_idx)
        entropy_left, entropy_right = this.entropy(y[left_idx]), this.entropy(y[right_idx])
        child_entropy = (num_left / num) * entropy_left + (num_right / num) * entropy_right 
        information_gain = p_entropy - child_entropy
        return information_gain

    def entropy(this,y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def split(this, X_col, split_threshold):
        left_idx = np.argwhere(X_col <= split_threshold).flatten()
        right_idx = np.argwhere(X_col > split_threshold).flatten()
        return left_idx, right_idx
    
    def predict(this, X):
        return np.array([this.traverse_tree(x,this.root) for x in X])

    def traverse_tree(this,x, node):
        if(node.leaf_node()):
            return node.value
        if x[node.feature] <= node.threshold:
            return this.traverse_tree(x,node.left)
        return this.traverse_tree(x,node.right)

if __name__ == "__main__":
    data_format = DataFormating()
    data_format.fully_balanced(True)
    data_format.set_training_and_test_data()
    dt = DecisionTree()
    dt.fit(data_format.X_train.values, data_format.y_train.values)

    predictions = dt.predict(data_format.X_test.values)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    print(predictions)
    print(data_format.y_test.values)

    acc = accuracy(data_format.y_test.values, predictions)
    print(acc)

