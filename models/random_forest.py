from decision_tree import DecisionTree
import numpy as np

from abstract_tuned_models import DataFormating

class RandomForest(DecisionTree):
    def __init__(this, min_sample_split=2, max_depth=50, n_features=None, num_trees = 50):
        super().__init__(min_sample_split, max_depth, n_features)
        this.num_trees = num_trees
        this.X_split = None
        this.y_split = None
    

    def create_forest(this,X,y):
        forest = []
        for i in range(this.num_trees):
            tree = DecisionTree()
            this.X_split, this.y_split = this.split_dataset(X,y)
            tree.fit(this.X_split,this.y_split)
            forest.append(tree)
        return forest
            

    def split_dataset(this,X,y):
        feature_indexes = np.random.choice(X.shape[0], int(X.shape[0] * 0.67), replace=True)
        return X[feature_indexes], y[feature_indexes]
    
    def get_best_combination(this,X,y,test_X, test_y):
        predictions = []
        size = 0
        forest = this.create_forest(X,y)
        for tree in forest: 
           prediction =  tree.predict(test_X)
           predictions.append(prediction)
           size = len(prediction)
        most_common_prediction = []
        for i in range(size):
            snapshot = []
            for prediction in predictions:
                snapshot.append(prediction[i])
            most_common_prediction.append(tree.most_common_label(snapshot))
        print(len(most_common_prediction))
        print(most_common_prediction)
        print(test_y)
        return most_common_prediction



data_format = DataFormating()
data_format.fully_balanced(True)
data_format.set_training_and_test_data()

rf = RandomForest()
res = rf.get_best_combination(data_format.X_train.values,data_format.y_train.values,data_format.X_test.values,data_format.y_test.values)
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(data_format.y_test.values, res)
print(acc)
# predictions = rf.predict(data_format.X_test.values)
# def accuracy(y_test, y_pred):
#         return np.sum(y_test == y_pred) / len(y_test)

# print(predictions)
# print(data_format.y_test.values)

# acc = accuracy(data_format.y_test.values, predictions)
# print(acc)
