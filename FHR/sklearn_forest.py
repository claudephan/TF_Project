# linear algebra
import numpy as np
import tensorflow as tf
# data processing
import pandas as pd

# data visualization
import seaborn as sns
#%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

num_classes = 3 # NSP
num_features = 17
maindir = '/home/claude/Documents/DS_CERT/'
df = pd.read_csv(maindir +'input/train2.csv')
train = df.sample(frac=0.8)
test = df.drop(train.index)

#Divide up the data and labels
X_train = train[train.columns[:num_features]]    #DF
#X_train = trainDF.values.astype(np.float32) #ndarray

Y_train = train[train.columns[-1]]
#Y_train = train_labelDF.values.astype(np.int32) -1

#Test data
#Divide up the data and labels
X_test = test[test.columns[:num_features]]    #DF
#X_test = testDF.values.astype(np.float32) #ndarray

Y_test = test[test.columns[-1]]
#Y_test = test_labelDF.values.astype(np.int32) -1


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%", '\n')


rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std(), '\n')

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances)
print(importances.plot.bar())
'''
sess = sess = tf.Session()
with tf.name_scope("accurcy_check"):

    tf_predictions = tf.convert_to_tensor(Y_prediction, dtype=tf.int32)
    y = tf.convert_to_tensor(Y_test, dtype=tf.int32)
    correct_prediction = tf.equal(tf_predictions, y)
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Print Accuracy
    print(sess.run(accuracy_op))

    #Print Incorrect
    for j in range(len(Y_prediction)):
        if(Y_prediction[j] != Y_test[j]):
            print(Y_prediction[j], Y_test[j])
'''
