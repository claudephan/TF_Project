import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Parameters
num_steps = 250 # Total steps to train
batch_size = 64 # The number of samples per batch
num_classes = 3 # NSP
num_features = 21
num_trees = 10
max_nodes = 100
maindir = '/home/claude/Documents/DS_CERT/'
df = pd.read_csv(maindir +'input/train2.csv')

#Divide up the data and labels
train = df.sample(frac=0.8)
test = df.drop(train.index)

#Train Data
trainDF = train[train.columns[:num_features]]    #DF
X_Train = trainDF.values.astype(np.float32) #ndarray

train_labelDF = train[train.columns[-1]]
Y_Train = train_labelDF.values.astype(np.int32) -1

#Test data
testDF = test[test.columns[:num_features]]    #DF
X_Test = testDF.values.astype(np.float32) #ndarray

test_labelDF = test[test.columns[-1]]
Y_Test = test_labelDF.values.astype(np.int32) -1

sess = tf.Session()
# Random Forest Parameters
with tf.name_scope("setup_RF"):
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes,
                                          regression=False).fill()

with tf.name_scope("RF_build"):
    previous_value = tf.Variable(0.0, dtype=tf.float32, trainable=False,name="previous_value")
    classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams, model_dir=maindir)
    classifier.fit(x=X_Train, y=Y_Train, steps=num_steps, batch_size=batch_size)
    classifier.evaluate(x=X_Train, y=Y_Train, steps=10)
    y_out = classifier.predict(X_Test)

with tf.name_scope("accurcy_check"):
    predictions = []
    for i in y_out:
        predictions.append(i['classes'])

    tf_predictions = tf.convert_to_tensor(predictions, dtype=tf.int32)
    y = tf.convert_to_tensor(Y_Test, dtype=tf.int32)
    correct_prediction = tf.equal(tf_predictions, y)
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    previous_value = previous_value.assign(accuracy_op)

    #Print Accuracy
    print(sess.run(accuracy_op))

    #Print Incorrect
    for j in range(len(predictions)):
        if(predictions[j] != Y_Test[j]):
            print(predictions[j], Y_Test[j])

writer = tf.summary.FileWriter(maindir+'/events', graph=tf.get_default_graph())
writer.flush()
writer.close()
sess.close()
