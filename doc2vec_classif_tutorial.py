# from https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# random
import random

# loading model
model = Doc2Vec.load('doc2vec_model1')

# We create a numpy array (since the classifier we use only takes numpy arrays.)
# There are two parallel arrays, one containing the vectors (train_arrays) and the other containing the labels (train_labels).
train_arrays = numpy.zeros((800, 100))
train_labels = numpy.zeros(800)
# test-irrel: 145
# test-rel: 55
# train-irrel: 583
# train-rel: 217

for i in range(217):
    prefix_train_rel = 'TRAIN_REL_' + str(i)
    train_arrays[i] = model[prefix_train_rel]
    train_labels[i] = 1

for i in range(1,583):
    prefix_train_irrel = 'TRAIN_IRREL_' + str(i)
    train_arrays[217 + i] = model[prefix_train_irrel]
    train_labels[217 + i] = 0

# We do the same for the testing data

test_arrays = numpy.zeros((200, 100))
test_labels = numpy.zeros(200)

for i in range(55):
    prefix_test_rel = 'TEST_REL_' + str(i)
    test_arrays[i] = model[prefix_test_rel]
    test_labels[i] = 1

for i in range(1,145):
    prefix_test_irrel = 'TEST_IRREL_' + str(i)
    test_arrays[55 + i] = model[prefix_test_irrel]
    test_labels[55 + i] = 0

# checking shape
print(train_arrays.shape, train_labels.shape)
print(test_arrays.shape, test_labels.shape)

# saving these doc2vec numpy arrays
numpy.savez('doc2vec_np_arrays', train_arrays = train_arrays, train_labels = train_labels, test_arrays = test_arrays, test_labels = test_labels)

# train logic regression classifier (linear SVM) and output accuracy

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print(classifier.score(test_arrays, test_labels))

# build confusion matrix

pred_labels = classifier.predict(test_arrays)
conf_matrix = confusion_matrix(test_labels, pred_labels)
print(conf_matrix)
