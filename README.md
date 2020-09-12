# doc2vec-embeddings-and-classification

Loading the doc2vec numpy arrays for training classifiers generated from the label_data excel. 1000 sentences, 800 in training arrays, 200 in test arrays, 2 categories - relevant and irrelevant, split 75-25 in the dataset. Loading for connection to google.colab.

The doc2vec_classif files contain SVM classifiers. Performed poorly.

The doc2vec_keras file is the first attempt at tensorflow.

The doc2vec_np_arrays.npz contains four numpy arrays of the doc2vec-embedded sentences from the .txt files. 
