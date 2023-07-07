# import all related packages
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from random import sample
import tensorflow as tf 
import os
import tempfile
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from scipy import sparse  

#load json
f = open('/dataset/train.json')
data = json.load(f)
f = open('/dataset/test.json')
data_test = json.load(f)

#change json to dataframe
train_dataset = pd.DataFrame(data)
test_dataframe = pd.DataFrame(data_test)

#split train dataset
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train_dataset, test_size=0.3, random_state=42)

def tfidf(vectorizer, data_list):
    # data_list : list of string
    the_abs = []

    for i in data_list:
        the_abs.append(" ".join(str(x) for x in i))

    X_tf = vectorizer.fit_transform(the_abs)
    col_names_co_abstract =vectorizer.get_feature_names_out()
    abstract_array = X_tf.toarray()

    return abstract_array, vectorizer

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

#utilize the abstract and title
#calculate most ferquent word
number_of_auto_word = {}
the_list_of_word = train_dataset.abstract.tolist() + train_dataset.title.tolist()
for i in the_list_of_word :
    for j in i :
        if j not in number_of_auto_word :
            number_of_auto_word[j] = 0
        number_of_auto_word[j] += 1
list_importance_key = list(number_of_auto_word.keys())
number_of_auto_word = dict(sorted(number_of_auto_word.items(), key=lambda item: item[1]))
ln = len(number_of_auto_word)-400
#exclude most ferquent word
exclude_stopword = dict(list(number_of_auto_word.items())[:1000])
all_words = list(exclude_stopword.keys())

#for kaggle
vectorizer = CountVectorizer(ngram_range = (1,1))
all_abstract_slice = []
for i in train_dataset.abstract :
    all_abstract_temp = [j for j in i if j in all_words]
    all_abstract_slice.append(all_abstract_temp)

abstract_array,vectorizer = tfidf(vectorizer, all_abstract_slice)
abstract_array_train = sparse.csr_matrix(abstract_array)

all_abstract_test = []
for i in test_dataframe.abstract :
    all_abstract_temp = " ".join([str(j) for j in i if j in all_words])
    all_abstract_test.append(all_abstract_temp)

abstract_array_test = sparse.csr_matrix(vectorizer.transform(all_abstract_test))

vectorizer = CountVectorizer(ngram_range = (1,1))
all_abstract_slice = []
for i in train_dataset.title :
    all_abstract_temp = [j for j in i if j in all_words]
    all_abstract_slice.append(all_abstract_temp)

title_array,vectorizer = tfidf(vectorizer, all_abstract_slice)
title_array_train = sparse.csr_matrix(title_array)

all_abstract_test = []
for i in test_dataframe.title :
    all_abstract_temp = " ".join([str(j) for j in i if j in all_words])
    all_abstract_test.append(all_abstract_temp)

title_array_test = sparse.csr_matrix(vectorizer.transform(all_abstract_test))

#for kaggle
#split coauthors and authors
real_authors = []
co_authors = []

n = train_dataset.shape[0]
for i in train_dataset.authors :
    real_authors.append(" ".join([str(j) for j in i if j < 100]))
    co_authors.append(" ".join([str(j) for j in i if j > 99]))

for i in test_dataframe.coauthors :
    #real_authors.append(" ".join([str(j) for j in i if j < 100]))
    co_authors.append(" ".join([str(j) for j in i if j > 99]))


#making unigram for co authors
vectorizer = CountVectorizer(ngram_range = (1,1))
X_tf = vectorizer.fit_transform(co_authors)
col_names_co_authors = vectorizer.get_feature_names_out()
X_coauthors = X_tf.toarray()
X_coauthors_test= X_coauthors[n:,:]
X_coauthors_train = X_coauthors[0:n,:]

X_coauthors_test = sparse.csr_matrix(X_coauthors_test)
X_coauthors_train = sparse.csr_matrix(X_coauthors_train)


#making unigram for real authors
vectorizer = CountVectorizer(ngram_range = (1,1))
X_tf = vectorizer.fit_transform(real_authors)
col_names_authors = vectorizer.get_feature_names_out()
y = X_tf.toarray()
#y_test= y[n:,:]
#y_train = y[0:n,:]

#del unimportance variables
del co_authors
del real_authors
del X_tf

#establish sparse matrix
from scipy.sparse import hstack
X_train=hstack((X_coauthors_train, abstract_array_train,title_array_train))
X_test=hstack((X_coauthors_test, abstract_array_test,title_array_test))

#calculate pos and neg
initial_bias = []

for i in range(0,y.shape[1]) :
  neg_temp, pos_temp = np.bincount(y[:,i])
  initial_bias = initial_bias + [np.log(pos_temp/neg_temp)]

#initial_bias = np.array([sum(initial_bias)/len(initial_bias)])
initial_bias = keras.initializers.Constant(np.array(initial_bias))

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')

X_train = convert_sparse_matrix_to_sparse_tensor(X_train)
X_test = convert_sparse_matrix_to_sparse_tensor(X_test)

#for kaggle
BATCH_SIZE = 4096

early_stopping = keras.callbacks.EarlyStopping(
    monitor='prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

input_shape = [X_train.shape[1]]

model = keras.Sequential([
    #layers.Dense(128, activation='tanh', input_shape=input_shape,kernel_regularizer=L2Regularizer(l2=0.001)),
    #layers.BatchNormalization(),
    layers.Dense(512, activation='tanh', input_shape=input_shape),
    layers.Dropout(0.3),
    #layers.BatchNormalization(),
    layers.Dense(512, activation='tanh'), 
    layers.Dropout(0.3),
    #layers.BatchNormalization(),
    layers.Dense(512, activation='tanh'), 
    layers.Dropout(0.3),
    #layers.BatchNormalization(),
    layers.Dense(512, activation='tanh'), 
    layers.Dropout(0.3),
   # layers.BatchNormalization(),
    #layers.BatchNormalization(),   
    layers.Dense(y.shape[1], activation='sigmoid',bias_initializer=initial_bias)
])

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      f1_m
]
model.compile(
    optimizer='adam',
    loss=keras.losses.BinaryFocalCrossentropy(),
    metrics=METRICS,
)

history = model.fit(
    tf.sparse.reorder(X_train), y,
    batch_size=BATCH_SIZE,
    epochs=1000,
    callbacks=[early_stopping]
)

#submit the test data prediction
y_submit = model.predict(tf.sparse.reorder(X_test),4096)
prediction_submit = []
for l,i in enumerate(y_submit) :
    temp_prediction = []
    for j,d in enumerate(i) :
      if d >= 0.5 :
       temp_prediction = temp_prediction + [col_names_authors[j]]
    if len(temp_prediction) > 0 :
      prediction_submit.append([l," ".join(temp_prediction)])
    else :
      prediction_submit.append([l,'-1'])

import csv

f = open('/dataset/Last Change ANN Abstract,title use 1000 tanh first word 0.3.csv', 'w')
writer = csv.writer(f)

writer.writerow(['ID','Predict'])
for row in prediction_submit:
    # write a row to the csv file
    writer.writerow(row)

# close the file
f.close()