import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#load json
f = open('/dataset/train.json')
data = json.load(f)
f = open('/dataset/test.json')
data_test = json.load(f)

#change json to dataframe
train_dataset = pd.DataFrame(data)
test_dataframe = pd.DataFrame(data_test)

#utilize the abstract and title
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
exclude_stopword = dict(list(number_of_auto_word.items())[:ln])
all_words = list(exclude_stopword.keys())

#exclude stopword
train_dataset["title"] = [[k for k in j if k in all_words] for j in train_dataset["title"]]
train_dataset["abstract"] = [[k for k in j if k in all_words] for j in train_dataset["abstract"]]

import numpy as np
#making y variable
real_authors = []
for i in train_dataset.authors :
    real_authors_temp = [0]*100
    onlyProAuthor = [j for j in i if j < 100]
    for l in onlyProAuthor :
      real_authors_temp[l] = 1
    real_authors.append(real_authors_temp)

y = np.array(real_authors)

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_dataset, y, test_size=0.3, random_state=42)

#making naive bayesian class
class probabilityGenerator() :
  def __init__(self):
    self.train_dict = {}
    self.freq_dict= {}
    self.probability = {}
    self.authors = None

  def probGen(self):
    for k,val in self.train_dict.items() :
      freq = self.freq_dict[k]
      self.probability[k] = {}
      for lk,lval in val.items() :
        if freq == 0 :
          self.probability[k][lk] = 0
        else :
          self.probability[k][lk] = lval/freq  

  def fitAuthors(self,x):
    self.authors = True
    for i in x :
      r_authors = [k for k in i if k < 99]
      nr_authors = [k for k in i if k > 100]
      for l in nr_authors :
        if l not in self.train_dict :
          self.train_dict[l] = {}
          self.freq_dict[l] = 0
        self.freq_dict[l] += 1
        for m in r_authors :
          if m not in self.train_dict[l] :
            self.train_dict[l][m] = 1
          else :
            self.train_dict[l][m] += 1
    self.probGen()

  def fitOthers(self,x,abs_train):
    self.authors = False
    for i, val in x.items() :
      r_authors = [k for k in abs_train[i] if k < 99]
      if val not in self.train_dict :
        self.train_dict[val] = {}
        self.freq_dict[val] = 0
      self.freq_dict[val] += 1
      for m in r_authors :
        if m not in self.train_dict[val] :
          self.train_dict[val][m] = 1
        else :
          self.train_dict[val][m] += 1
    self.probGen()

  def fitTitle(self,x,abs_train):
    self.authors = False
    for i, val in x.items() :
      r_authors = [k for k in abs_train[i] if k < 99]
      for j in val :
        if j not in self.train_dict :
          self.train_dict[j] = {}
          self.freq_dict[j] = 0
        self.freq_dict[j] += 1
        for m in r_authors :
          if m not in self.train_dict[j] :
            self.train_dict[j][m] = 1
          else :
            self.train_dict[j][m] += 1
    self.probGen()

def predict(x,y_ts,prob) :
  y_prob=sum(y_ts)/y.shape[0]
  adj_matrix_train = []
  for i,rows in x.iterrows() :
    adj_matrix_temp = [0]*100
    neg_matrix_temp = [0]*100
    fix_matrix_temp = [0]*100
    for j,val in enumerate(prob) :
      if type(rows[j]) == list :
        theList = rows[j]
      else :
        theList = [rows[j]]
      for l in theList :
        #print(l)
        if l not in val.probability :
          continue
        for m,mvalue in val.probability[l].items():
          if adj_matrix_temp[m] == 0 :
            adj_matrix_temp[m] += mvalue
            neg_matrix_temp[m] += (1-mvalue)
            #print(adj_matrix_temp[m])
          else :
            adj_matrix_temp[m] *= mvalue
            neg_matrix_temp[m] *= (1-mvalue)
            #print(adj_matrix_temp[m])
    for o in range(0,100) :
      denom = y_prob[o]*adj_matrix_temp[o]+(1-y_prob[o])*neg_matrix_temp[o]
      anom = y_prob[o]*adj_matrix_temp[o]
      if denom > 0 :
        fix_matrix_temp[o] = (anom)/(denom)
    adj_matrix_train.append(fix_matrix_temp)   
  return adj_matrix_train   

from random import sample

#making cross validation
x = train_dataset[["authors","venue","title","abstract"]]
list_index = list(x.index)
n = len(list_index)
numK = round(n/10)
list_predict = []
list_real = []
while(len(list_index) > 0) :
  if len(list_index)> numK :
      tes = sample(list_index,numK)
      list_index = [i for i in list_index if i not in tes]
  else :
      tes = list_index.copy()
      list_index.clear()
  X_test = x[x.index.isin(tes)]
  y_test = y[tes,:]
  X_train = x[~x.index.isin(tes)]
  y_train = np.delete(y,np.s_[tes],0)
  authorsProb = probabilityGenerator()
  authorsProb.fitAuthors(X_train.authors)
  venueProb = probabilityGenerator()
  venueProb.fitOthers(X_train.venue,X_train.authors)
  titleProb = probabilityGenerator()
  titleProb.fitTitle(X_train.title,X_train.authors)
  abstractProb = probabilityGenerator()
  abstractProb.fitTitle(X_train.abstract,X_train.authors)
  y_submit = predict(X_test,y_train,[authorsProb,venueProb,titleProb,abstractProb])
  list_predict.append(y_submit)
  list_real.append(y_test)    

def predict_y_value(y_s,p) :
  prediction_submit = []
  for l,i in enumerate(y_s) :
      prediction_temp = [0]*100
      for j,d in enumerate(i) :
        if d >= p :
          prediction_temp[j] = 1
      prediction_submit.append(prediction_temp)
  y_submit_array = np.array(prediction_submit)
  return y_submit_array

def f1_manual (y_hat,y_real) :
  if sum(y_real) == 0 :
    return 0
  tp = 0
  fp = 0
  fn = 0
  for i in range(0,len(y_hat)) :
    if (y_hat[i] == y_real[i]) & (y_hat[i] == 1) :
      tp += 1
      #print("tp :",tp)
    elif (y_hat[i] != y_real[i]) & (y_hat[i] == 1) :
      fp += 1
      #print("fp :",fp)
    elif (y_hat[i] != y_real[i]) & (y_hat[i] == 0) :
      fn += 1
      #print("fn :",fn)
  if (tp+fp) > 0 :
    prec = tp/(tp+fp)
  else :
    prec = 0
  if (tp+fn) > 0 :
    recall = tp/(tp+fn)
  else :
    recall = 0
  if (prec+recall) > 0 :
    f1 = 2*(prec*recall)/(prec+recall)
  else :
    f1 = 0
  return f1

from sklearn.metrics import f1_score
p = np.arange(0.1,1,0.01)
list_best =[]

for j in p :
  p_temp = []
  for i in range(0,(len(list_predict)-1)) :
    y_hat = predict_y_value(list_predict[i],j)
    y_real_temp = list_real[i]
    list_best_temp = [0]*100
    for l in range(0,y_real_temp.shape[1]):
      f1 = f1_manual(y_hat[:,l].tolist(), y_real_temp[:,l].tolist())
      list_best_temp[l] = f1
    p_temp.append(list_best_temp)
  array_p = np.array(p_temp)
  mean_columns = np.mean(array_p, axis=0).tolist()
  list_best.append(mean_columns)

cv_result = np.array(list_best)
result_prob = np.argmax(cv_result, axis=0)

#making prediction
authorsProb = probabilityGenerator()
authorsProb.fitAuthors(X_train.authors)
venueProb = probabilityGenerator()
venueProb.fitOthers(X_train.venue,X_train.authors)
titleProb = probabilityGenerator()
titleProb.fitTitle(X_train.title,X_train.authors)
y_submit = predict(X_test[["authors","venue"]],y_train,[authorsProb,venueProb])
#y_submit = predict(X_test[["authors"]],y_train,[authorsProb])

#making prediction
authorsProb = probabilityGenerator()
authorsProb.fitAuthors(train_dataset.authors)
venueProb = probabilityGenerator()
venueProb.fitOthers(train_dataset.venue,train_dataset.authors)
titleProb = probabilityGenerator()
titleProb.fitTitle(train_dataset.title,train_dataset.authors)
abstractProb = probabilityGenerator()
abstractProb.fitTitle(train_dataset.abstract,train_dataset.authors)

y_submit = predict(test_dataframe[["coauthors","venue","title","abstract"]],y,[authorsProb,venueProb,titleProb,abstractProb])
prediction_submit = []
for l,i in enumerate(y_submit) :
    temp_prediction = []
    for j,d in enumerate(i) :
      if d >= 0.1 :
       temp_prediction = temp_prediction + [str(j)]
    if len(temp_prediction) > 0 :
      prediction_submit.append([l," ".join(temp_prediction)])
    else :
      prediction_submit.append([l,'-1'])

import csv

f = open('/dataset/real bayesian with coauthors,exclude 400 abstract and exclude 400 title only 0.1.csv', 'w')
writer = csv.writer(f)

writer.writerow(['ID','Predict'])
for row in prediction_submit:
    # write a row to the csv file
    writer.writerow(row)

# close the file
f.close()