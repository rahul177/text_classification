# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:07:57 2017

@author: Rahul Kumar
"""

import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

# Importing Datasets
reviews=load_files("txt_sentoken/")
X,y=reviews.data,reviews.target

# Storing as Pickel File
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
# Unpickling the Dataset
with open('X.pickle','rb') as f:
    X=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)

# Preprocessing Data
corpus=[]
for i in range(len(X)):
    review=re.sub(r'\W',' ',str(X[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+','',review)
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)

# Bag Of Words Model Using Python In-bulit Library :
from sklearn.feature_extraction.text import CountVectorizer
# min_df : When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. 
# max_df : If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()

# Transform Bag Of Model to TF-IDF Model :
from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()

#For Twitter Sentiment Analysis we need to pickle TfidfVectorizer:
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()


# Splitting Training and Testing Data:
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test=train_test_split(X,y,test_size=0.2)

# Logistics Regression:
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(text_train,sent_train)

# Testing Model Performance:
sent_pred=classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,sent_pred)    

# Pickling the Classifier:
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

# Picking the Vectorizer:
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
#Importing and Saving Model:
    
# Step 1 : Unpickling Classifier and Vectorizer:
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)

# Step 2 : Use Model
sample=['You are nice person man']
sample=tfidf.transform(sample).toarray()
print(clf.predict(sample))