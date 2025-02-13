# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:57:06 2025

@author: grade
"""

#NOTES: MAKE SURE THAT I RERUN THIS TO SAVE THE MODEL

#WILL LOAD MODEL ON DIFFERENT SCRIPT AFTERWARDS TO TEST IF IT CAN DETECT USER INPUTS

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import re #text cleaning
import nltk
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from keras import models
import matplotlib.pyplot as plt
import warnings

import pickle

warnings.filterwarnings("ignore")

np.random.seed(1)

random.seed(1)

df = pd.read_csv(r"C:\Users\grade\Downloads\sentiment analysis on reddit comments data\Reddit_Data.csv")
df2 = pd.read_csv(r"C:\Users\grade\Downloads\sentiment analysis on reddit comments data\Twitter_Data.csv")

df =df.rename(columns={'clean_comment': 'clean_text'})
all_text_df = pd.concat([df, df2], ignore_index=True)

print(all_text_df.shape)
print(all_text_df.head(5))
print(all_text_df.isnull().sum())
print("")
print(all_text_df.iloc[:,1].value_counts()) #number of positive and negative reviews, 1 is positive, -1 is negative, 0 is neutral

#small number of nulls, they will be removed
all_text_df.dropna(subset = ['clean_text'],inplace = True)
all_text_df.dropna(subset = ['category'],inplace = True)
print(all_text_df.isnull().sum())
print(all_text_df.shape)
#___________________________________________________________________________
#THIS TEXT CLEAN INSURANCE CODE MAY BE REMOVED AS THE DATASET CLAIMS TO BE CLEAN TEXT ALREADY
#text cleaning insurance
all_text_df["clean_text"] = all_text_df["clean_text"].str.lower()    #all characters to lowercase

#assistance for text cleaning in Amazon dataset
for text in all_text_df.clean_text:
    text = re.sub("[a-zA-Z]","",text) #removes all special characters in a text and replaces it with an empty string
    text = nltk.word_tokenize(text)
    lemma = nltk.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text] 
    
#matve do this before tokenization and lemmatization possibly
stop = stopwords.words('english')
all_text_df['clean_text'] = all_text_df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#____________________________________________________________________________________________
#THE TEXT CLEAN ENDS ABOVE THIS LINE IT MAY BE REMOVED LIKE POSSIBLY


#identification of vocabulary size in text dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text_df["clean_text"])
print("Vocabulary Size: ", len(tokenizer.word_index)+1)
vocab_size = len(tokenizer.word_index)+1
embedding_dim = 400 #embedding dim for this vocab size is recommended at least 300 or more, experiement with this ok?

#identifies number of words in each text, NOT characters
text_arr = all_text_df["clean_text"].to_numpy()
text_word_cnt= []
for word_cnt in text_arr:
   text_word_cnt.append(len(word_cnt.split(" ")))

print(np.max(text_word_cnt))
print(np.min(text_word_cnt))
print(np.median(text_word_cnt))

max_length = np.max(text_word_cnt)

#creating the train and test sets
X = np.array(all_text_df['clean_text'])
y = all_text_df.category.values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state=1,stratify=y)

y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

print(X_train.shape)
print(X_test.shape)

#padding train and test sets
sequences_train = tokenizer.texts_to_sequences(X_train)
padded_train = pad_sequences(sequences_train,maxlen = max_length) #chatgpt fix 

sequences_test = tokenizer.texts_to_sequences(X_test)
padded_test = pad_sequences(sequences_test,maxlen = max_length) #chatgpt fix

#converting data to numpy array for model usage

trainPadArr = np.array(padded_train)
trainLab = np.array(y_train)

testPadArr = np.array(padded_test)
testLab = np.array(y_test)

#__________________________________________________________________________
#CREATION OF MODEL
trainLab = trainLab+1
testLab = testLab+1

#_______________________________________________chat gpt fix converting float point to integer, and debugging before training 
trainLab = trainLab.astype(int)
testLab = testLab.astype(int)
print("Train Padded Shape:", trainPadArr.shape)
print("Test Padded Shape:", testPadArr.shape)
print("Unique Labels:", np.unique(trainLab))


earlyStop = EarlyStopping(patience = 2,restore_best_weights=True) #will stop running if 2 runs with no improvement and restore prior weights

input_layer = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)
lstm1 = keras.layers.LSTM(128, return_sequences=True)(embedding)
lstm2 = keras.layers.LSTM(64)(lstm1)
dense1 = Dense(32, activation="relu")(lstm2)
output = Dense(3, activation='softmax')(dense1)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
   
    
    
#____________________________________________________________________________________
#STUFF UNDDER HEAR IS FROM MY MODEL FROM MY COURSEWORK PROJECT
#model.add(Embedding(input_dim = vocab_size,output_dim = 100,input_length = max_length))
#model.add(SimpleRNN(units = 100)) #test 100 and 17 17 is max sequence length, leave blank lets see as well
#model.add(Dense(units = 1,activation = "sigmoid")) #sigmoid is better for binary classification, classify if review is good or bad
#model.build(input_shape=(None, 10))




#model.compile(loss = "binary_crossentropy", optimizer = 'adam',metrics = ['accuracy']) 
#model.summary()


history = model.fit(trainPadArr,trainLab,epochs = 10,validation_data = (testPadArr,testLab),batch_size = 128,callbacks=[earlyStop]) #question epoch and batch_size for my dataset

# Save the trained model
model.save("model.keras")

# Save the tokenizer

with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
test_loss, test_acc = model.evaluate(testPadArr, testLab)

print(f"Test Accuracy: {test_acc:.4f}")

# 7️⃣ Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


