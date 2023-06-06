# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:03:53 2023

@author: Mohan Patil
"""

import streamlit as st
import pandas as pd
import pickle 
import numpy as np
import string
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


with open("final_model_.pkl","rb") as file:
    model = pickle.load(file,encoding="utf-8")

# Set up stopwords and stemmer
stop = stopwords.words('english')
stemmer = PorterStemmer()

def preprocess_text(text):
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\w*\d\w*', '', text)
  text = re.sub("[0-9" "]+", " ", text)
  text = re.sub('[‘’“”…]', '', text)
  text = re.sub(r"http\S+|www\S+|https\S+","",text,flags=re.MULTILINE)
  text = re.sub(r"\@\w+|\#","",text)
  text = re.sub(r"[^\w\s]","",text)
  text = re.sub(r"\d+","",text)
  return text

word2_vec = model['word2vec']
classifier = model['log_reg']

# Remove stopwords
def remove_stopwords(sentence):
    return " ".join(i for i in sentence.split() if i not in stop)
# Stemming
def stem_text(text):
    stem_text = [stemmer.stem(token) for token in text.split()]
    return " ".join(stem_text)

def preprocess_tweet(tweet):
    tokenized_text = tweet.split()
    embeddings = [word2_vec.wv[word] for word in tokenized_text if word in word2_vec.wv]
    if embeddings:
        tweet_embedding = np.mean(embeddings, axis=0)
    else:
        tweet_embedding = np.zeros(word2_vec.vector_size)
    return tweet_embedding

class_names = {
    0: "figurative",
    1: "irony",
    2: "regular",
    3: "sarcasm"}

def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Enter a tweet to predict its sentiment and emotion:")
    tweet = st.text_input("Enter a tweet")
    
    if st.button("Predict"):
        clean_tweet = preprocess_text(tweet)
        remove_stopword = remove_stopwords(clean_tweet)
        stemming = stem_text(remove_stopword)
        tweet_embedding = preprocess_tweet(stemming)
        predict = classifier.predict([tweet_embedding])
        sentiment_label = predict[0]
        sentiment_class = class_names[sentiment_label]
        st.success(sentiment_class)
        
        
if __name__ == "__main__":
    main()