# Streamlit dependencies
import streamlit as st
import joblib,os
import plotly.express as px
from plotly import graph_objects as go

# Data dependencies
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import string
import re    #for regex
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from wordcloud import WordCloud 
from collections import Counter
#import spacy
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer,PorterStemmer,LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns 

# Vectorizer
news_vectorizer = open("resources/models/TS5Vectorizer2.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/clean.csv")


df = raw.copy()



def main():
    """Tweets classifier App with Streamlit"""

    st.title('Tweet Sentiment Classifier')

    
    #image = Image.open('resources/imgs/image.jpg')

    #st.image(image, caption='Tweet Sentiments', use_column_width=True)

    st.subheader('Climate Change Belief Analysis: Based on Tweets')
    

    ##creating a sidebar for selection purposes


    options = ['Home','Information', 'EDA', 'predict tweet', 'Lets connect!']
    st.sidebar.subheader("Navigation")
    selection = st.sidebar.selectbox('Go to', options)

    
    


    # Building information page
    if selection == 'Information':
        st.info("General Information")
        st.markdown('This section explains how to naviagte the app')
        st.markdown('**********************************************************************************')
        st.markdown(' This app will take user input tweet in a form of a text')
        st.markdown('The app then allows the user to choose the model they want to use for their tweet')
        st.markdown('***********************************************************************************')
        st.markdown('To access all pages, use the navigation bar')
        st.markdown('Check the EDA page for visuals on trained dataset and use predict tweet page to predict your tweet')
        st.markdown('Let the tweet spy game begin hahhaa!!!')

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):
            st.write(raw[['sentiment', 'message']].head())

    
    ## prediction page
    if selection == 'predict tweet':

        st.info("Make tweet Predictions with ML Models of your choice")

        choice = ['Single Tweet', 'Dataset'] #choices between a single tweet and a dataset input

        option = st.selectbox('What to classify?', choice)

        # load pickle file
        def load_pickle(pkl_file):
            model = joblib.load(open(os.path.join(pkl_file),"rb"))
            return model




        if choice == 'Single Tweet':
            #For single tweet
            st.subheader('Classify single tweet')

            tweet_text1 = st.text_area("Enter Tweet","Type Here") #Creating a text box for user input
            models1 = ['LinearSVC', 'KNeighborsClassifier','DecisionTreeClassifier', 
			'RandomForestClassifier', 'ComplementNB', 'MultinomialNB',
			'AdaBoostClassifier']

            # Selection for model to be used
            model1 = st.selectbox("Choose ML Model",models1)

            

            dict_labels = {'Positive':1,'Neutral':0,'Negative':-1,'News':2}
            if st.button('Classify'):

                
                #clean_text1 = preprocess_stemm(tweet_text1) 
                vect_text = tweet_cv.transform([tweet_text1]).toarray()

                if model1 == 'LinearSVC':
                    predictor = load_pickle("resources/models/LinearSVC.pkl")
                    prediction = predictor.predict(vect_text)
                    
                elif model1 == 'KNeighborsClassifier':
                    predictor = load_pickle("resources/models/KNeighborsClassifier.pkl")
                    prediction = predictor.predict(vect_text)
                    
                elif model1 == 'DecisionTreeClassifier':
                    predictor = load_pickle("resources/models/DecisionTreeClassifier.pkl")
                    prediction = predictor.predict(vect_text)
                    
                elif model1 == 'RandomForestClassifier':
                    predictor = load_pickle("resources/models/RandomForestClassifier.pkl")
                    prediction = predictor.predict(vect_text)
				
                elif model1 == 'ComplementNB':
                    predictor = load_pickle('resources/models/ComplementNB.pkl')
                    prediction = predictor.predict(vect_text)

                elif model1 == 'MultinomialNB':
                    predictor = load_pickle('resources/models/MultinomialNB.pkl')
                    prediction = predictor.predict(vect_text)

                elif model1 == 'AdaBoostClassifier':
                    predictor = load_pickle('resources/models/AdaBoostClassifier.pkl')
                    prediction = predictor.predict(vect_text)    



                final_pred = dict_labels[prediction]
                st.success("Tweet Categorized as:: {}".format(final_pred))

        if option == 'Dataset':
            #For data set classification
            st.subheader('Multiple tweet classification')

            models2 = ['LinearSVC', 'KNeighborsClassifier','DecisionTreeClassifier', 
			'RandomForestClassifier', 'ComplementNB', 'MultinomialNB',
			'AdaBoostClassifier']

            model2 = st.selectbox("Choose Model",models2)

            dict_labels2 = {'Positive':1,'Neutral':0,'Negative':-11,'News':2}
            
            tweet_text2 = st.text_area('Enter column to classify')

            
            if st.button('Classify'):

                vect_text = tweet_cv.transform([tweet_text2]).to_array()
                if model2 == 'LinearSVC':
                    predictor = load_pickle("resources/models/LinearSVC.pkl")
                    prediction = predictor.predict(vect_text)
                    
                elif model2 == 'KNeighborsClassifier':
                    predictor = load_pickle("resources/models/KNeighborsClassifier.pkl")
                    prediction = predictor.predict(vect_text)
                    
                elif model2 == 'DecisionTreeClassifier':
                    predictor = load_pickle("resources/models/DecisionTreeClassifier.pkl")
                    prediction = predictor.predict(vect_text)
                    
                elif model2 == 'RandomForestClassifier':
                    predictor = load_pickle("resources/models/RandomForestClassifier.pkl")
                    prediction = predictor.predict(vect_text)

                elif model2 == 'ComplementNB':
                    predictor = load_pickle("resources/models/ComplementNB.pkl")
                    prediction = predictor.predict(vect_text)

                elif model2 == 'MultinomialNB':
                    predictor = load_pickle("resources/models/MultinomialNB.pkl")
                    prediction = predictor.predict(vect_text)

                elif model2 == 'AdaBoostClassifier':
                    predictor = load_pickle("resources/models/AdaBoostClassifier.pkl")
				
                
                final_pred2 = dict_labels2[prediction]
                st.success("Tweets Categorized as:: {}".format(final_pred2))

                



    #Home page
    if selection == 'Home':
        
        welcome_image = Image.open('resources/imgs/image.jpg')
        st.image(welcome_image,use_column_width=True)
        st.subheader('Classifying Climate Change based Tweets')

     # Eploratory data analysis   
    if selection =='EDA':
        st.title('Exploratory Data Analysis')
        if st.checkbox('Count of Tweets per Sentiment'):
            st.markdown('Sentiment by class')
            st.markdown('**Positive(1)**: These are tweets that for climate change')
            st.markdown('**Neutral(0)**: These are tweets that are neither for or against climate change')
            st.markdown('**News(2)**: These tweets are about news that report on climate change')
            st.markdown('**Negative(-1)**: These tweets are against climate change')
            st.image(Image.open('resources/imgs/countSentimentPerT.png'),caption='Count of Sentiment', use_column_width=True)
            st.markdown('This figure shows an imbalance between the different sentiments')
            st.markdown('With positive being the highest followed by news, neutral tweets and less negative tweets')
            st.markdown('Most people tweet positively about climate change')
        #Distribution
        if st.checkbox('Distribution of words'):
           st.markdown('Sentiment Distribution')
           st.image(Image.open('resources/imgs/DistributionWords.png'),caption='Class Distribution by words', use_column_width=True)
           #st.markdown('')
           st.image(Image.open('resources/imgs/BarplotWordsperSent.png'),caption='Box plot of each sentiment', use_column_width=True)

        if st.checkbox('Distribution of length by Sentiment'):
            st.markdown('Sentiment message length Dustribution')    
            st.image(Image.open('resources/imgs/DistributionLengthSent.png'),caption='Class Distribution by length', use_column_width=True)
            st.image(Image.open('resources/imgs/Avg_len_sent.png'),caption='Bar plot of each Sentiment avg_length', use_column_width=True)
        
        if st.checkbox('Bar plot and wordClouds: Most common words'):
            st.markdown('Most common words')
            choices = st.radio("Choose an option", ("Don't believe in man-made climate change (Negative)", "neither supports nor against the man-mde climate change", "Believes in man-made climate change", "News related to climate change"))
            if choices == "Don't believe in man-made climate change (Negative)":
                st.image(Image.open('resources/imgs/commonNegative.png'), caption='Most common words for negative tweet sentiments', use_column_width=True)
                st.image(Image.open('resources/imgs/cloudNegative.png'), caption='Most common words for negative tweet sentiments', use_column_width=True)

            if choices == "neither supports nor against the man-mde climate change":
                st.image(Image.open('resources/imgs/commonNeutral.png'), caption='Most common words with neutral tweet sentiments')
                st.image(Image.open('resources/imgs/cloudNeutral.png'), caption='Most common words for neutral tweet sentiments', use_column_width=True)

            if choices == "Believes in man-made climate change":
                st.image(Image.open('resources/imgs/commonPositive.png'), caption="Most common words for positive tweet sentiments", use_column_width=True)
                st.image(Image.open('resources/imgs/cloudPositive.png'), caption='Most common words for positive tweet sentiments', use_column_width=True)

            if choices == "News related to climate change":
                st.image(Image.open('resources/imgs/commonNews.png'), caption="Most common words for news tweet sentiments", use_column_width=True)
                st.image(Image.open('resources/imgs/cloudNews.png'), caption='Most common words for news tweet sentiments', use_column_width=True)

            if choices == 'general common words':
                st.image(Image.open('resources/imgs/CommonOverBar.png'), caption='Most common words tweet sentiments', use_column_width=True)
                st.image(Image.open('resources/imgs/cloudOverall.png'), caption='Most common words tweet sentiments', use_column_width=True)   


    
    ##contact page
    if selection == 'Lets connect!':
        st.markdown('****************************************************************')
        st.subheader("Have questions? We are an email away to answer your questions")
        st.markdown('***************************************************************')
        st.write("Noxolo: wendyngcobo98@gmail.com")
        st.write("Sabelo: isabelomakhanya@gmail.com")
        st.write("Morgan: letlhogonolomorgan69@gmail.com")
        st.write("Tebogo: mrtjsambo@gmail.com")
        st.write("Sergio: sergiomornayseptember@gmail.com")
        st.write("Vuyelwa: vuyelwaf22@gmail.com")
        
        st.markdown('****************************************************************')
        image = Image.open('resources/imgs/EDSA_logo.png')
        st.image(image, caption='TS5_EDSA_2021', use_column_width=True)



if __name__ == '__main__':
	main()