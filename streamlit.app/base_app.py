"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy and Team TS5.

    Note:
	
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import plotly.express as px
from plotly import graph_objects as go

# Data dependencies
import pandas as pd
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
news_vectorizer = open("resources/models/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# cleaning
df = raw.copy()

#removing noise using lemma and stemma
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# filtered words
def preprocess_fil(sentence):
    '''function removes noise/cleans text data'''
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>') 
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

# stemma
def preprocess_stemm(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(stem_words)
#lemma
def preprocess_lemm(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words ]
    return " ".join(lemma_words)


# cleaning the text messages and creates a new column named 'clean_message'
df['clean_message']=df['message'].map(lambda s:preprocess_fil(s))

# lemmatizes the cleaned text data and creates new column named 'Lemma"
df['Lemma']=df['message'].map(lambda s:preprocess_lemm(s))

# stemmatize the cleaned text data and creates a new column named 'Stemm'
df['stemm']=df['message'].map(lambda s:preprocess_stemm(s)) 




# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit"""

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	
	st.subheader("Climate change tweet classification")
	image = Image.open('resources/imgs/image.jpg')
	st.image(image, caption='Tweet Sentiments', use_column_width=True)

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = [ "Information", "EDA","predict tweet", "Lets connect!"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info(""" This Machine learning model helps companies
		classify tweets about climate change to get some insights on
		whether or not an individual believes in climate change based 
		on their tweet(s) and this helps them derive better marketing
		strategies in the future.
		""")
		st.write('Let the tweet spy game begin hahhaa!!! ')
		# You can read a markdown file from supporting resources folder
		#st.markdown()

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	elif selection == "predict tweet":
		st.info("Make tweet Predictions with ML Models of your choice")

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Tweet","Type Here")


		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/models/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	elif selection == "EDA":
		st.subheader("Exploratory Data Analysis")
		df = raw.copy()
		# Labeling the target

        df['class_label'] = [['Negative(-1)', 'Neutral(0)', 'Positive(1)', 'News(2)'][x+1] for x in df['sentiment']]
		dist = df.groupby('class_label').count()['clean_message'].reset_index().sort_values(by='clean_message',ascending=False)
		st.dataframe(df.head())

		plt.figure(figsize=(12,6))
		sns.countplot(x='sentiment',data=df, palette='Blues_d')
		plt.title('Count of Sentiments')
		st.pyplot()

		   # average length of words overall
        df['clean_message'].str.split().\
            apply(lambda x : [len(i) for i in x]).\
            map(lambda x : np.mean(x)).hist()
        plt.title('Avg number of words used per tweet')
        plt.xlabel('Number of words per tweet')
        plt.ylabel('Count of Tweets')
		
	    df['length_tweet'] = df['clean_message'].apply(len)
		h = sns.FacetGrid(df,col = 'class_label')
		h.map(plt.hist,'length_tweet')
		plt.show()

		#Box plot visual of distribution between length of tweet vs class label
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['class_label'], 
		y=df.clean_message.str.split().apply(len),
		data=df,
		palette="Blues")

        plt.title('No of Words per Tweet by Sentiment Class')
        plt.xlabel('Sentiment Class')
        plt.ylabel('Word Count per Tweet');


        fig = go.Figure(go.Funnelarea(
        text = dist.class_label,
        values = dist.clean_message,
        title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}))
        fig.show()

		plt.figure(figsize=(12,6))
		sns.barplot(x='sentiment', y=df['message'].apply(len) ,data = df, palette='Blues_d')
		plt.ylabel('avg_Length')
		plt.xlabel('Sentiment')
		plt.title('Average Length of Message by Sentiment')
        st.pyplot()

        # most common words in tweet messages
		fig = px.bar(new, x="count", y="Common_words", color_discrete_sequence =['']*len(df), title='Commmon Words in tweet messages', orientation='h', 
             width=600, height=600)
        fig.show()
		
		
    elif selection == 'Lets connect!':

		st.subheader("Have questions? We are an email away to answer your questions")

		st.write("Noxolo: wendyngcobo98@gmail.com")
		st.write("Sabelo: isabelomakhanya@gmail.com")
		st.write("Morgan: letlhogonolomorgan69@gmail.com")
		st.write("Tebogo: mrtjsambo@gmail.com")
		st.write("Sergio: sergiomornayseptember@gmail.com")
		st.write("Vuyelwa: vuyelwaf22@gmail.com")
		image = Image.open('resources/imgs/EDSA_logo.png')
		st.image(image, caption='TS5_EDSA_2021', use_column_width=True)


    #st.write()
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
