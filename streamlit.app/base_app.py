"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

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
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
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
clean_df = raw.copy()

def remove_punctuations(msg):
    msg = str(msg).lower()
    msg = re.sub('\[.*?\]', '', msg)
    msg = re.sub('https?://\S+|www\.\S+', '', msg)
    msg = re.sub('<.*?>+', '', msg)
    msg = re.sub('[%s]' % re.escape(string.punctuation), '', msg)
    msg = re.sub('\n', '', msg)
    msg = re.sub('\w*\d\w*', '', msg)
    msg = re.sub('rt','',msg)
    return msg

clean_df['clean_message'] = clean_df['message'].apply(lambda x:remove_punctuations(x))

#Remove stop words
clean_df['clean_message'] = clean_df['clean_message'].apply(lambda x: ' '.join([a for a in x.split() if len(a)>3]))


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
	options = [ "Information", "EDA", "Data visualisation", "predict tweet", "Lets connect!"]
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
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

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
		df['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in df['sentiment']]

		st.dataframe(df.head())

		if st.checkbox("Show Shape"):
			st.write(df.shape)

		elif st.checkbox("Show Columns"):
			all_columns = df.columns.to_list()
			st.write(all_columns)

		elif st.checkbox("Summary"):
			st.write(df.describe())

		


		elif st.checkbox("visuals"):
			plt.figure(figsize=(12,6))
			sns.countplot(x='sentiment',data=df, palette='Greens')
			st.pyplot()

			
			pie_plot = df['sentiment'].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()
			
			plt.figure(figsize=(12,6))
			sns.barplot(x='sentiment', y=df['message'].apply(len) ,data = df, palette='inferno')
			plt.ylabel('avg_Length')
			plt.xlabel('Sentiment')
			plt.title('Average Length of Message by Sentiment')
			#plt.show()            
			st.pyplot()
			
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
