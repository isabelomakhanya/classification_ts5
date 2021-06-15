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
	options = [ "Information", "EDA","Data visualisation", "predict tweet", "Lets connect!"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("Modelling tweet sentiments for climate change")
		st.write('Let the tweet spy game begin hahhaa!!! ')
		# You can read a markdown file from supporting resources folder
		st.markdown(""" This Machine learning model helps companies
		classify tweets about climate change to get some insights on
		whether or not an individual believes in climate change based 
		on their tweet(s) and this helps them derive better marketing
		strategies in the future.
		""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Predict tweet":
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

	if selection == "EDA":
		st.subheader("Exploratory Data Analysis")

		
		df = raw
		st.dataframe(df.head())

		if st.checkbox("Show Shape"):
			st.write(df.shape)

		if st.checkbox("Show Columns"):
			all_columns = df.columns.to_list()
			st.write(all_columns)

		if st.checkbox("Summary"):
			st.write(df.describe())

		if st.checkbox("Show Selected Columns"):
			selected_columns = st.multiselect("Select Columns",all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.checkbox("Show Value Counts"):
			st.write(df.iloc[:,-1].value_counts())

		if st.checkbox("Correlation Plot(Matplotlib)"):
			plt.matshow(df.corr())
			st.pyplot()

		if st.checkbox("Correlation Plot(Seaborn)"):
			st.write(sns.heatmap(df.corr(),annot=True))
			st.pyplot()


		if st.checkbox("Pie Plot"):
			all_columns = df.columns.to_list()
			column_to_plot = st.selectbox("Select 1 Column",all_columns)
			pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()
        
        


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
