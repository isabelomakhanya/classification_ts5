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
news_vectorizer = open("resources/models/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


df = raw.copy()

#removing noise using lemma and stemma
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# stemma

@st.cache(suppress_st_warning=True)
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

# stemmatize the cleaned text data and creates a new column named 'Stemm'
df['stemm']=df['message'].map(lambda s:preprocess_stemm(s))

@st.cache(suppress_st_warning=True)
def main():
    """Tweets classifier App with Streamlit"""

    st.title('Tweet Sentiment Classifier')

    
    image = Image.open('resources/imgs/image.jpg')

    st.image(image, caption='Tweet Sentiments', use_column_width=True)

    st.subheader('Climate Change Belief Analysis: Based on Tweets')
    

    ##creating a sidebar for selection purposes


    options = ['Information', 'EDA', 'Predict tweet', 'Lets Connect!']

    selection = st.sidebar.radio('Go to', options)

    



    # Building information page

    if selection == 'Information':
        st.info(""" This Machine learning model helps companies
		classify tweets about climate change to get some insights on
		whether or not an individual believes in climate change based 
		on their tweet(s) and this helps them derive better marketing
		strategies in the future.
		""")
        st.write('Let the tweet spy game begin hahhaa!!! ')
        


        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):
            st.write(raw[['sentiment', 'message']].head())

    ## Exploratory data analysis page

    if selection == 'EDA':
        st.subheader("Exploratory Data Analysis")


        # labelling target
        df['class_label'] = [['Negative(-1)', 'Neutral(0)', 'Positive(1)', 'News(2)'][x+1] for x in df['sentiment']]
        dist = df.groupby('class_label').count()['stemm'].reset_index().sort_values(by='stemm',ascending=False)

        
        st.write('The Bar chart of count per sentiment')

        #bar plot of the count of each sentiment
        plt.figure(figsize=(12,6))
        sns.countplot(x='sentiment',data=df, palette='Blues_d')
        plt.title('Count of Sentiments')
        st.pyplot()
        
        # average length of words overall
        df['stemm'].str.split().\
            apply(lambda x : [len(i) for i in x]).\
                map(lambda x: np.mean(x)).hist()
        plt.title('avg number of words used per tweet')
        plt.xlabel('Number of words per tweet')
        plt.ylabel('Count of Tweets')  
        st.pyplot()      

        # distribution of each of length of the tweets
        df['length_tweet'] = df['stemm'].apply(len)
        h = sns.FacetGrid(df, col ='class_label')
        h.map(plt.hist,'length_tweet')
        st.pyplot()

        #Box plot visual of distribution between length of tweet vs sentiment
        plt.figure(figsize =(10,6))
        sns.boxplot(x=df['sentiment'],
        y=df.stemm.str.split().apply(len),
        data=df,
        palette='Blues')
        
        plt.title('No of words per Tweet by sentiment Class')
        plt.xlabel('Sentiment Class')
        plt.ylabel('Word Count per Tweet')
        st.pyplot()

        #Funnel chart of proportion of each sentiment
        fig = go.Figure(go.Funnelarea(
            text = dist.class_label,
            values = dist.stemm,
            title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}))
        st.pyplot(fig)

        #bar plot for average length of messages by sentiments
        plt.figure(figsize=(12,6))
        sns.barplot(x='sentiment', y=df['message'].apply(len), data=df, palette='Blues_d')
        plt.ylabel('avg_Length')
        plt.xlabel('Sentiment')
        plt.title('Average Length of Message by Sentiment')
        st.pyplot()

        #most common words in tweet messages bar plt
        df['new_lis'] = df['stemm'].apply(lambda x:str(x).split())
        words = Counter([item for sublist in df['new_lis'] for item in sublist])
        new = pd.DataFrame(words.most_common(20))

        fig = px.bar(new, x='count', y="common_words", 
        color_discrete_sequence=['']*len(df),
        title ='Commmon Words in tweet messages', orientation='h',
        width=600, height=600)
        st.pyplot(fig)

        #word cloud of most common words
        train_msg = " ".join(tweet for tweet in df.stemm)
        train_wordcloud = WordCloud(max_font_size=250,
        background_color="white",
        width=1500,
        height=700,
        collocations=False,
        colormap='Paired').generate(train_msg)
        plt.figure(figsize=(16, 10))
        plt.imshow(train_wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot()
       

     

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

                
                clean_text1 = preprocess_stemm(tweet_text1) 
                vect_text = tweet_cv.transform([clean_text1]).toarray()

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
            #input = st.file_uploader("Choose a CSV file", type="csv")
            #if input is not None:
                #input = pd.read_csv(input)


            #upload_dataset = st.checkbox('See uploaded dataset')
            #if upload_dataset:
                #st.dataframe(input.head())

            tweet_text2 = st.text_area('Enter column to classify')

            
            if st.button('Classify'):

                # Transforming user input with vectorizer
                clean_text2 = df[tweet_text2].apply(preprocess_stemm) #passing the text through the stemma function
                vect_text = tweet_cv.transform([clean_text2]).toarray()
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

                


    ##contact page
    if selection == 'Lets connect!':

        st.subheader("Have questions? We are an email away to answer your questions")
        st.write("Noxolo: wendyngcobo98@gmail.com")
        st.write("Sabelo: isabelomakhanya@gmail.com")
        st.write("Morgan: letlhogonolomorgan69@gmail.com")
        st.write("Tebogo: mrtjsambo@gmail.com")
        st.write("Sergio: sergiomornayseptember@gmail.com")
        st.write("Vuyelwa: vuyelwaf22@gmail.com")

        image = Image.open('resources/imgs/EDSA_logo.png')
        st.image(image, caption='TS5_EDSA_2021', use_column_width=True)

if __name__ == '__main__':
	main()
