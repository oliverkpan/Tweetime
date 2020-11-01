import datetime as dt
import re

import pandas as pd
import numpy as np
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier
from twitterscraper import query_tweets

from PIL import Image


def main(): 

    st.sidebar.title('Tweetime')
    st.sidebar.selectbox(
    "How was your experience?",
    ("Great", "Neutral", "Bad"))


    # Set page title
    st.title('Twitter Sentiment Analysis')

    html_temp = """ 
    <div style ="background-color:white;padding:8px"> 
    <h1 style ="color:black;text-align:left;">Please enter tweets below</h1> 
    </div> 
    """

    image = Image.open('/Users/oliverpan/Desktop/twitter.jpg')
    st.image(image)
    

    st.markdown(html_temp, unsafe_allow_html = True) 

    # Load classification model
    with st.spinner('Loading classification model...'):
        classifier = TextClassifier.load('/Users/oliverpan/Desktop/tweets/final-model.pt')

    allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
    punct = '!?,.@#'
    maxlen = 280

    def preprocess(text):
        return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])[:maxlen]


    tweet_input = st.text_input('Tweet:')

    if tweet_input != '':
        # Pre-process tweet
        sentence = Sentence(preprocess(tweet_input))

        # Make predictions
        with st.spinner('Predicting...'):
            classifier.predict(sentence)

        # Show predictions
        label_dict = {'0': 'Negative', '4': 'Positive'}

        if len(sentence.labels) > 0:
            st.write('Prediction:')
            st.write(label_dict[sentence.labels[0].value] + ' with ',
                    sentence.labels[0].score*100, '% confidence')
            st.balloons()

if __name__=='__main__': 
    main() 