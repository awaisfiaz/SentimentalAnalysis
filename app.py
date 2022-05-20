from sys import displayhook
from nltk.corpus import sentiwordnet as swn
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from googleapiclient.discovery import build  # for api
import nltk
import time
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text


pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB,
            'N': wordnet.NOUN, 'R': wordnet.ADV}


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


wordnet_lemmatizer = WordNetLemmatizer()


def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity
    # function to calculate polarity


def getPolarity(review):
    return TextBlob(review).sentiment.polarity


# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


youTubeApiKey = "AIzaSyC6TWa4miwsTZ7yhD44z18S6sWRwFWwBNg"
youtube = build('youtube', 'v3', developerKey=youTubeApiKey)

st.set_page_config(page_title="TubeStack ", page_icon=":zap:", layout="wide")
hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Find Sentimental Response On Your Video 🔥")
st.write("\n")
st.subheader("➡️ Dont waste your time analyzing each and very comment to find the viewers sentiment! ")
st.text("✔️ Analyze the sentiment of the comments and generate the sentimental response of the overall video")
st.text("and shows top positive and negative comments on their videos based on their polarity and subjectivity")
video_id = \
    st.text_input('YOUTUBE VIDEO LINK   (eg -> https://www.youtube.com/watch?******************* )'
                    )
    
def sentimentalAnalysis(youtube):    
    parsed = video_id.split("v=")
    Id = parsed[1]

    video_id_pop = []

    comments = []
    comment_id = []
    reply_count = []
    like_count = []

    comments_temp = []
    comment_id_temp = []
    reply_count_temp = []
    like_count_temp = []

    nextPage_token = None
    while 1:
        response = youtube.commentThreads().list(
            part='snippet,replies', videoId=Id, maxResults=100, order='relevance',
            textFormat='plainText', pageToken=nextPage_token).execute()
        # for i in response['nextPageToken']:
        nextPage_token = response.get('nextPageToken')
        for i in response['items']:
            comments_temp.append(i['snippet']['topLevelComment']
                                ['snippet']['textDisplay'])
            comment_id_temp.append(i['snippet']['topLevelComment']['id'])
            reply_count_temp.append(i['snippet']['totalReplyCount'])
            like_count_temp.append(
                i['snippet']['topLevelComment']['snippet']['likeCount'])
            comments.extend(comments_temp)
            comment_id.extend(comment_id_temp)
            reply_count.extend(reply_count_temp)
            like_count.extend(like_count_temp)

            video_id_pop.extend([video_id]*len(comments_temp))


        if nextPage_token is None:
            break
    output_dict = {

        'Comment': comments,

    }
    output_df = pd.DataFrame(output_dict, columns=output_dict.keys())
    output_df.head()

    # to remove duplicate commments if any
    duplicates = output_df[output_df.duplicated('Comment')]


    unique_df = output_df.drop_duplicates(subset=['Comment'])
    print(unique_df.shape)
    # unique_df.to_csv("extracted_comments.csv", index=False)
    mydata = pd.DataFrame(unique_df)
    # mydata = pd.read_csv('extracted_comments.csv', error_bad_lines=False)
    mydata['Cleaned Comment'] = mydata['Comment'].apply(clean)
    mydata['POS tagged'] = mydata['Cleaned Comment'].apply(token_stop_pos)
    mydata['Lemma'] = mydata['POS tagged'].apply(lemmatize)

    fin_data = pd.DataFrame(mydata[['Cleaned Comment','POS tagged', 'Lemma']])
    fin_data['Subjectivity'] = fin_data['Lemma'].apply(getSubjectivity)
    fin_data['Polarity'] = fin_data['Lemma'].apply(getPolarity)
    fin_data['Result'] = fin_data['Polarity'].apply(analysis)
    displaydata = pd.DataFrame(fin_data[['Cleaned Comment', 'Subjectivity', 'Polarity', 'Result']])
    tb_counts = fin_data.Result.value_counts() / len(displaydata) * 100
    st.subheader("Overall Comments Sentimental Response Percentage %")
    st.table(tb_counts)
    
    # Positive Comments
    
    st.subheader("✔️ Top Positive Comments on your Video")
    select_positive = displaydata.loc[displaydata['Result'] == 'Positive']
    topPositive = select_positive.head(15)
    sorted = topPositive.sort_values(by = 'Polarity', ascending = False)
    final = sorted.set_index('Cleaned Comment')
    st.table(final)
    
    # Negative Comments
    
    st.subheader("❌ Top Negative Comments on your Video")
    select_negative = displaydata.loc[displaydata['Result'] == 'Negative']
    topNegative = select_negative.head(15)
    sorted = topNegative.sort_values(by = 'Polarity', ascending = False)
    final = sorted.set_index('Cleaned Comment')
    st.table(final)
    
    
if(len(video_id) != 0):
    time.sleep(3)
    st.text("Here is the Sentimental Response details of the commentator's on your video ⬇️")
    st.write("\n")
    st.write("\n")
    response = sentimentalAnalysis(youtube)
    print(response)