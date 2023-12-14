import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import json

# Load data
@st.cache
def load_data():
    with open('Trailor_feedback.json', 'r') as file:
        data = json.load(file)['feedback_results']
    return pd.json_normalize(data)

data = load_data()

# Sentiment analysis on text feedback
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

data['sentiment'] = data['text_feedback'].apply(analyze_sentiment)

# Prepare the data for visualizations
def prepare_chart_data(series, top_n=5):
    chart_data = series.value_counts().head(top_n)
    chart_data_percent = (chart_data / chart_data.sum()) * 100
    return pd.DataFrame(chart_data_percent).reset_index().rename(columns={'index': 'category', series.name: 'percent'})

# Convert to datetime
data['itinerary.departure'] = pd.to_datetime(data['itinerary.departure'], format='%d-%m-%y')

# Streamlit layout
st.title("Itinerary Feedback Analysis Dashboard")

with st.container():
    st.header("Overview Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Total Itineraries")
        st.write(f"{len(data)}")
    with col2:
        st.subheader("Liked Itineraries")
        st.write(f"{data['liked'].sum()}")
    with col3:
        st.subheader("Average Budget")
        st.write(f"{data['itinerary.budget'].mean():.2f}")

with st.expander("Itinerary Details"):
    origin_filter = st.selectbox("Select Origin", options=['All'] + list(data['itinerary.origin'].unique()))
    if origin_filter != 'All':
        filtered_data = data[data['itinerary.origin'] == origin_filter]
    else:
        filtered_data = data
    st.dataframe(filtered_data)

st.header("Visualizations")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Liked/Disliked Itineraries")
        like_dislike_counts = data['liked'].value_counts(normalize=True) * 100
        st.bar_chart(like_dislike_counts)
    with col2:
        st.subheader("Sentiment Analysis of Feedback")
        st.bar_chart(data['sentiment'])

with st.container():
    st.subheader("Top Common Destinations")
    top_n_dest = st.slider("Select number of top destinations", 1, 20, 5, key='slider_dest')
    top_destinations_data = prepare_chart_data(data['itinerary.destination'], top_n=top_n_dest)
    st.bar_chart(top_destinations_data.set_index('category'))

    st.subheader("Top Common Origins")
    top_n_orig = st.slider("Select number of top origins", 1, 20, 5, key='slider_orig')
    top_origins_data = prepare_chart_data(data['itinerary.origin'], top_n=top_n_orig)
    st.bar_chart(top_origins_data.set_index('category'))

    st.subheader("Top Vibes")
    top_n_vibes = st.slider("Select number of top vibes", 1, 20, 5, key='slider_vibes')
    top_vibes_data = prepare_chart_data(data['itinerary.vibe'], top_n=top_n_vibes)
    st.bar_chart(top_vibes_data.set_index('category'))

with st.container():
    st.subheader("Budget Distribution")
    sns.histplot(data['itinerary.budget'], kde=True)
    st.pyplot(plt)

    st.subheader("Itineraries Over Time")
    sns.lineplot(x='itinerary.departure', y='liked', data=data)
    st.pyplot(plt)
