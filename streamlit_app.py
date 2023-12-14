import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob  # for sentiment analysis

# Load data
@st.cache
def load_data():
    with open('Trailor_feedback.json', 'r') as file:
        data = json.load(file)['feedback_results']
    return pd.json_normalize(data)

data = load_data()

# Sentiment analysis on text feedback
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

data['sentiment'] = data['text_feedback'].apply(analyze_sentiment)

# Streamlit layout
st.title("Itinerary Feedback Analysis Dashboard")

# Overview statistics
st.header("Overview Statistics")
st.markdown(f"**Total Itineraries:** {len(data)}")
st.markdown(f"**Liked Itineraries:** {data['liked'].sum()}")
st.markdown(f"**Average Budget:** {data['itinerary.budget'].mean():.2f}")

# Filterable table of itinerary details
st.header("Itinerary Details")
origin_filter = st.selectbox("Select Origin", options=data['itinerary.origin'].unique())
filtered_data = data[data['itinerary.origin'] == origin_filter]
st.dataframe(filtered_data)

# Visualization
st.header("Visualizations")

# Bar chart for count of itineraries per destination
st.subheader("Itineraries Count per Destination")
destination_count = data['itinerary.destination'].value_counts()
st.bar_chart(destination_count)

# Pie chart for vibes distribution
st.subheader("Vibes Distribution")
vibes_count = data['itinerary.vibe'].value_counts()
plt.pie(vibes_count, labels=vibes_count.index, autopct='%1.1f%%')
st.pyplot(plt)

# Histogram of budget distribution
st.subheader("Budget Distribution")
sns.histplot(data['itinerary.budget'], kde=True)
st.pyplot(plt)

# Time series plot of itineraries based on departure
st.subheader("Itineraries Over Time")
data['itinerary.departure'] = pd.to_datetime(data['itinerary.departure'], format='%d-%m-%y')
sns.lineplot(x='itinerary.departure', y='liked', data=data)
st.pyplot(plt)

# Sentiment Analysis
st.header("Sentiment Analysis of Feedback")
st.bar_chart(data['sentiment'])

# Run the app with `streamlit run your_script.py`
