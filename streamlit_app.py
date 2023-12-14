import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import json
import numpy as np

# Load data
@st.experimental_memo
def load_data():
    with open('Trailor_feedback.json', 'r') as file:
        data = json.load(file)['feedback_results']
    df = pd.json_normalize(data)
    df['itinerary.departure'] = pd.to_datetime(df['itinerary.departure'], format='%d-%m-%y')
    return df

data = load_data()

# Prepare the data for visualizations
def prepare_chart_data(series, top_n=5):
    chart_data = series.value_counts().head(top_n)
    chart_data_percent = (chart_data / chart_data.sum()) * 100
    return chart_data_percent

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
    origin_filter = st.selectbox("Select Origin", options=['All'] + sorted(data['itinerary.origin'].unique()))
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
        like_dislike_data = pd.DataFrame(like_dislike_counts).reset_index()
        like_dislike_data.columns = ['Response', 'Percentage']
        st.bar_chart(like_dislike_data.set_index('Response'))

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Common Destinations")
        top_n_dest = st.slider("Select number of top destinations", 1, 20, 5, key='slider_dest')
        top_destinations_data = prepare_chart_data(data['itinerary.destination'], top_n=top_n_dest)
        top_destinations_chart_data = pd.DataFrame(top_destinations_data).reset_index()
        top_destinations_chart_data.columns = ['Destination', 'Percentage']
        st.bar_chart(top_destinations_chart_data.set_index('Destination'))

    with col2:
        st.subheader("Top Common Origins")
        top_n_orig = st.slider("Select number of top origins", 1, 20, 5, key='slider_orig')
        top_origins_data = prepare_chart_data(data['itinerary.origin'], top_n=top_n_orig)
        top_origins_chart_data = pd.DataFrame(top_origins_data).reset_index()
        top_origins_chart_data.columns = ['Origin', 'Percentage']
        st.bar_chart(top_origins_chart_data.set_index('Origin'))

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Vibes")
        top_n_vibes = st.slider("Select number of top vibes", 1, 20, 5, key='slider_vibes')
        top_vibes_data = prepare_chart_data(data['itinerary.vibe'], top_n=top_n_vibes)
        top_vibes_chart_data = pd.DataFrame(top_vibes_data).reset_index()
        top_vibes_chart_data.columns = ['Vibe', 'Percentage']
        st.bar_chart(top_vibes_chart_data.set_index('Vibe'))

    with col2:
        st.subheader("Budget Distribution")
        # Create a histogram with KDE for the budget using Plotly
        fig = px.histogram(data, x='itinerary.budget', nbins=20, marginal='violin', title='Distribution of Budgets')
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

