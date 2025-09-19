import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
import time

st.set_page_config(layout="wide", page_title="eConsultation Sentiment Analyzer")

st.title("📊 eConsultation Sentiment Analyzer MVP")
st.write("Upload stakeholder comments and get **Sentiment Analysis + Summaries + Word Cloud + Geo Heatmap**")

# ---------- File Upload ----------
uploaded = st.file_uploader("Upload CSV file", type=['csv'])
use_demo = st.checkbox("Use demo data (if you don’t have a file)")

if use_demo:
    demo = {
        "id":[1,2,3,4],
        "comment":[
            "I strongly oppose clause 3. It removes consumer rights.",
            "This is a good step; just clarify timelines and implementation.",
            "Neutral - needs better structure but I like the intent.",
            "The draft is terrible and harmful to small businesses."
        ],
        "location":["Mumbai","New Delhi","Bengaluru","Pune"]
    }
    df = pd.DataFrame(demo)
elif uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV or check demo mode to continue.")
    st.stop()

if "comment" not in df.columns:
    st.error("CSV must contain a 'comment' column.")
    st.stop()

# ---------- Load AI Models ----------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization")
    return sentiment, summarizer

with st.spinner("Loading AI models..."):
    sentiment_pipe, summarizer = load_models()

# ---------- Run Analysis ----------
if st.button("🔍 Run Analysis"):
    with st.spinner("Processing comments..."):
        df['comment'] = df['comment'].astype(str)

        # Sentiment
        sentiments = sentiment_pipe(df['comment'].tolist(), truncation=True)
        df['sentiment'] = [s['label'] for s in sentiments]
        df['confidence'] = [s['score'] for s in sentiments]

        # Summaries
        def summarize_text(text):
            try:
                if len(text.split()) > 20:
                    out = summarizer(text, max_length=60, min_length=10, truncation=True)
                    return out[0]['summary_text']
                return text
            except:
                return text
        df['summary'] = df['comment'].apply(summarize_text)

    st.success("Analysis complete!")

    # ---------- Results Table ----------
    st.subheader("📄 Results Preview")
    st.dataframe(df[['id','comment','summary','sentiment','confidence']].head(20))

    # ---------- Sentiment Distribution ----------
    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())

    # ---------- Word Cloud ----------
    st.subheader("☁️ Word Cloud of Comments")
    text = " ".join(df['comment'].tolist())
    wc = WordCloud(stopwords=STOPWORDS, background_color="white", width=900, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # ---------- Geo Heatmap ----------
    st.subheader("🗺 Geographic Heatmap")
    geo_df = pd.DataFrame()
    if 'lat' in df.columns and 'lon' in df.columns:
        geo_df = df.dropna(subset=['lat','lon'])
    elif 'location' in df.columns:
        geolocator = Nominatim(user_agent="econsult_mvp")
        coords = []
        for loc in df['location'].dropna().unique()[:20]:  # limit for hackathon
            try:
                time.sleep(1)
                p = geolocator.geocode(loc, timeout=10)
                if p: coords.append((p.latitude, p.longitude))
            except: pass
        geo_df = pd.DataFrame(coords, columns=['lat','lon'])

    if not geo_df.empty:
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)  # India center
        HeatMap(geo_df[['lat','lon']].values.tolist()).add_to(m)
        st_folium(m, width=800, height=500)
    else:
        st.info("No location data found. Add 'location' or 'lat/lon' column to see the heatmap.")

    # ---------- Download ----------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results CSV", csv, "results.csv", "text/csv")
