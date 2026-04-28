import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from prophet import Prophet
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="University Social Media Intelligence System",
    layout="wide"
)

# ---------------- CLEAN STYLE ----------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: white;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("University Social Media Intelligence System")
st.markdown("""
Comparative Analysis of University Social Media Performance  
Machine Learning-Based Sentiment Classification and Predictive Analytics
""")

@st.cache_data
def load_data():
    return pd.read_csv("university_social_media_comparative_dataset_v4.csv")

df = load_data()

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("Global Filters")
selected_uni = st.sidebar.selectbox(
    "Select University",
    ["All"] + list(df["university"].unique())
)

if selected_uni != "All":
    df = df[df["university"] == selected_uni]

# ---------------- EXECUTIVE SUMMARY ----------------
st.markdown("## Executive Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", f"{len(df):,}")
col2.metric("Active Users", df["user_id"].nunique())
col3.metric("Universities", df["university"].nunique())
col4.metric("Positive Sentiment %",
            f"{(df['sentiment_label']=='positive').mean()*100:.1f}%")

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment Analysis",
    "Machine Learning Evaluation",
    "Network Analysis",
    "Forecasting"
])

# =====================================================
# TAB 1 — SENTIMENT ANALYSIS
# =====================================================
with tab1:

    # Donut Chart
    sentiment_counts = df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.55,
        template="plotly_dark",
        color_discrete_sequence=["#1f77b4","#6baed6","#9ecae1"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sentiment Trend
    df['date'] = pd.to_datetime(df['date'])
    trend_data = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')

    fig_trend = px.line(
        trend_data,
        x='date',
        y='count',
        color='sentiment_label',
        template="plotly_dark",
        color_discrete_sequence=["#1f77b4","#6baed6","#9ecae1"]
    )

    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Posts"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # Textual Frequency Analysis (Word Cloud)
    st.markdown("### Textual Frequency Analysis")

    text_data = " ".join(df["post_text"].astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Blues"
    ).generate(text_data)

    fig_wc, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig_wc)

    # Comparative Engagement Profile (Radar)
    st.markdown("### Comparative Engagement Profile")

    radar_df = df.groupby("university").agg({
        "likes":"mean",
        "retweets":"mean"
    }).reset_index()

    categories = ["Average Likes", "Average Retweets"]

    fig_radar = go.Figure()

    for i in range(len(radar_df)):
        fig_radar.add_trace(go.Scatterpolar(
            r=[radar_df.loc[i,"likes"],
               radar_df.loc[i,"retweets"]],
            theta=categories,
            fill='toself',
            name=radar_df.loc[i,"university"]
        ))

    fig_radar.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True))
    )

    st.plotly_chart(fig_radar, use_container_width=True)

# =====================================================
# TAB 2 — MACHINE LEARNING
# =====================================================
with tab2:

    X = df["post_text"]
    y = df["sentiment_label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Logistic Regression Accuracy", f"{accuracy:.2f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig_cm)

    # Feature Importance
    st.markdown("### Top Influential Terms")

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_features = np.argsort(coefs)[-10:]

    fig_feat = px.bar(
        x=feature_names[top_features],
        y=coefs[top_features],
        template="plotly_dark",
        color_discrete_sequence=["#1f77b4"]
    )

    fig_feat.update_layout(
        xaxis_title="Terms",
        yaxis_title="Model Coefficient"
    )

    st.plotly_chart(fig_feat, use_container_width=True)

# =====================================================
# TAB 3 — NETWORK ANALYSIS
# =====================================================
with tab3:

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(row['user_id'])

    for i in range(0, len(df)-1, 2):
        G.add_edge(df.iloc[i]['user_id'], df.iloc[i+1]['user_id'])

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.3),
                            mode='lines')

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode='markers',
                            marker=dict(size=5, color='#1f77b4'))

    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(template="plotly_dark",
                          title="User Interaction Network")

    st.plotly_chart(fig_net, use_container_width=True)

# =====================================================
# TAB 4 — FORECASTING
# =====================================================
with tab4:

    positive_df = df[df['sentiment_label'] == "positive"]
    daily_counts = positive_df.groupby('date').size().reset_index()
    daily_counts.columns = ['ds', 'y']

    model = Prophet()
    model.fit(daily_counts)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig_forecast = px.line(
        forecast,
        x="ds",
        y="yhat",
        template="plotly_dark",
        color_discrete_sequence=["#1f77b4"]
    )

    fig_forecast.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Positive Posts"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Stability Index
    st.markdown("### Sentiment Stability Index")

    daily_sent = df.groupby("date")["sentiment_label"].apply(
        lambda x: (x=="positive").mean())

    stability = np.std(daily_sent)

    st.metric("Stability Index (Lower = More Stable)",
              f"{stability:.3f}")