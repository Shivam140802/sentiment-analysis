import os
import pandas as pd
import streamlit as st
import psycopg2
import time
from dotenv import load_dotenv


load_dotenv()


LOCAL_PG_HOST = os.getenv("LOCAL_PG_HOST")
LOCAL_PG_PORT = int(os.getenv("LOCAL_PG_PORT"))
LOCAL_PG_DB   = os.getenv("LOCAL_PG_DB")
LOCAL_PG_USER = os.getenv("LOCAL_PG_USER")
LOCAL_PG_PASS = os.getenv("LOCAL_PG_PASS")
PREDICTION_TABLE = os.getenv("PREDICTION_TABLE", "reviews_prediction")


def get_predicted_reviews(limit=None):
    conn = psycopg2.connect(
        host=LOCAL_PG_HOST, port=LOCAL_PG_PORT,
        dbname=LOCAL_PG_DB, user=LOCAL_PG_USER, password=LOCAL_PG_PASS
    )
    if limit:
        query = f"""
            SELECT content, prediction, predicted_at
            FROM {PREDICTION_TABLE}
            ORDER BY predicted_at DESC LIMIT {limit};
        """
    else:
        query = f"""
            SELECT content, prediction, predicted_at
            FROM {PREDICTION_TABLE}
            ORDER BY predicted_at DESC;
        """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


st.set_page_config(page_title="Reviews Dashboard", layout="wide")
st.title("App Reviews Prediction Dashboard")
st.sidebar.header("Options")


# --- Session state for auto-refresh and manual refresh ---
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()


# Manual refresh button
if st.sidebar.button("Refresh Reviews"):
    st.session_state.refresh_count += 1
    st.session_state.last_refresh = time.time()


# Auto-refresh every 5 seconds
refresh_interval = 5  # seconds
if time.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.refresh_count += 1
    st.session_state.last_refresh = time.time()


# --- Fetch and display reviews ---
df = get_predicted_reviews()
if df.empty:
    st.warning("No predicted reviews yet.")
else:
    st.write("### Predicted Reviews")
    st.dataframe(df[['content','prediction']])
    positive_count = (df['prediction']==1).sum()
    negative_count = (df['prediction']==0).sum()
    st.write("### Feedback Summary")
    st.bar_chart(
        pd.DataFrame({"Feedback":["Positive","Negative"],"Count":[positive_count,negative_count]})
        .set_index("Feedback")
    )