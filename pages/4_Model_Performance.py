import pandas as pd
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.set_page_config(page_title="Model Performance")
st.markdown("# Model Performance Detail")
st.divider()

@st.cache_data
def load_performance(assets="./assets/streamlit_model_perf.csv"):
    return pd.read_csv(assets)
@st.cache_data
def load_raw_df(assets="./assets/raw_results.csv"):
    return pd.read_csv(assets)

perf_table = load_performance()
raw_table = load_raw_df()
st.write("FPR @ 0.01: TPR=99.898%")
st.area_chart(perf_table, x="Threshold", y=["TPR", "FPR"], use_container_width=True)
df_exp_obj = dataframe_explorer(raw_table, case=False)
st.dataframe(df_exp_obj, use_container_width=True)