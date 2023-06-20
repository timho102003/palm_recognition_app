import os
import time
import pandas as pd
import streamlit as st
from streamlit_extras.stateful_button import button
from streamlit_extras.switch_page_button import switch_page

from utils import cache_index_users, load_banner, add_sidebar_logo

st.set_page_config(page_title="Feature DB Management")
add_sidebar_logo("assets/customize_logo.png")
st.image(load_banner("assets/customize_logo.png"), width=450)
st.markdown("# Manage Feature DB")
st.divider()

if "SUPER" not in st.session_state:
    st.session_state.update({"SUPER": False})

if st.session_state["SUPER"]:
    st.session_state["register_users"] = cache_index_users()
    options = st.multiselect("Which user to delete", st.session_state["register_users"])
    if options:
        st.warning(f"Selected User: {options}")
        if button("Delete User", key="deleteUser"):
            if button("Go Ahead", key="doublecheck"):
                try:
                    st.session_state["index"].delete(ids=options, namespace="")
                    st.success(f"Successfully delete user {options}")
                    time.sleep(1)
                    switch_page("manage_feature_db")
                except Exception as e:
                    st.error(e)
                    st.error("Something went wrong when deleting user!!!")
else:
    st.error("You have to be a superuser to manage the feature database!")
    col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
    with col2:
        st.image("assets/sad_face.png")
