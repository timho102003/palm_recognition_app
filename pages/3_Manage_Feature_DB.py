import os
import pandas as pd
import streamlit as st
from utils import center_image, cache_index_users
from streamlit_extras.stateful_button import button
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Feature DB Management")
st.markdown("# Manage Feature DB")
st.divider()


if os.environ["SUPER"] == "True":
    st.session_state["register_users"] = cache_index_users()
    options = st.multiselect(
    'Which user to delete',
    st.session_state["register_users"])
    if options:
        st.warning(f"Selected User: {options}")
        if button("Delete User", key="deleteUser"):
            if button("Go Ahead", key="doublecheck"):
                try:
                    st.session_state["index"].delete(ids=options, namespace='')
                    st.success(f"Successfully delete user {options}")
                    switch_page("manage_feature_db")
                except Exception as e:
                    st.error(e)
                    st.error("Something went wrong when deleting user!!!")
else:
    st.error("You have to be a superuser to manage the feature database!")
    center_image("assets/sad_face.png")

