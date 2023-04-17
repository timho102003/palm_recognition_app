import os, ast
import yaml
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils import load_keypoint_model, load_model, load_index, cache_user_cnt

#TODO: Future will change to connect a database to verify if is superuser

st.set_page_config(
    page_title="Instruction",
    page_icon="👋",
)

st.write("# Palm Recognition App! 👋")

st.sidebar.success("Please Register your PALM before Identify!")

r1_c1, r1_c2 = st.columns(2)

with r1_c1:
    st.markdown(
        """
        ### User Guide:
        1. Register your `palm` in the Register Tab
        - Currently, we recommend to use your `left` hand
        2. Start recognize your identity!
    """
    )

with r1_c2:
    go_register = st.button("Go Register")
    go_identify = st.button("Go Identify")
    if go_register:
        switch_page("register")
    if go_identify:
        switch_page("identify")

st.divider()

st.session_state["PARAMS"] = yaml.safe_load(open("./params.yaml"))["streamlit"]
st.session_state["DELTA"] = 0
# Create an ImageClassifier object.
info_col1, info_col2 = st.columns(2)
with info_col2:
    st.session_state["detector"] = load_keypoint_model()
    st.success("Finish loading keypoint model", icon="🔥")
    st.session_state["index"] = load_index()
    st.success("Finish loading Palm Gallery", icon="🔥")
    st.session_state["model"] = load_model()
    st.success("Finish loading Palm Recognition Model", icon="🔥")
    st.session_state["user_tot_current"] = st.session_state["index"].describe_index_stats()["total_vector_count"]

with info_col1:
    cache_user_tot = cache_user_cnt()
    st.session_state["DELTA"] = st.session_state["user_tot_current"] - cache_user_tot
    st.metric(label="Registered Users", 
              value=f'{st.session_state["user_tot_current"]} Users', 
              delta=str(st.session_state["DELTA"]))
    
with st.sidebar:
    st.divider()

    super_user = ast.literal_eval(os.environ["superuser"])
    username = st.text_input(
                    "Enter SuperUser Name 👇",
                    label_visibility="visible",
                    disabled=False,
                    placeholder="SUPERUSER-ID",
                )
    password = st.text_input(
                    "Enter SuperUser Password 👇",
                    label_visibility="visible",
                    disabled=False,
                    placeholder="SUPERUSER-PW",
                    type="password"
                )

    if username and password:
        if username in super_user.keys():
            if password != super_user[username]:
                st.error("Incorrect Password")
        else:
            st.error("Incorrect username")
    sb_c1, sb_c2 = st.columns([0.2, 0.5])
    with sb_c1:
        is_login = st.button("LOGIN") 
    with sb_c2:
        is_logout = st.button("LOGOUT")
    if is_login:
        os.environ["SUPER"] = "True"
        st.success("Login as SUPERUSER Successfully!!!")
    if is_logout:
        os.environ["SUPER"] = "False"
        st.success("Logout Successfully!!!")
    