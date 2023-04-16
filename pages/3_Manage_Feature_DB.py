import streamlit as st
from utils import center_image
from streamlit_extras.customize_running import center_running
from streamlit_extras.stateful_button import button

if "SUPER" not in list(st.session_state.keys()):
    center_running()
else:
    if st.session_state["SUPER"]:
        st.write("Delete User:")
        username = st.text_input(
                    "Enter User ID 👇",
                    label_visibility="visible",
                    disabled=False,
                    placeholder="YOUR-ID",
                )
        if username:
            isexist_check = len(st.session_state["index"].fetch(
                    ids=[f"streamlit_user.{username.lower()}"]
                )["vectors"])
            if isexist_check:
                st.success(f"The user {username} is exist")
                if button("Delete User", key="deleteUser"):
                    if button("Go Ahead", key="doublecheck"):
                        try:
                            st.session_state["index"].delete(ids=[f"streamlit_user.{username.lower()}"], namespace='')
                            isexist_check_2 = len(st.session_state["index"].fetch(ids=[f"streamlit_user.{username.lower()}"])["vectors"])
                            st.write(isexist_check_2)
                            if isexist_check_2:
                                st.error("Something went wrong when deleting user!!!")
                            else:
                                st.success("Successfully delete user {username}")

                        except Exception as e:
                            st.error(e)
                            st.error("Something went wrong when deleting user!!!")
            else:
                st.error(f"The user {username} does not exist, please check your username input")
    else:
        st.error("You have to be a superuser to manage the feature database!")
        center_image("assets/sad_face.png")

