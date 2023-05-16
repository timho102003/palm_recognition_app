import os
import time

import streamlit as st
from PIL import Image
from streamlit_extras.switch_page_button import switch_page

from utils import detect_keypoints, feature_extract, normalize_img, remove_bg, check_load_status, load_banner, add_sidebar_logo

st.set_page_config(page_title="Palm Identification")
add_sidebar_logo("assets/katsh_logo.png")
st.image(load_banner("assets/katsh_logo.png"), width=450)
st.markdown("# Identification")
st.divider()


info_string = """
    Please follow the rules below when identify:
    1. Make sure there is sufficient light when taking the image.
    2. Place your left palm facing up and spread your fingers.
    3. Ensure that the image is not blurry or cover by your phone shadow.
"""

st.info(info_string)

if not check_load_status():
    st.warning("Be patient, models are still loading...")
    col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
    with col2:
        st.image("assets/sad_face.png")
    if "SUPER" not in st.session_state:
        st.session_state.update({"SUPER": False})
else:
    in_col1, in_col2 = st.columns(2)

    with in_col1:
        st.write("Identify Example (Left Hand)")
        st.image("assets/right_hand_contour.png")
    with in_col2:
        my_upload = st.camera_input("Take a picture")#, on_change=clear_cache)

    if st.session_state["SUPER"]:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2 = st.columns(2)

    if my_upload is not None:
        imgfile = my_upload
        img = Image.open(imgfile)
        col1.write("Original Image :camera:")
        col1.caption("elapse time: 0 ms")
        col1.image(img)

        rm_bg_start = time.time()
        st.session_state["rm_bg"] = remove_bg(image=img)
        rm_bg_end = time.time()

        if st.session_state["SUPER"]:
            col2.write("Remove Background")
            col2.caption("elapse time: {:.3f} ms".format((rm_bg_end - rm_bg_start) * 1000))
            col2.image(st.session_state["rm_bg"])

        pt_det_start = time.time()
        st.session_state["keypoints"], st.session_state["keypoint_plot"] = detect_keypoints(rm_bg=st.session_state["rm_bg"])
        pt_det_end = time.time()
        if st.session_state["keypoints"] is not None:
            if st.session_state["SUPER"]:
                col3.write("Keypoint Detection")
                col3.caption(
                    "elapse time: {:.3f} ms".format((pt_det_end - pt_det_start) * 1000)
                )
                col3.image(st.session_state["keypoint_plot"])
            norm_start = time.time()
            st.session_state["rot_angle"], st.session_state["norm_img"] = normalize_img(image=st.session_state["rm_bg"], points=st.session_state["keypoints"])
            norm_end = time.time()
            if st.session_state["SUPER"]:
                col4.write("Normalize: ")
                col4.caption(
                    "elapse time: {:.3f} ms, rotate: {:.1f} deg".format(
                        (norm_end - norm_start) * 1000, st.session_state["rot_angle"]
                    )
                )
                col4.image(st.session_state["norm_img"])
            else:
                col2.write("Normalize: ")
                col2.caption(
                    "elapse time: {:.3f} ms, rotate: {:.1f} deg".format(
                        (norm_end - norm_start) * 1000, st.session_state["rot_angle"]
                    )
                )
                col2.image(st.session_state["norm_img"])
        else:
            if st.session_state["SUPER"]:
                col3.write("Keypoint Detection")
                col3.warning("Fail to detect keypoints")
                col4.write("Normalize: ")
                col4.warning("Fail to detect keypoints")
            else:
                col2.write("Normalize: ")
                col2.warning("Fail to detect keypoints")
            st.session_state["norm_img"] = None

        st.session_state["feature"] = None
        if st.session_state["norm_img"] is not None:
            st.write("Start extract feature")
            start_ext = time.time()
            with st.spinner("Extracting Feature"):
                st.session_state["feature"] = feature_extract(st.session_state["norm_img"])
            end_ext = time.time()

        if st.session_state["feature"] is not None:
            if st.session_state["SUPER"]:
                st.info(
                    "Elapse Time: {:.2f} ms".format((end_ext - start_ext) * 1000), icon="🔥"
                )
            else:
                tot_time_spend = (
                    (rm_bg_end - rm_bg_start)
                    + (pt_det_end - pt_det_start)
                    + (norm_end - norm_start)
                    + (end_ext - start_ext)
                )
                st.info(
                    "Total Process Time: {:.2f} ms".format(tot_time_spend * 1000), icon="🔥"
                )

            with st.expander("See Feature Values:"):
                st.json({"feature": st.session_state["feature"][0].tolist()})

            if st.session_state["SUPER"]:
                threshold = st.slider('Select the threshold', 0.0, 1.0, st.session_state["PARAMS"]["threshold"], 0.01)
            else:
                threshold = st.session_state["PARAMS"]["threshold"]

            # button_clicked_identify = st.button("Identify")
            # if button_clicked_identify:
            start = time.time()
            response = st.session_state["index"].query(
                st.session_state["feature"][0].tolist(), top_k=3, include_metadata=True
            )
            with st.expander("See Top 3 Meta Data"):
                st.write("Top 3 MetaData:")
                st.json(response.to_dict())
            end = time.time()
            if type(response["matches"][0]["metadata"]["label"]) == str:
                pred = response["matches"][0]["metadata"]["label"]
            elif isinstance(response["matches"][0]["metadata"]["label"], (int, float)):
                pred = str(int(response["matches"][0]["metadata"]["label"]))
            else:
                pred = str(response["matches"][0]["metadata"]["label"])
            score = response["matches"][0]["score"]
            if (score + 1) / 2 < threshold:
                pred = "unknown user"
            our_str = "Identity: {}, Sim-Score: {:.2f}, Elapse Time: {:.2f} ms".format(
                pred.upper(), (score + 1) / 2, (end_ext - start_ext) * 1000
            )
            if pred == "unknown user":
                st.warning(our_str, icon="🔥")
                go_register = st.button("Go Register")
                if go_register:
                    # clear_cache()
                    switch_page("register")
            else:
                st.info(our_str, icon="🔥")
                st.balloons()
        else:
            st.error("Fail to extract features")
