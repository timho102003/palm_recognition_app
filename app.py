import math
import os
import time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime
import pinecone
import streamlit as st
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from rembg import remove
from torchvision import transforms as T

PARAMS = yaml.safe_load(open("./params.yaml"))["streamlit"]

INDEX_API_KEY = os.environ["pinecone_api"]
INDEX_COLLECTION_NAME = os.environ["index_collection_name"]
ENV_NAME = os.environ["pinecone_env"]

# `set_page_config()` can only be called once per app, and must be called as the first Streamlit command in your script.
st.set_page_config(layout="wide", page_title="Image Background Remover")
st.header("Palmar Recognition")
st.write("Please register before you start recognize")

info_string = """
    Please follow the rules below when registering / recognizing:
    1. Make sure there is sufficient light when taking the image.
    2. Place your palm facing up.
    3. Ensure that the image is not blurry.
"""
st.info(info_string)

# import pdb; pdb.set_trace()


@st.cache_resource
def load_keypoint_model(asset="hand_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=asset)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


@st.cache_resource
def load_model():
    model = onnxruntime.InferenceSession(PARAMS["model"])
    return model


@st.cache_resource
def load_index():
    pinecone.init(INDEX_API_KEY, environment=ENV_NAME)
    index = pinecone.Index(INDEX_COLLECTION_NAME)
    return index

@st.cache_data
def cache_user_cnt():
    user_tot = index.describe_index_stats()["total_vector_count"]
    return user_tot

# Create an ImageClassifier object.
info_col1, info_col2 = st.columns(2)
with info_col2:
    detector = load_keypoint_model()
    st.success("Finish loading keypoint model", icon="🔥")
    index = load_index()
    st.success("Finish loading Palm Gallery", icon="🔥")
    model = load_model()
    st.success("Finish loading Palm Recognition Model", icon="🔥")

with info_col1:
    user_tot_current = index.describe_index_stats()["total_vector_count"]
    cache_user_tot = cache_user_cnt()
    delta = user_tot_current - cache_user_tot
    st.metric(label="Registered Users", value=f"{user_tot_current} Users", delta=str(delta))

agree = st.checkbox("ENABLE_CAMERA")
if agree:
    my_upload = st.camera_input("Take a picture")
    is_take = st.button("Take Photo", key="custom_camera_trigger")
    st.success("Successfully take a photo!")
else:
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
register_agree = st.checkbox("Register Your Palm", key="register_box")

def rotate_points(points, angle, center):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(points - center, R) + center


def detect_keypoints(rm_bg):
    image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rm_bg))
    detection_result = detector.detect(image_frame)
    img_w, img_h = rm_bg.size
    if len(detection_result.hand_landmarks):
        points = mp_points_to_np(
            detection_result.hand_landmarks[0], width=img_w, height=img_h
        )
        annotated_image = draw(image_frame.numpy_view(), points)
    else:
        points = None
        annotated_image = None
    return points, annotated_image


def draw(img: np.ndarray, hand_landmarks_list: np.ndarray, ratio=1.0) -> np.ndarray:
    frameCopy = img.copy()
    height, width, _ = img.shape
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        x_pt = int(hand_landmarks[0])
        y_pt = int(hand_landmarks[1])
        cv2.circle(
            frameCopy,
            (x_pt, y_pt),
            20,
            (0, 255, 255),
            thickness=-1,
            lineType=cv2.FILLED,
        )
        cv2.putText(
            frameCopy,
            "{}".format(idx),
            (x_pt, y_pt),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            6,
            lineType=cv2.LINE_AA,
        )
    if ratio < 1.0:
        r_width = int(width * ratio)
        r_height = int(height * ratio)
        frameCopy = cv2.resize(
            frameCopy, (r_width, r_height), interpolation=cv2.INTER_AREA
        )
    return frameCopy


def mp_points_to_np(point_list: List, width: float, height: float) -> np.ndarray:
    points = np.zeros((len(point_list), 2))
    for idx in range(len(point_list)):
        hand_landmarks = point_list[idx]
        points[idx, 0] = int(hand_landmarks.x * width)
        points[idx, 1] = int(hand_landmarks.y * height)
    return points


def remove_bg(image: Image, bg_color: Tuple = (255, 255, 255), ratio=1.0):
    rm_bg_img = remove(image)
    # Create a new image object with a white background
    background = Image.new("RGB", rm_bg_img.size, bg_color)
    # Paste the processed image onto the white background
    background.paste(rm_bg_img, (0, 0), rm_bg_img)
    return background


def normalize_img(image: Image, points: np.ndarray):
    width, height = image.size
    x1, y1 = points[9, 0], points[9, 1]
    x2, y2 = points[12, 0], points[12, 1]
    # Assuming you have two facial landmark points as (x1, y1) and (x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    angle = math.degrees(angle)
    if abs(angle) > 90:
        rot_angle = angle - 90
    else:
        rot_angle = -1 * (90 - angle)
    rotated_image = image.rotate(rot_angle, fillcolor="white")
    # Rotate Points
    rot_points = rotate_points(
        points=points, angle=rot_angle, center=np.asarray((width // 2, height // 2))
    )
    rot_points = np.clip(rot_points, a_max=None, a_min=0)
    x_min, y_min = rot_points.min(axis=0)
    x_max, y_max = rot_points.max(axis=0)
    r_width, r_height = rotated_image.size
    x_min = max(0, x_min - 100)
    y_min = max(0, y_min - 100)
    x_max = min(r_width, x_max + 100)
    y_max = min(r_height, y_max + 100)
    crop_box = (x_min, y_min, x_max, y_max)
    cropped_image = rotated_image.crop(crop_box)
    return rot_angle, cropped_image


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def feature_extract(img):
    test_transform = T.Compose(
        [
            T.CenterCrop(PARAMS["img_crop"]),
            T.ToTensor(),
            T.Normalize(mean=PARAMS["img_mean"], std=PARAMS["img_std"]),
        ]
    )
    img = test_transform(img)
    img = img.unsqueeze(0)
    ort_inputs = {model.get_inputs()[0].name: to_numpy(img)}
    ort_outs = model.run(None, ort_inputs)
    onnx_out = ort_outs[0]
    return onnx_out

if __name__ == "__main__":
    col1, col2, col3, col4 = st.columns(4)

    if my_upload is not None:
        imgfile = my_upload
    else:
        imgfile = "./palmar_example.jpg"

    image = Image.open(imgfile)
    col1.write("Original Image :camera:")
    col1.caption("elapse time: 0 ms")
    col1.image(image)

    start = time.time()
    rm_bg = remove_bg(image=image)
    end = time.time()
    col2.write("Remove Background")
    col2.caption("elapse time: {:.3f} ms".format((end - start) * 1000))
    col2.image(rm_bg)

    start = time.time()
    keypoints, keypoint_plot = detect_keypoints(rm_bg=rm_bg)
    end = time.time()
    if keypoints is not None:
        col3.write("Keypoint Detection")
        col3.caption("elapse time: {:.3f} ms".format((end - start) * 1000))
        col3.image(keypoint_plot)
        rot_angle, norm_img = normalize_img(image=rm_bg, points=keypoints)
        col4.write("Normalize: ")
        col4.caption(
            "elapse time: {:.3f} ms, rotate: {:.1f} deg".format(
                (end - start) * 1000, rot_angle
            )
        )
        col4.image(norm_img)
    else:
        col3.write("Keypoint Detection")
        col3.warning("Fail to detect keypoints")
        col4.write("Normalize: ")
        col4.warning("Fail to detect keypoints")
        norm_img = None

    feature = None
    if norm_img is not None:
        st.write("Start extract feature")
        start_ext = time.time()
        with st.spinner("Extracting Feature"):
            feature = feature_extract(norm_img)
        end_ext = time.time()

    if feature is not None:
        st.info("Elapse Time: {:.2f} ms".format((end_ext - start_ext) * 1000), icon="🔥")
        with st.expander("See Feature Values:"):
            st.json({"feature": feature[0].tolist()})
        if not register_agree:
            button_clicked_identify = st.button("Identify")
            if button_clicked_identify:
                start = time.time()
                response = index.query(
                    feature[0].tolist(), top_k=3, include_metadata=True
                )
                with st.expander("See Top 3 Meta Data"):
                    st.write("Top 3 MetaData:")
                    st.json(response.to_dict())
                end = time.time()
                pred = response["matches"][0]["metadata"]["label"]
                score = response["matches"][0]["score"]
                if (score + 1) / 2 < 0.6:
                    pred = "unknown user"
                our_str = (
                    "Identity: {}, Sim-Score: {:.2f}, Elapse Time: {:.2f} ms".format(
                        pred.upper(), (score + 1) / 2, (end_ext - start_ext) * 1000
                    )
                )
                if pred == "unknown user":
                    st.warning(our_str, icon="🔥")
                else:
                    st.info(our_str, icon="🔥")
                st.balloons()
        else:
            text_input = st.text_input(
                "Enter Your Name 👇",
                label_visibility="visible",
                disabled=False,
                placeholder="YOUR-ID",
            )
            if text_input:
                st.write(
                    "Verify your name before pressing register button: ", text_input
                )
            button_clicked_register = st.button("Register")
            if button_clicked_register:
                isexist_check = index.fetch(
                    ids=[f"streamlit_user.{text_input.lower()}"]
                )
                if len(isexist_check["vectors"]) == 0:
                    index.upsert(
                        [
                            (
                                f"streamlit_user.{text_input.lower()}",
                                feature[0].tolist(),
                                {"label": text_input.lower()},
                            )
                        ]
                    )
                else:
                    st.warning(f"User: {text_input}, Already Registered!!!!!")
    else:
        st.error("Fail to extract features")
