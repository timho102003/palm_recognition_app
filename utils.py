import os
import cv2
import math
import pinecone
import onnxruntime
import numpy as np
from PIL import Image
import mediapipe as mp
import streamlit as st
from rembg import remove
from typing import List, Tuple
from mediapipe.tasks import python
import torchvision.transforms as T
from mediapipe.tasks.python import vision


@st.cache_resource
def load_keypoint_model(asset="hand_landmarker.task"):
    asset = os.path.join(st.session_state["PARAMS"]["data_fld"], asset) 
    base_options = python.BaseOptions(model_asset_path=asset)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


@st.cache_resource
def load_model():
    asset = os.path.join(st.session_state["PARAMS"]["data_fld"], st.session_state["PARAMS"]["model"])
    model = onnxruntime.InferenceSession(asset)
    return model


@st.cache_resource
def load_index():
    pinecone.init(os.environ["pinecone_api"], environment=os.environ["pinecone_env"])
    index = pinecone.Index(os.environ["index_collection_name"])
    return index

@st.cache_data
def cache_user_cnt():
    user_tot = st.session_state["index"].describe_index_stats()["total_vector_count"]
    return user_tot

@st.cache_data
def cache_user_cnt():
    user_tot = st.session_state["index"].describe_index_stats()["total_vector_count"]
    return user_tot

def cache_index_users():
    out = st.session_state["index"].query(
                                            top_k=1000,
                                            vector= [0] * 512, # embedding dimension
                                            namespace='',
                                            include_values=False
                                        )
    reg_users = []
    for vec in out["matches"]:
        reg_users.append(vec["id"])
    return sorted(reg_users)

def rotate_points(points, angle, center):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(points - center, R) + center


def detect_keypoints(rm_bg):
    image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rm_bg))
    detection_result = st.session_state["detector"].detect(image_frame)
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
            5,
            (0, 255, 255),
            thickness=-1,
            lineType=cv2.FILLED,
        )
        cv2.putText(
            frameCopy,
            "{}".format(idx),
            (x_pt, y_pt),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
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
    palm_but_y = rot_points[0, 1] 
    x_min = max(0, x_min - 5)
    y_min = max(max(0, y_min - 5), palm_but_y)
    x_max = min(r_width, x_max + 5)
    y_max = min(r_height, y_max + 5)
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
            T.Resize(st.session_state["PARAMS"]["img_crop"]),
            # T.CenterCrop(st.session_state["PARAMS"]["img_crop"]),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=st.session_state["PARAMS"]["img_mean"], std=st.session_state["PARAMS"]["img_std"]),
        ]
    )
    img = test_transform(img)
    img = img.unsqueeze(0)
    ort_inputs = {st.session_state["model"].get_inputs()[0].name: to_numpy(img)}
    ort_outs = st.session_state["model"].run(None, ort_inputs)
    onnx_out = ort_outs[0]
    return onnx_out

def center_image(imgpath=""):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image(imgpath)

    with col3:
        st.write(' ')