import streamlit as st
from streamlit_card import card
from streamlit_extras.mention import mention
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Author")
st.markdown("# Profile")


link_col1, link_col2, link_col3, link_col4 = st.columns(4)

with link_col1:
    mention(
        label="LinkedIn",
        icon="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png",
        url="https://www.linkedin.com/in/kangchi-ho/",
    )

with link_col2:
    mention(
        label="GitHub",
        icon="https://cdn1.iconfinder.com/data/icons/unicons-line-vol-3/24/github-alt-1024.png",  # You can also just use an emoji
        url="https://github.com/timho102003",
    )

with link_col3:
    mention(
        label="DagsHub",
        icon="https://workable-application-form.s3.amazonaws.com/advanced/production/61dccaba344d40f0d1d4a88b/42591a89-dd53-9730-9978-e3f57c30e7e7",
        url="https://dagshub.com/timho102003",
    )

with link_col4:
    to_home = st.button("Back to Home")
    if to_home:
        switch_page("home")

st.divider()

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.image("assets/author.jpg", width=300)

with row2_col2:
    st.markdown(
        """
        ### Tim K.C, Ho
        #### Experience:   
        **Delta Research Center:**    
        - Data Science Intern  
        - 2022.11 - Now  

        **Delta Research Center:**    
        - R&D Computer Vision Engineer  
        - 2019.01 - 2022.06  
        """
    )

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    card(
        title="ICIP2018 Publication",
        text="Edge-Coupled and Multi-Dropout Face Alignment",
        image="https://artsformysake.files.wordpress.com/2020/08/face-id-glitch.gif",
        url="https://ieeexplore.ieee.org/document/8451165",
    )

with row3_col2:
    card(
        title="IEEE-ARIS Publication",
        text="Mixture of trees with three layers",
        image="https://cdn.dribbble.com/users/307908/screenshots/4449308/big.gif",
        url="https://ieeexplore.ieee.org/document/8297179",
    )

row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    card(
        title="[NLP] News Classification",
        text="PyTorch News Classifier",
        image="https://i.gifer.com/U9Si.gif",
        url="https://dagshub.com/timho102003/news_classifier",
    )

with row4_col2:
    card(
        title="[CV] Image Classification",
        text="PyTorch Cat&Dog Classification",
        image="https://files.cults3d.com/uploaders/26071354/illustration-file/264da7c9-75c9-4225-827c-8489aa28d951/Cat-dog6.gif",
        url="https://dagshub.com/timho102003/pytorch-image-classification.git",
    )

row5_col1, row5_col2 = st.columns(2)

with row5_col1:
    card(
        title="[CV] Paper Implementation",
        text="Background-aware Classification Activation Map for Weakly Supervised Object Localization",
        image="https://raw.githubusercontent.com/gowrishankarin/data_science/master/topics/dl/model_explanation/anime.gif",
        url="https://github.com/timho102003/Background-Aware-CAM-WSOL",
    )

with row5_col2:
    card(
        title="[CV] Facial Detection",
        text="C++ Facial Landmark Detection",
        image="https://camo.githubusercontent.com/6e2f31c0c5d81660e7249dc31a2f570bb41685c21bea8ba4e1f1572ea692ed18/68747470733a2f2f777977752e6769746875622e696f2f70726f6a656374732f4c41422f737570706f72742f57464c575f616e6e6f746174696f6e2e706e67",
        url="https://github.com/timho102003/FaceDetection",
    )

row6_col1, row6_col2 = st.columns(2)

with row6_col1:
    card(
        title="DagsHub's Integration with PyCaret",
        text="DagsHub Blog 01",
        image="https://dagshub.com/blog/content/images/size/w2000/2023/01/dagshub_loves_pycaret--1-.png",
        url="https://dagshub.com/blog/pycaret-integration/",
    )

with row6_col2:
    card(
        title="[Time-Series] PyCaret Stock Price Prediction",
        text="DagsHub Blog 02",
        image="https://dm0qx8t0i9gc9.cloudfront.net/thumbnails/video/TOEwt0C/videoblocks-stock-market-trading-graphic-background-animation-of-chart_bgbdx8ktl_thumbnail-1080_01.png",
        url="https://dagshub.com/blog/how-to-use-pycaret-with-dagshub/",
    )

row7_col1, row7_col2 = st.columns(2)

with row7_col1:
    card(
        title="400 Dataset from AWS Data Registry are Available on DagsHub",
        text="DagsHub Blog 03",
        image="https://dagshub.com/blog/content/images/size/w2000/2023/02/Blog-Banner.png",
        url="https://dagshub.com/blog/400-dataset-from-aws-data-registry-available-on-dagshub/",
    )
