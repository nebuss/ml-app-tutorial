# Core Pkgs
import streamlit as st
from yolov5_page import main as yolov5_page
from yolov5_mlflow_page import main as yolov5_mlflow_page

PAGE_CONFIG = {
    "page_title": "Deep Learning Application Demo",
    "page_icon": "📺",
    "layout": "wide",
}
st.set_page_config(**PAGE_CONFIG)


def main():
    st.title("DL model apps")
    menu = ["Home", "Object Detection Sample", "Object Detection Trained"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
    if choice == "Object Detection Sample":
        yolov5_page()
    if choice == "Object Detection Trained":
        yolov5_mlflow_page()


if __name__ == "__main__":
    main()
