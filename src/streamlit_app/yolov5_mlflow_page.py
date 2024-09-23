import os
from io import BytesIO
from typing import Any, Dict

import mlflow
import numpy as np
import streamlit as st
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

URI = os.environ.get("MLFLOW_URI")


def read_image(img_file_buffer: BytesIO) -> np.array:
    """이미지를 읽습니다.

    Args:
        img_file_buffer (BytesIO): 이미지 파일 버퍼. Defaults to None.

    Returns:
        np.array: np.array 형식의 이미지
    """
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    return img_array


def detect(model: Any, image: np.array) -> np.array:
    """yolov5 모델을 사용하여 이미지 내의 객체를 탐지하고 탐지 결과를 이미지로 반환합니다.

    Args:
        image (np.array): 이미지

    Returns:
        np.array: 객체의 위치가 상자로 표시된 이미지
    """
    results = model([image])
    return results[0].plot()


def draw_image(image: np.array):
    """이미지를 streamlit 이미지로 표현합니다.

    Args:
        image (np.array): 이미지
    """
    st.image(
        image,
        use_column_width=True,
    )


def upload_image() -> UploadedFile:
    """이미지를 업로드합니다.

    Returns:
        UploadedFile: 업로드된 이미지
    """
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    return img_file_buffer


def download_model(model_version, model_name="yolov5_model"):
    mlflow.set_tracking_uri(URI)

    # Initialize MLflow client
    client = MlflowClient()
    model_version_details = client.get_model_version(
        name=model_name, version=model_version
    )

    # Extract the run_id associated with this model version
    run_id = model_version_details.run_id

    # Use the run_id to get the artifacts for the run
    artifact_path = (
        "weights/best.pt"  # Change this to the artifact path you're interested in
    )
    local_dir = client.download_artifacts(run_id, artifact_path)
    return local_dir


def list_models(model_name="yolov5_model"):
    mlflow.set_tracking_uri(URI)
    models_list: Dict[str, ModelVersion] = {}
    for mv in mlflow.search_model_versions(filter_string=f"name='{model_name}'"):
        models_list[mv.version] = mv

    return models_list


def load_model(version):
    local_dir = download_model(version)
    return YOLO(local_dir)


def main():
    st.subheader("Yolov5")
    models_list = list_models()
    st.sidebar.markdown("----")
    model_version = st.sidebar.selectbox("model version", models_list.keys())
    st.sidebar.button("Refresh")
    model = load_model(model_version)

    img_file_buffer = upload_image()
    col1, col2 = st.columns(2)
    if img_file_buffer is not None:
        image = read_image(img_file_buffer)
        with col1:
            st.button("Show Original Only")
            draw_image(image)

        with col2:
            button = st.button("Detect")
            if button:
                detected_image = detect(model, image)
                draw_image(detected_image)


if __name__ == "__main__":
    main()
