"""Streamlit frontend for the PyTorch blind-assistance prototype."""
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from app import BlindAssistanceDetector


@st.cache_resource(show_spinner=False)
def load_detector(device: str) -> BlindAssistanceDetector:
    return BlindAssistanceDetector(device=device)


def read_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The selected file is not a supported, readable image.") from exc
    rgb_frame = np.asarray(image)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    return image, bgr_frame


def main() -> None:
    st.set_page_config(
        page_title="Blind Assistance",
        page_icon="👁️",
        layout="wide",
    )
    st.title("AI Blind Assistance")
    st.caption("Upload or capture an image to identify familiar objects.")
    st.warning(
        "Educational prototype only. It does not measure distance or reliably "
        "detect hazards. Do not use it for navigation or as a substitute for "
        "a cane, guide dog, or other established mobility aid."
    )

    input_mode = st.radio("Choose an image source", ("Upload image", "Take a photo"))
    if input_mode == "Upload image":
        image_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=("jpg", "jpeg", "png"),
        )
    else:
        image_file = st.camera_input("Take a photo")

    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    device = st.selectbox("Inference device", available_devices)
    threshold = st.slider(
        "Minimum confidence",
        min_value=0.1,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Higher values show fewer, higher-confidence detections.",
    )

    if image_file is None:
        st.info("Choose or take an image to get started.")
        return

    try:
        original, frame = read_image(image_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    if not st.button("Analyze image", type="primary"):
        st.image(original, caption="Selected image", use_container_width=True)
        return

    with st.spinner("Loading the object detection model and analyzing the image..."):
        detector = load_detector(device)
        detections = detector.predict(frame, threshold=threshold)
        description = detector.describe(detections)
        annotated = detector.draw(frame.copy(), detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.subheader("Result")
    st.success(description)
    original_column, result_column = st.columns(2)
    with original_column:
        st.image(original, caption="Original image", use_container_width=True)
    with result_column:
        st.image(
            annotated_rgb,
            caption=f"Detections: {len(detections)}",
            use_container_width=True,
        )

    if detections:
        st.subheader("Detected objects")
        st.dataframe(
            [
                {
                    "Object": detection["label"],
                    "Position": detection["position"],
                    "Confidence": f"{detection['score']:.1%}",
                }
                for detection in detections
            ],
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
