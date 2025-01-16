import streamlit as st
from PIL import Image
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from langflow.interface.run import run_flow
from langflow.interface.utils import load_flow_from_json

# Initialize logging
logging.basicConfig(level=logging.INFO)

# YOLO model path
TRAINED_MODEL_PATH = 'epoch20.pt'
best_mood_recognition_model = YOLO(TRAINED_MODEL_PATH)

# Load Langflow flow
@st.cache_resource
def load_langflow_flow(flow_file_path):
    try:
        with open(flow_file_path, "r") as file:
            flow_data = file.read()
        return load_flow_from_json(flow_data)
    except Exception as e:
        st.error(f"Error loading Langflow flow: {e}")
        return None

# Streamlit app
def main():
    st.title("Music Recommender Chatbot")
    st.write("Please capture an image to detect your mood, then start chatting!")

    # Sidebar for camera
    with st.sidebar:
        enable_camera = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable_camera)
        detected_mood = None

        if picture is not None:
            # Process the uploaded image
            image = Image.open(picture).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert to NumPy array (RGB to BGR for YOLO)
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Perform inference with YOLO
            results = best_mood_recognition_model(image_bgr)
            detection_image = results[0].plot()
            detection_image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

            # Show detection results
            st.image(detection_image_rgb, caption="Detection Results", use_container_width=True)

            # Extract detected classes
            detected_classes = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                class_name = best_mood_recognition_model.names[class_id]
                detected_classes.append(class_name)
                logging.info(f"Detected: {class_name} with confidence: {confidence:.2f}")

            # Assign mood for Langflow tweak
            detected_mood = detected_classes[0] if detected_classes else "neutral"
            st.info(f"Detected Mood: {detected_mood}")

    # Load Langflow flow
    flow_path = "flow.json"  # Update this path to your actual flow file
    flow = load_langflow_flow(flow_path)
    if not flow:
        return

    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Input for user message
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session
        st.session_state.messages.append(
            {"role": "user", "content": query, "avatar": "💬"}
        )
        with st.chat_message("user", avatar="💬"):
            st.write(query)

        # Run Langflow flow
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    tweaks = {"detected_mood": detected_mood} if detected_mood else {}
                    response = run_flow(flow, {"query": query, **tweaks})
                    assistant_response = response["result"]
                except Exception as e:
                    assistant_response = f"Error: {e}"

                message_placeholder.write(assistant_response)

        # Add assistant response to session
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response, "avatar": "🤖"}
        )

if __name__ == "__main__":
    main()
