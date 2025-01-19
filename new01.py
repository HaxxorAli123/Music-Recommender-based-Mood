import streamlit as st
from PIL import Image
import cv2
import numpy as np
import json
import requests
import logging
from typing import Optional
from ultralytics import YOLO

# Constants
BASE_API_URL = "https://24ea-2001-d08-d5-cd9c-b1aa-aec5-8672-ab17.ngrok-free.app"
FLOW_ID = "1ed1584c-f09f-4768-b20f-c5f851190ce7"
ENDPOINT = ""  # You can set a specific endpoint name in the flow settings

TWEAKS = {
    "ChatInput-VPUMd": {},
    "ChatOutput-0RRlf": {},
    "Prompt-dS0hJ": {},
    "OpenAIModel-Vbowz": {},
    "Memory-EAO2G": {},  # memory
    "Prompt-tSOBL": {},  # detected_emo
    "Prompt-mTB7X": {}  # camera
}

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to run the flow
def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)

    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}

# Function to extract the assistant's message from the response
def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

# Load the YOLO model
TRAINED_MODEL_PATH = 'epoch20.pt'
best_mood_recognition_model = YOLO(TRAINED_MODEL_PATH)

# Streamlit app
def main():
    st.title("Music Recommender Chatbot")
    st.write("Use the sidebar to upload an image first.")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat messages
    if "detected_emotions" not in st.session_state:
        st.session_state.detected_emotions = ""  # Detected emotions

    # Sidebar for image capture and detection
    with st.sidebar:
        st.header("Image Upload & Detection")
        picture = st.camera_input("Take a picture")

        if picture is not None:
            # Convert the uploaded file to an image
            image = Image.open(picture).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert the image to a format compatible with YOLO
            image_np = np.array(image)  # Convert to numpy array (RGB format)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Perform inference on the uploaded image
            results = best_mood_recognition_model(image_bgr)

            # Get the plotted detection result and ensure consistent colors
            detection_image = results[0].plot()
            detection_image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
            st.image(detection_image_rgb, caption="Detection Results", use_container_width=True)

            # Extract detected emotions
            detected_classes = []
            for result in results:
                detections = result.boxes
                for box in detections:
                    class_id = int(box.cls[0])
                    class_name = best_mood_recognition_model.names[class_id]
                    detected_classes.append(class_name)

            # Save detected emotions in session state
            st.session_state.detected_emotions = ", ".join(detected_classes)

    # Main section for recommendations and chatbot
    st.subheader("Recommendations & Chat")
    if st.session_state.detected_emotions:
        query = f"Based on the detected emotions: {st.session_state.detected_emotions}, what songs do you recommend?"
        assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))

        # Display detected emotions and assistant response
        st.write("Detected Emotions:", st.session_state.detected_emotions)
        st.subheader("Assistant Response")
        st.write(assistant_response)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Input box for user query
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "avatar": "💬",  # Emoji for user
            }
        )
        with st.chat_message("user", avatar="💬"):
            st.write(query)

        # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)

        # Save assistant response in session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤖",  # Emoji for assistant
            }
        )

if __name__ == "__main__":
    main()
