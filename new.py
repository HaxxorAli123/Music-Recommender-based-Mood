import streamlit as st
from PIL import Image
import cv2
import numpy as np
import json
import requests
import logging
from typing import Optional
from ultralytics import YOLO
from ultralytics import YOLOWorld


# Constants
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "1ed1584c-f09f-4768-b20f-c5f851190ce7"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

TWEAKS = {
  "ChatInput-VPUMd": {},
  "ChatOutput-0RRlf": {},
  "Prompt-dS0hJ": {},
  "OpenAIModel-Vbowz": {},
  "Memory-EAO2G": {}, #memory
  "Prompt-tSOBL": {}, #detected_emo
  "Prompt-mTB7X": {} # camera
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
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
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

    # Log the response for debugging
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
        # Extract the response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."
    
#Load the YOLO model 
TRAINED_MODEL_PATH = 'epoch20.pt'
best_mood_recognition_model = YOLO(TRAINED_MODEL_PATH)

#streamlit app
def main():
    st.title("Music Recommender Chatbot")
    st.write("🢀Please capture an image first before you start asking question.")

    with st.sidebar:
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)
        json_full = []

        if picture is not None:
            # Convert the uploaded file to an image
            image = Image.open(picture).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert the image to a format compatible with YOLO (numpy array)
            image_np = np.array(image)  # Convert to numpy array (RGB format)

            # YOLO expects BGR format, so convert if needed
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Perform inference on the uploaded image
            results = best_mood_recognition_model(image_bgr)

            # Get the plotted detection result and ensure consistent colors
            detection_image = results[0].plot()
            detection_image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

            json_full = results[0].to_json()
            
            # Show the detection results on the original image
            st.image(detection_image_rgb, caption="Detection Results", use_container_width=True)

            # Iterate over the results and print detected classes
            detected_classes = []
            for result in results:
                detections = result.boxes
                for box in detections:
                    class_id = int(box.cls[0])  # Class index
                    confidence = box.conf[0]  # Confidence score
                    class_name = best_mood_recognition_model.names[class_id]  # Class name

                    detected_classes.append(class_name)
                    print(f"Detected: {class_name} with confidence: {confidence:.2f}")

            # Assign json_full based on detections
            TWEAKS["Prompt-mTB7X"]["camera"] = json_full

# Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False  # Track if chat has started

    # Auto-start chat if it hasn't been started
    if not st.session_state.chat_started:
        # Default starting query
        default_query = "Hello! What can I help you with today?"
        st.session_state.chat_started = True

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages with avatars
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Input box for user message
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "avatar": "💬",  # Emoji for user
            }
        )
        with st.chat_message("user", avatar="💬"):  # Display user message
            st.write(query)

        # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):  # Emoji for assistant
            message_placeholder = st.empty()  # Placeholder for assistant response
            with st.spinner("Thinking..."):
                # Fetch response from Langflow with updated TWEAKS and using `query`
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)

        # Add assistant response to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤖",  # Emoji for assistant
            }
        )

        if picture is not None:
            # Show detection result if picture was uploaded
            st.image(results[0].plot())

if __name__ == "__main__":
    main()
