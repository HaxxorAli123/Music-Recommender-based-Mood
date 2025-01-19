# Music Recommender Chatbot

## Overview
The Music Recommender Chatbot combines **Computer Vision** and **Large Language Models (LLMs)** to recommend songs based on the user's current emotional state. This project uses facial expression recognition to detect emotions and provides personalized music recommendations through an interactive chatbot interface.

## Features
- **Emotion Detection**: Uses a YOLO-based model to analyze the user's facial expressions.
- **Music Recommendation**: Recommends songs based on detected emotions using a custom-trained language model.
- **Interactive Chatbot**: Powered by Langflow and Streamlit, enabling seamless user interaction.
- **Deployment**: Hosted using Streamlit Cloud and an Ngrok server.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - `streamlit`: For creating the web interface.
  - `Pillow`: For image handling.
  - `cv2` (OpenCV): For image processing.
  - `numpy`: For numerical computations.
  - `requests`: For API communication with Langflow.
  - `ultralytics`: For the YOLO model.
- **Tools**:
  - **Langflow**: For creating and managing LLM workflows.
  - **Ngrok**: For secure public access to the local server.

## Architecture
1. **Image Upload**: The user capture its image via the Streamlit sidebar using a webcam.
2. **Emotion Detection**:
   - The uploaded image is processed using a YOLO-based emotion detection model.
   - Detected emotions are extracted and displayed.
3. **Music Recommendation**:
   - The detected emotions are sent to the Langflow API.
   - The chatbot provides personalized song recommendations based on the emotional state.
4. **User Interaction**:
   - Users can chat with the bot to ask for additional recommendations or explore specific genres.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HaxxorAli123/Music-Recommender-based-Mood.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application** (local use):
   ```bash
   streamlit run app.py
   ```
4. **Ngrok Setup** (for remote use):
   - Start an Ngrok tunnel:
     ```bash
     ngrok http 8501
     ```
   - Update the `BASE_API_URL` in the code with the Ngrok URL.

## Project Workflow
- **Langflow Integration**:
  - The chatbot communicates with the Langflow API to handle queries and generate song recommendations.
  - The API endpoint and workflow ID are defined in the code.
- **Emotion Detection Pipeline**:
  - The YOLO model processes the uploaded image to detect emotions.
  - Detected emotions are passed to the chatbot for further processing.
- **User Interaction**:
  - Users can ask questions and receive responses in real-time.

## Deployment
- Hosted using **Streamlit Cloud**.
- Publicly accessible via **Ngrok**.

## Example Usage
1. Upload or capture a picture in the sidebar.
2. View the detected emotions.
3. Get personalized song recommendations based on the detected emotions.
4. Interact with the chatbot for additional recommendations or queries.

## Future Improvements
- Incorporate more robust emotion detection models.
- Expand the music database for diverse recommendations.
- Enable multilingual support for global users.

## Credits
- **Emotion Detection Model**: YOLO-based.
- **Chatbot Integration**: Langflow.
- **Frontend**: Streamlit.



