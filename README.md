AI Workshop: Human Face Recognition & Attendance Counting
Welcome to the AI Workshop! 🎉 This repository contains all the necessary materials to train a YOLO model for human face recognition and deploy it on a Streamlit web application.

Prerequisites ✅
Before diving in, ensure you have:

Python (Version 3.8 & above)
Visual Studio Code (VS Code)
Software Tools 🛠️
We will be using the following platforms:

Roboflow (for dataset preparation)
Google Colab (for model training)
Streamlit (for web deployment)
If you haven't register an account before, please go ahead and register one.
Step 1: Dataset Preparation 📂
We will use a publicly available human face dataset from Roboflow.

Dataset Links:
Reference 1
Reference 2
Download Options:
Using API (Recommended):
Click "Show download code" and then "Continue".
Copy and paste the generated code into your Google Colab notebook.
Replace the api_key with your unique key.
Manual Download:
Click "Download Dataset" and save the zip file locally.
Upload the zip file to Google Colab.
Step 2: Train YOLO Model 🏋️‍♂️
Refer to the following Google Colab Notebook for training.

Training Steps:
Copy the Roboflow dataset API key and paste it into your Colab notebook.
Expand the file directory to verify dataset import.
Install dependencies using pip install ultralytics.
Set the correct dataset path and train the YOLO model for 20 epochs.
Monitor training logs (this takes ~30 minutes, so grab a coffee ☕).
Check the mAP score (above 0.7 is considered good).
Visualize model performance using F1 Curve, PR Curve, P Curve, and R Curve.
Save the trained model and export the PyTorch weights for later use.
Step 3: Implement YOLO Model on Streamlit 🖥️
Steps:
Install required libraries using pip install streamlit torch opencv-python.
Create a main.py file and import necessary dependencies.
Load the trained YOLO model.
Implement an image processing function to allow image uploads for detection.
Implement real-time face recognition using a webcam.
Step 4: Deploy on Streamlit 🚀
Run streamlit run main.py in the terminal.
A local Streamlit web app should open in your browser.
Features:
✅ Upload an image to detect the number of attendees. ✅ Use a live camera to track attendance in real time.

Final Output 🎯
✅ Recognizes human faces & counts attendance accurately!
Hope you enjoy the workshop. Have a great day! 🎉
Don't forget to mark your attendance! ✅
