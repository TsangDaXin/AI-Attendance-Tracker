import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import datetime
import os

# Load YOLOv8 model
model_path = "best.pt"  # Change this if needed
model = YOLO(model_path)

# Initialize the session state if not already done
if 'attendance_log' not in st.session_state:
    st.session_state.attendance_log = pd.DataFrame(columns=['Date', 'Time', 'Count', 'Source'])

# Create directories for logs and screenshots
if not os.path.exists('attendance_logs'):
    os.makedirs('attendance_logs')
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

st.title("Classroom Attendance Tracker 🏫")

# Sidebar for navigation
option = st.sidebar.radio("Choose an option", ["Upload Image", "Live Camera", "View Logs", "View Screenshots"])

# Function to detect people in an image
def detect_people(frame):
    results = model(frame)  # Run YOLO detection
    detected_objects = results[0].boxes
    num_students = 0
    
    for box in detected_objects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        
        if cls == 0:  # Class 0 is 'person' in COCO dataset
            num_students += 1
            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, num_students

# Function to save attendance to log
def save_attendance(count, source):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Add new row to the log
    new_entry = pd.DataFrame([{
        'Date': date_str,
        'Time': time_str,
        'Count': count,
        'Source': source
    }])
    
    st.session_state.attendance_log = pd.concat([st.session_state.attendance_log, new_entry], ignore_index=True)
    
    # Also save to CSV file
    log_path = f'attendance_logs/attendance_log.csv'
    st.session_state.attendance_log.to_csv(log_path, index=False)
    
    return f"Attendance of {count} students recorded at {time_str} on {date_str}"

# Function to save screenshot
def save_screenshot(frame, count):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/classroom_{timestamp}_{count}students.jpg"
    
    # Save the image
    if isinstance(frame, np.ndarray):
        # If it's already a numpy array, save directly with OpenCV
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        # If it's a PIL Image, convert to numpy array first
        frame_np = np.array(frame)
        cv2.imwrite(filename, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    
    # Add to screenshots log if we're tracking them
    if 'screenshot_log' not in st.session_state:
        st.session_state.screenshot_log = pd.DataFrame(columns=['Date', 'Time', 'Count', 'Filename'])
    
    # Add new row to the screenshots log
    new_entry = pd.DataFrame([{
        'Date': now.strftime("%Y-%m-%d"),
        'Time': now.strftime("%H:%M:%S"),
        'Count': count,
        'Filename': filename
    }])
    
    st.session_state.screenshot_log = pd.concat([st.session_state.screenshot_log, new_entry], ignore_index=True)
    
    # Save screenshots log to CSV
    log_path = 'attendance_logs/screenshot_log.csv'
    st.session_state.screenshot_log.to_csv(log_path, index=False)
    
    return filename, f"Screenshot saved as {filename}"

# Upload Image Option
if option == "Upload Image":
    st.subheader("Upload an Image for Attendance Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert to numpy array and process
        image_np = np.array(image)
        processed_image, count = detect_people(image_np)
        
        # Show results
        st.image(processed_image, caption=f"Detected Attendance: {count}", use_container_width=True)
        st.write(f"### Attendance Count: {count} 🧑‍🎓")
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
        
        # Add a button to save attendance
        if col1.button("Save Attendance Record 📝", key="save_image"):
            msg = save_attendance(count, "Uploaded Image")
            st.success(msg)
        
        # Add a button to save the image as a screenshot
        if col2.button("Save Screenshot 📸", key="screenshot_image"):
            filename, msg = save_screenshot(processed_image, count)
            st.success(msg)

# Live Camera Option
elif option == "Live Camera":
    st.subheader("Live Attendance Tracking 🎥")
    st.write("Click 'Start Camera' to begin tracking.")
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    start = col1.button("Start Camera")
    stop = col2.button("Stop Camera")
    
    # Create columns for action buttons
    col3, col4, col5 = st.columns(3)
    save_button = col3.button("Save Attendance Record 📝")
    screenshot_button = col4.button("Take Screenshot 📸")
    
    # Placeholder for current count and frame
    current_count = st.empty()
    current_frame = st.empty()
    
    # Store current frame and count in session state
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    if 'current_student_count' not in st.session_state:
        st.session_state.current_student_count = 0
    
    if start:
        st.session_state.camera_running = True
        cap = cv2.VideoCapture(0)  # Open webcam
        
        while st.session_state.get('camera_running', True):
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame.")
                break
            
            # Process frame
            processed_frame, num_students = detect_people(frame)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
            
            # Update UI
            current_frame.image(processed_frame_rgb, caption="Live Attendance Tracking", use_container_width=True)
            current_count.write(f"### Current Attendance Count: {num_students} 🧑‍🎓")
            
            # Store the current frame and count in session state
            st.session_state.current_frame = processed_frame_rgb
            st.session_state.current_student_count = num_students
            
            # Stop the camera when button is pressed
            if stop:
                st.session_state.camera_running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Save current count when save button is pressed
    if save_button and st.session_state.current_frame is not None:
        msg = save_attendance(st.session_state.current_student_count, "Live Camera")
        st.success(msg)
    
    # Take screenshot when screenshot button is pressed
    if screenshot_button and st.session_state.current_frame is not None:
        filename, msg = save_screenshot(st.session_state.current_frame, st.session_state.current_student_count)
        st.success(msg)
        
        # Display the captured screenshot
        st.image(st.session_state.current_frame, caption=f"Captured Screenshot - {st.session_state.current_student_count} students", use_container_width=True)

# View Logs Option
elif option == "View Logs":
    st.subheader("Attendance Log History 📊")
    
    if not st.session_state.attendance_log.empty:
        st.dataframe(st.session_state.attendance_log, use_container_width=True)
        
        # Download button for logs
        csv = st.session_state.attendance_log.to_csv(index=False)
        st.download_button(
            label="Download Attendance Logs 📥",
            data=csv,
            file_name="attendance_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records found. Start recording attendance first.")
        
    # Clear logs button
    if st.button("Clear All Logs ❌"):
        st.session_state.attendance_log = pd.DataFrame(columns=['Date', 'Time', 'Count', 'Source'])
        if os.path.exists('attendance_logs/attendance_log.csv'):
            os.remove('attendance_logs/attendance_log.csv')
        st.success("All logs cleared successfully!")

# View Screenshots Option
elif option == "View Screenshots":
    st.subheader("Captured Screenshots 📸")
    
    # Load screenshot log if it exists
    screenshot_csv = 'attendance_logs/screenshot_log.csv'
    if os.path.exists(screenshot_csv):
        screenshot_df = pd.read_csv(screenshot_csv)
        st.session_state.screenshot_log = screenshot_df
    
    # Check if we have any screenshots
    if 'screenshot_log' in st.session_state and not st.session_state.screenshot_log.empty:
        # Display the screenshot log
        st.dataframe(st.session_state.screenshot_log, use_container_width=True)
        
        # Allow viewing individual screenshots
        st.subheader("View Individual Screenshots")
        
        # Get list of all screenshot filenames
        screenshot_files = st.session_state.screenshot_log['Filename'].tolist()
        selected_screenshot = st.selectbox("Select a screenshot to view:", screenshot_files)
        
        if os.path.exists(selected_screenshot):
            # Display the selected screenshot
            img = Image.open(selected_screenshot)
            st.image(img, caption=f"Screenshot: {selected_screenshot}", use_container_width=True)
        else:
            st.error(f"Screenshot file not found: {selected_screenshot}")
        
        # Button to delete selected screenshot
        if st.button("Delete This Screenshot"):
            if os.path.exists(selected_screenshot):
                os.remove(selected_screenshot)
                # Remove from the log
                st.session_state.screenshot_log = st.session_state.screenshot_log[
                    st.session_state.screenshot_log['Filename'] != selected_screenshot
                ]
                st.session_state.screenshot_log.to_csv(screenshot_csv, index=False)
                st.success(f"Screenshot {selected_screenshot} deleted successfully!")
                st.rerun()  # Refresh the page
        
        # Clear all screenshots button
        if st.button("Delete All Screenshots ❌"):
            for file in screenshot_files:
                if os.path.exists(file):
                    os.remove(file)
            st.session_state.screenshot_log = pd.DataFrame(columns=['Date', 'Time', 'Count', 'Filename'])
            if os.path.exists(screenshot_csv):
                os.remove(screenshot_csv)
            st.success("All screenshots deleted successfully!")
            st.rerun()  # Refresh the page
    else:
        st.info("No screenshots found. Take some screenshots first.")