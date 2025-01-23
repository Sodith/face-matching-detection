import base64
from datetime import time
from socket import SocketIO

import numpy as np
from flask import Flask
# from flask import Flask, render_template, send_file
# from flask_socketio import SocketIO, emit
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from base64 import b64decode
# import face_recognition
# import uuid
# import pickle
# import face_recognition
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# from face_recognition import face_distance
# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import face_recognition
# import pickle
# import base64
# import pickle
# import os
# from concurrent.futures import ThreadPoolExecutor
#
# from matplotlib.pyplot import gray
#
# # from onBoarding.panfacedetection import pan_face_detection

# Flask and SocketIO setup



app = Flask(__name__)
# socketio = SocketIO(app,
#                     cors_allowed_origins="*",
#                     ping_timeout=20000,  # Increased timeout
#                     ping_interval=10000,  # Increased interval
#                     max_http_buffer_size=100000000,  # Increased buffer size
#                     async_mode='threading',  # Use threading mode
#                     logger=True,  # Enable logging
#                     engineio_logger=True)  # Enable engine logging
# # Import the face detection routes
# # Import the face detection routes
# # app.register_blueprint(pan_face_detection)
# # MediaPipe Face Detection setup
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
#
# # Directory to save face images
# OUTPUT_FACE_DIR = "FaceImages"
# os.makedirs(OUTPUT_FACE_DIR, exist_ok=True)
# OUTPUT_VIDEO_DIR = "VideoImage"
# os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
# # Directories for reference images
# REFERENCE_IMAGES_DIR = "FaceDB"
#
# # Ensure the reference directory exists
# os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)
#
# # Global variables
# MAX_FACES = 100
# face_count = 0
# confidence_level = 0.5
# current_user_dir = None
# known_encodings = []
# known_employee_ids = []
#
# # Global log counter
# log_counter = 0
#
# # Global variables
# cap = None
# is_capturing = False
#
# # Load reference images
# # def load_reference_images():
# #     global known_encodings, known_employee_ids
# #     known_encodings = []
# #     known_employee_ids = []
# #
# #     for employee_id in os.listdir(OUTPUT_FACE_DIR):
# #         employee_path = os.path.join(OUTPUT_FACE_DIR, employee_id)
# #         if os.path.isdir(employee_path):
# #             for file in os.listdir(employee_path):
# #                 if file.endswith(('.png', '.jpg', '.jpeg')):
# #                     filepath = os.path.join(employee_path, file)
# #                     image = face_recognition.load_image_file(filepath)
# #                     encodings = face_recognition.face_encodings(image)
# #                     if encodings:
# #                         known_encodings.append(encodings[0])
# #                         known_employee_ids.append(employee_id)
# #
# #
# # # Load reference images on app startup
# # load_reference_images()
#
# import os
# import face_recognition
# import pickle
#
# OUTPUT_FACE_DIR = "faces"
# os.makedirs(OUTPUT_FACE_DIR, exist_ok=True)
# ENCODINGS_FILE = "encodings.pkl"
#
# # def load_reference_images():
# #     global known_encodings, known_employee_ids
#
# #     # Load existing encodings if the file exists
# #     if os.path.exists(ENCODINGS_FILE):
# #         with open(ENCODINGS_FILE, "rb") as file:
# #             data = pickle.load(file)
# #             known_encodings = data["encodings"]
# #             known_employee_ids = data["employee_ids"]
# #     else:
# #         # Ensure the encodings file exists
# #         if not os.path.exists(ENCODINGS_FILE):
# #             with open(ENCODINGS_FILE, "wb") as file:
# #                 # Write an empty structure to initialize the file
# #                 pickle.dump({"encodings": [], "employee_ids": []}, file)
# #             print(f"Created empty encodings file: {ENCODINGS_FILE}")
#
# #     # Check for new images and update the file if necessary
# #     updated = False
# #     for employee_id in os.listdir(OUTPUT_FACE_DIR):
# #         employee_path = os.path.join(OUTPUT_FACE_DIR, employee_id)
# #         if os.path.isdir(employee_path):
# #             for file in os.listdir(employee_path):
# #                 if file.endswith(('.png', '.jpg', '.jpeg')):
# #                     filepath = os.path.join(employee_path, file)
#
# #                     # Skip if already encoded
# #                     if filepath in known_employee_ids:
# #                         continue
#
# #                     # Process new image
# #                     image = face_recognition.load_image_file(filepath)
# #                     encodings = face_recognition.face_encodings(image)
# #                     if encodings:
# #                         known_encodings.append(encodings[0])
# #                         known_employee_ids.append(filepath)
# #                         updated = True
#
# #     # Save updated encodings back to the file
# #     if updated:
# #         with open(ENCODINGS_FILE, "wb") as file:
# #             pickle.dump({"encodings": known_encodings, "employee_ids": known_employee_ids}, file)
#
# #     print(f"Loaded {len(known_encodings)} face encodings.")
#
#
# # def load_reference_images():
# #     global known_encodings, known_employee_ids
#
# #     # Load existing encodings if the file exists
# #     if os.path.exists(ENCODINGS_FILE):
# #         with open(ENCODINGS_FILE, "rb") as file:
# #             data = pickle.load(file)
# #             known_encodings = data["encodings"]
# #             known_employee_ids = data["employee_ids"]
# #     else:
# #         # Ensure the encodings file exists
# #         if not os.path.exists(ENCODINGS_FILE):
# #             with open(ENCODINGS_FILE, "wb") as file:
# #                 # Write an empty structure to initialize the file
# #                 pickle.dump({"encodings": [], "employee_ids": []}, file)
# #             print(f"Created empty encodings file: {ENCODINGS_FILE}")
# #         known_encodings = []
# #         known_employee_ids = []
#
# #     # Clean up encodings for deleted files
# #     valid_encodings = []
# #     valid_employee_ids = []
# #     for filepath in known_employee_ids:
# #         if os.path.exists(filepath):
# #             valid_employee_ids.append(filepath)
# #             valid_encodings.append(known_encodings[known_employee_ids.index(filepath)])
#
# #     # Update global variables with valid entries
# #     known_encodings = valid_encodings
# #     known_employee_ids = valid_employee_ids
#
# #     # Check for new images and update the file if necessary
# #     updated = False
# #     for employee_id in os.listdir(OUTPUT_FACE_DIR):
# #         employee_path = os.path.join(OUTPUT_FACE_DIR, employee_id)
# #         if os.path.isdir(employee_path):
# #             for file in os.listdir(employee_path):
# #                 if file.endswith(('.png', '.jpg', '.jpeg')):
# #                     filepath = os.path.join(employee_path, file)
#
# #                     # Skip if already encoded
# #                     if filepath in known_employee_ids:
# #                         continue
#
# #                     # Process new image
# #                     image = face_recognition.load_image_file(filepath)
# #                     encodings = face_recognition.face_encodings(image)
# #                     if encodings:
# #                         known_encodings.append(encodings[0])
# #                         known_employee_ids.append(filepath)
# #                         updated = True
#
# #     # Save updated encodings back to the file
# #     if updated or len(valid_encodings) != len(data["encodings"]):
# #         with open(ENCODINGS_FILE, "wb") as file:
# #             pickle.dump({"encodings": known_encodings, "employee_ids": known_employee_ids}, file)
#
# #     print(f"Loaded {len(known_encodings)} face encodings.")
# import os
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# import face_recognition
#
# MAX_WORKERS = 4  # Adjust based on system capabilities
# MAX_IMAGES_PER_USER = 100  # Limit the number of images to process per user
#
#
# def process_user_directory(user_directory, known_encodings, known_employee_ids, file_timestamps):
#     """Processes all images in a user's directory and updates encodings."""
#     updated = False
#     user_encodings = []
#     user_filepaths = []
#
#     # Iterate through images in the user's directory
#     with os.scandir(user_directory) as files:
#         images = [f for f in files if f.name.endswith(('.png', '.jpg', '.jpeg'))]
#         images = sorted(images, key=lambda x: x.name)[:MAX_IMAGES_PER_USER]  # Limit processed images
#
#         for image in images:
#             filepath = image.path
#             file_mtime = image.stat().st_mtime
#
#             # Skip files that are already processed and not modified
#             if filepath in file_timestamps and file_timestamps[filepath] == file_mtime:
#                 continue
#
#             # Process the image
#             try:
#                 img = face_recognition.load_image_file(filepath)
#                 encodings = face_recognition.face_encodings(img)
#                 if encodings:
#                     user_encodings.append(encodings[0])
#                     user_filepaths.append(filepath)
#                     file_timestamps[filepath] = file_mtime
#                     updated = True
#             except Exception as e:
#                 print(f"Error processing file {filepath}: {e}")
#
#     if updated:
#         known_encodings.extend(user_encodings)
#         known_employee_ids.extend(user_filepaths)
#
#     return updated
#
#
# def load_reference_images():
#     global known_encodings, known_employee_ids, file_timestamps
#
#     # Load existing encodings if the file exists
#     if os.path.exists(ENCODINGS_FILE):
#         with open(ENCODINGS_FILE, "rb") as file:
#             data = pickle.load(file)
#             known_encodings = data["encodings"]
#             known_employee_ids = data["employee_ids"]
#             file_timestamps = data.get("file_timestamps", {})
#     else:
#         known_encodings = []
#         known_employee_ids = []
#         file_timestamps = {}
#
#     updated = False
#     user_directories = []
#
#     # Collect all user directories
#     with os.scandir(OUTPUT_FACE_DIR) as entries:
#         user_directories = [entry.path for entry in entries if entry.is_dir()]
#
#     # Process each user's directory in parallel
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         results = executor.map(
#             process_user_directory,
#             user_directories,
#             [known_encodings] * len(user_directories),
#             [known_employee_ids] * len(user_directories),
#             [file_timestamps] * len(user_directories)
#         )
#
#     if any(results):
#         updated = True
#
#     # Save updated encodings back to the file if any changes occurred
#     if updated:
#         with open(ENCODINGS_FILE, "wb") as file:
#             pickle.dump({
#                 "encodings": known_encodings,
#                 "employee_ids": known_employee_ids,
#                 "file_timestamps": file_timestamps
#             }, file)
#
#     print(f"Loaded {len(known_encodings)} face encodings across {len(user_directories)} users.")
#
#
# # Load reference images on app startup
#
#
# # Load data from the file
# def load_encodings():
#     global known_encodings, known_employee_ids
#
#     if os.path.exists(ENCODINGS_FILE):
#         with open(ENCODINGS_FILE, "rb") as file:
#             data = pickle.load(file)
#             known_encodings = data.get("encodings", [])[::-1]  # Reverse the encodings list
#             known_employee_ids = data.get("employee_ids", [])[::-1]  # Reverse the employee IDs list
#             print(f"Loaded {len(known_encodings)} encodings from the file (latest first).")
#     else:
#         known_encodings = []
#         known_employee_ids = []
#         print("Encodings file not found. No data loaded.")
#
#
# refresh_in_progress = False
#
#
# @socketio.on("refresh_data")
# def handle_refresh_data():
#     global refresh_in_progress
#
#     if refresh_in_progress:
#         print("Refresh already in progress. Ignoring the request.")
#         return  # Ignore if refresh is already in progress
#
#     print("Received refresh request from client.")
#     refresh_in_progress = True  # Set flag to prevent further refreshes
#
#     try:
#         load_reference_images()  # Call the function to reload reference images
#         load_encodings()  # Call the function to reload encodings
#         print("Reference123 images and encodings reloaded.")
#         emit("refresh_complete")  # Notify client that refresh is complete
#         print("Reference images and encodings reloaded.")
#     finally:
#         refresh_in_progress = False  # Reset flag once processing is done
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# @app.route("/match.html")
# def match():
#     return render_template("match.html")
#
@app.route("/")
def pan():
    return render_template("pan.html")
#
# @socketio.on("start_capture")
# def start_capture(data):
#     global face_count, current_user_dir, log_counter
#     log_counter += 1
#     username = data.get("username")
#     if username:
#         current_user_dir = os.path.join(OUTPUT_FACE_DIR, username)
#         os.makedirs(current_user_dir, exist_ok=True)
#         face_count = 0
#         emit("status", {"message": f"[Log {log_counter}] Capture started for {username}", "completed": False})
#
#
# @socketio.on("stop_capture")
# def stop_capture():
#     global current_user_dir, log_counter, video_writer
#     log_counter += 1
#
#     if video_writer is not None:
#         video_writer.release()
#         video_writer = None
#
#     current_user_dir = None
#     emit("status", {"message": f"[Log {log_counter}] Capture stopped", "completed": True})
#
#
# # Add this near other global variables
# current_frame = None
#
#
# @socketio.on("send_frame")
# def handle_frame(data):
#     global face_count, current_user_dir, log_counter, video_writer, current_frame
#     log_counter += 1
#
#     # Store the current frame
#     current_frame = data["image"]
#
#     if face_count >= MAX_FACES or not current_user_dir:
#         if video_writer is not None:
#             video_writer.release()
#             video_writer = None
#             # Send the last captured frame back to frontend
#             # if current_frame:
#             #     emit("last_frame", {"image": current_frame, "completed": True})
#         emit("status", {"message": f"[Log {log_counter}] Face capture complete", "completed": True})
#         return
#
#     if current_user_dir is None:
#         emit("status", {"message": f"[Log {log_counter}] No active capture session", "completed": True})
#         return
#
#     # Decode the received image
#     img_data = b64decode(data["image"].split(",")[1])
#     np_arr = np.frombuffer(img_data, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#     # Initialize VideoWriter if not already initialized
#     if 'video_writer' not in globals() or video_writer is None:
#         video_filename = os.path.join(OUTPUT_VIDEO_DIR, f"{data['username']}.mp4")
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         height, width, _ = frame.shape
#         video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
#
#     frame = cv2.resize(frame, (640, 480))
#     video_writer.write(frame)
#
#     # Face detection
#     with mp_face_detection.FaceDetection(
#             model_selection=0, min_detection_confidence=confidence_level
#     ) as face_detection:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(rgb_frame)
#
#         face_detected = False
#         if results.detections:
#             detection = results.detections[0]
#             bbox = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x_min = max(0, int(bbox.xmin * w))
#             y_min = max(0, int(bbox.ymin * h))
#             box_width = int(bbox.width * w)
#             box_height = int(bbox.height * h)
#
#             # Send face coordinates to frontend for drawing
#             face_coords = {
#                 "x": x_min,
#                 "y": y_min,
#                 "width": box_width,
#                 "height": box_height
#             }
#             emit("face_detected", face_coords)
#             face_detected = True
#
#             # Crop and save face
#             face = frame[y_min:y_min + box_height, x_min:x_min + box_width]
#             face_resized = cv2.resize(face, (300, 300))
#             face_filename = os.path.join(current_user_dir, f"face_{face_count + 1}.jpg")
#             cv2.imwrite(face_filename, face_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#             face_count += 1
#             # Draw bounding box on the new code1
#             if face_detected:
#                cv2.rectangle(
#                 frame,
#                 (x_min, y_min),
#                 (x_min + box_width, y_min + box_height),
#                 (0, 255, 0),  # Green rectangle
#                 2  # Thickness of the rectangle
#             )
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
#             emit("status", {"message": f"[Log {log_counter}] Saved face {face_count}","image": frame_base64, "completed": False})
#             # emit("status", {"message": f"[Log {log_counter}] Saved face {face_count}", "completed": False})
#
#         if not face_detected:
#             # Tell frontend no face was detected
#             emit("face_detected", None)
#
#     # Modify the final emit in your backend to include both image and coordinates
#     if face_count >= MAX_FACES:
#         if video_writer is not None:
#             video_writer.release()
#             video_writer = None
#         # Convert the current frame to base64 for sending
#         if face_detected:
#             cv2.rectangle(
#                 frame,
#                 (x_min, y_min),
#                 (x_min + box_width, y_min + box_height),
#                 (0, 255, 0),  # Green rectangle
#                 2  # Thickness of the rectangle
#             )
#         # if face_detected:
#         #     cv2.ellipse(
#         #         frame,
#         #         (x_min + box_width // 2, y_min + box_height // 2),  # Center
#         #         (box_width // 2, box_height // 2),  # Axes lengths
#         #         0,  # Angle of rotation
#         #         0,  # Start angle
#         #         360,  # End angle
#         #         (0, 255, 255),  # Yellow ellipse
#         #         2  # Thickness
#         #     )
#         # if face_detected:
#         #     overlay = frame.copy()
#         #     cv2.rectangle(
#         #         overlay,
#         #         (x_min, y_min),
#         #         (x_min + box_width, y_min + box_height),
#         #         (0, 255, 0),  # Green color
#         #         -1  # Negative thickness means fill the rectangle
#         #     )
#         #     # Add transparency
#         #     alpha = 0.5
#         #     frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
#
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
#         emit("status", {
#             "message": f"[Log {log_counter}] Face capture complete",
#             "image": frame_base64,
#             # "face_coords": face_coords,
#             "completed": True
#         })
#
#
# # def generate_frames():
# #     global is_capturing, cap
# #
# #     # Initialize MediaPipe Face Detection
# #     with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
# #         while True:
# #             if not is_capturing or cap is None:
# #                 continue
# #
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #
# #             # Convert frame to RGB for face recognition
# #             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #
# #             # Detect faces
# #             face_locations = face_recognition.face_locations(rgb_frame)
# #             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
# #
# #             for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# #                 # Match face encoding to known encodings
# #                 matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
# #                 employee_id = "Unknown"
# #
# #                 if True in matches:
# #                     match_index = matches.index(True)
# #                     employee_id = known_employee_ids[match_index]
# #
# #                 # Draw bounding box and employee ID
# #                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
# #                 cv2.putText(frame, employee_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# #
# #             # Encode the frame to JPEG
# #             _, buffer = cv2.imencode('.jpg', frame)
# #             frame = buffer.tobytes()
# #
# #             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
# @socketio.on("start_match")
# def start_match():
#     global cap, is_capturing
#     emit("status", {"message": "Match started"})
#
#
# @socketio.on("stop_match")
# def stop_match():
#     global cap, is_capturing
#     # is_capturing = False
#     # if cap is not None:
#     #     cap.release()
#     #     cap = None
#     emit("status", {"message": "Match stopped"})
#
#
# @socketio.on("send_match_frame")
# def handle_match_frame(data):
#     try:
#         # Decode the received base64 image
#         img_data = b64decode(data["image"].split(",")[1])
#         np_arr = np.frombuffer(img_data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#         if frame is None:
#             print("Error: Received empty frame")
#             emit("match_result", {"result": "Error: Invalid frame"})
#             return
#
#         # Convert frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Detect faces
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#         if not face_locations:
#             emit("match_result", {"result": "No faces detected"})
#             return
#
#         matched_user = "Unknown"
#         match_found = False
#         bounding_box = None
#
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             # Compare face encoding with known encodings
#             matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
#
#             if True in matches:
#                 match_index = matches.index(True)
#                 matched_user = known_employee_ids[match_index]
#                 match_found = True
#                 bounding_box = {"x": left, "y": top, "width": right - left, "height": bottom - top}
#
#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#                 cv2.putText(frame, matched_user, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#                 break  # Stop after the first match
#
#         # Encode the frame to base64 to send back for debugging (optional)
#         _, buffer = cv2.imencode('.jpg', frame)
#         encoded_frame = base64.b64encode(buffer).decode("utf-8")
#
#         if match_found:
#             emit("match_result", {
#                 "result": "Match found",
#                 "user": matched_user,
#                 "Confidence": "High",
#                 "frame": f"data:image/jpeg;base64,{encoded_frame}",
#                 "bounding_box": bounding_box  # Include bounding box information
#             })
#         else:
#             emit("match_result", {
#                 "result": "No match found",
#                 "frame": f"data:image/jpeg;base64,{encoded_frame}"  # Optional: Debugging visualization
#             })
#
#     except Exception as e:
#         print(f"Error in face matching: {str(e)}")
#         emit("match_result", {"result": f"Error: {str(e)}"})
#
#
# @socketio.on('connect')
# def handle_connect():
#     print("Client connected")
#     emit('status', {'message': 'Connected to server'})
#
#
# @socketio.on('disconnect')
# def handle_disconnect():
#     print("Client disconnected")
#
#
# @socketio.on_error()
# def error_handler(e):
#     print(f"SocketIO error: {str(e)}")
#     emit('status', {'message': f'Server error occurred: {str(e)}'})
#

#################################################pan card####################################################
from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
from datetime import datetime
UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'detected_faces'

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# @app.route('/detect-face', methods=['POST'])
# def detect_face():
#     try:
#         # Ensure the request contains an image file
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image uploaded'}), 400

#         file = request.files['image']
#         filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
#         file_path = os.path.join('uploads', filename)
#         os.makedirs('uploads', exist_ok=True)
#         os.makedirs('detected_faces', exist_ok=True)
#         file.save(file_path)

#         # Read the uploaded image (no conversion to grayscale)
#         img = cv2.imread(file_path)
#         if img is None:
#             return jsonify({'error': 'Invalid image file'}), 400

#         # Load Haar cascade for face detection (no gray conversion)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#         # Detect faces
#         # faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         faces = face_cascade.detectMultiScale(
#                 img,
#                 scaleFactor=1.1,       # Reduce this to detect smaller changes (e.g., 1.05)
#                 minNeighbors=5,        # Increase this to reduce false positives
#                 minSize=(20, 20),      # Reduce minSize for smaller faces
#                 maxSize=(500, 500)     # Optional: Limit maximum size for large faces
# )

#         if len(faces) == 0:
#             return jsonify({'error': 'No face detected'}), 400

#         # Crop the first detected face
#         x, y, w, h = faces[0]
#         face_img = img[y:y+h, x:x+w]

#         # Check if face_img is not empty
#         if face_img.size == 0:
#             return jsonify({'error': 'Failed to extract face from image'}), 500

#         # Save the cropped face image as is (no resizing, no gray conversion)
#         detected_file_path = os.path.abspath(os.path.join('detected_faces', filename))
#         success = cv2.imwrite(detected_file_path, face_img)
#         if not success:
#             return jsonify({'error': 'Failed to save detected face'}), 500

#         # Ensure the file exists before sending
#         if not os.path.exists(detected_file_path):
#             return jsonify({'error': 'File not found after processing'}), 500

#         # Send the face image as is (no modification to the image)
#         return send_file(detected_file_path, mimetype='image/jpeg')

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# Load the DNN model
import os

# Get the absolute path of the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths
prototxt_path = os.path.join(os.getcwd(), "deploy.prototxt")
caffemodel_path = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")


# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

#working -but some roation not working ----version--2
# @app.route('/detect-face', methods=['POST'])
# def detect_face():
#     try:
#         # Ensure the request contains an image file
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image uploaded'}), 400

#         file = request.files['image']
#         filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
#         file_path = os.path.join('uploads', filename)
#         os.makedirs('uploads', exist_ok=True)
#         os.makedirs('detected_faces', exist_ok=True)
#         file.save(file_path)

#         # Read the uploaded image
#         img = cv2.imread(file_path)
#         if img is None:
#             return jsonify({'error': 'Invalid image file'}), 400

#         # Prepare the image for DNN face detection
#         (h, w) = img.shape[:2]
#         blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()

#         # Process detections
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.5:  # Confidence threshold
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (x1, y1, x2, y2) = box.astype("int")

#                 # Ensure coordinates are within the image boundaries
#                 x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

#                 # Crop the detected face
#                 face_img = img[y1:y2, x1:x2]

#                 # Check if face_img is not empty
#                 if face_img.size == 0:
#                     continue

#                 # Save the cropped face image
#                 detected_file_path = os.path.abspath(os.path.join('detected_faces', filename))
#                 success = cv2.imwrite(detected_file_path, face_img)
#                 if not success:
#                     return jsonify({'error': 'Failed to save detected face'}), 500

#                 # Ensure the file exists before sending
#                 if not os.path.exists(detected_file_path):
#                     return jsonify({'error': 'File not found after processing'}), 500

#                 # Send the face image as is (no modification to the image)
#                 return send_file(detected_file_path, mimetype='image/jpeg')

#         # If no face is detected
#         return jsonify({'error': 'No face detected'}), 400

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

#version======================3
@app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        # Ensure the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        file_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('detected_faces', exist_ok=True)
        file.save(file_path)

        # Read the uploaded image
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Prepare the image for DNN face detection
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Process DNN detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure coordinates are within the image boundaries
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # Crop the detected face
                face_img = img[y1:y2, x1:x2]

                # Check if face_img is not empty
                if face_img.size == 0:
                    continue

                # Save the cropped face image
                detected_file_path = os.path.abspath(os.path.join('detected_faces', filename))
                success = cv2.imwrite(detected_file_path, face_img)
                if not success:
                    return jsonify({'error': 'Failed to save detected face'}), 500

                # Send the face image
                return send_file(detected_file_path, mimetype='image/jpeg')

        # If no face is detected by DNN, use fallback detection with Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for Haar detection
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Use the first detected face
            (x, y, w, h) = faces[0]
            face_img = img[y:y+h, x:x+w]

            # Save the cropped face image
            detected_file_path = os.path.abspath(os.path.join('detected_faces', filename))
            success = cv2.imwrite(detected_file_path, face_img)
            if not success:
                return jsonify({'error': 'Failed to save detected face'}), 500

            # Send the face image
            return send_file(detected_file_path, mimetype='image/jpeg')

        # If no face is detected by both methods
        return jsonify({'error': 'No face detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    

if __name__ == "__main__":
    app.run(
        debug=True,
        host='0.0.0.0',  # Allow external connections
        port=5000
    )

    # socketio.run(app,
    #              debug=True,
    #              host='0.0.0.0',  # Allow external connections
    #              port=5000,
    #              allow_unsafe_werkzeug=True)  # Add this if using Werkzeug 2.3+
