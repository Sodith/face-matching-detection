# import numpy as np
# from flask import Flask, request, jsonify, send_file, Blueprint
# import os
# import cv2
# from datetime import datetime



# # from onBoarding.onBoardingApp import app
# # Create a Blueprint for face detection
# pan_face_detection = Blueprint('pan_face_detection', __name__)
# UPLOAD_FOLDER = 'uploads'
# DETECTED_FOLDER = 'detected_faces'

# # Ensure the folders exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(DETECTED_FOLDER, exist_ok=True)

# # @pan_face_detection.route('/detect-face', methods=['POST'])
# # def detect_face():
# #     try:
# #         # Ensure the request contains an image file
# #         if 'image' not in request.files:
# #             return jsonify({'error': 'No image uploaded'}), 400

# #         file = request.files['image']
# #         filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
# #         file_path = os.path.join('uploads', filename)
# #         os.makedirs('uploads', exist_ok=True)
# #         os.makedirs('detected_faces', exist_ok=True)
# #         file.save(file_path)

# #         # Read the uploaded image (no conversion to grayscale)
# #         img = cv2.imread(file_path)
# #         if img is None:
# #             return jsonify({'error': 'Invalid image file'}), 400

# #         # Load Haar cascade for face detection (no gray conversion)
# #         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# #         # Detect faces
# #         # faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# #         faces = face_cascade.detectMultiScale(
# #                 img,
# #                 scaleFactor=1.1,       # Reduce this to detect smaller changes (e.g., 1.05)
# #                 minNeighbors=5,        # Increase this to reduce false positives
# #                 minSize=(20, 20),      # Reduce minSize for smaller faces
# #                 maxSize=(500, 500)     # Optional: Limit maximum size for large faces
# # )

# #         if len(faces) == 0:
# #             return jsonify({'error': 'No face detected'}), 400

# #         # Crop the first detected face
# #         x, y, w, h = faces[0]
# #         face_img = img[y:y+h, x:x+w]

# #         # Check if face_img is not empty
# #         if face_img.size == 0:
# #             return jsonify({'error': 'Failed to extract face from image'}), 500

# #         # Save the cropped face image as is (no resizing, no gray conversion)
# #         detected_file_path = os.path.abspath(os.path.join('detected_faces', filename))
# #         success = cv2.imwrite(detected_file_path, face_img)
# #         if not success:
# #             return jsonify({'error': 'Failed to save detected face'}), 500

# #         # Ensure the file exists before sending
# #         if not os.path.exists(detected_file_path):
# #             return jsonify({'error': 'File not found after processing'}), 500

# #         # Send the face image as is (no modification to the image)
# #         return send_file(detected_file_path, mimetype='image/jpeg')

# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # Load the DNN model
# prototxt_path = "deploy.prototxt"
# caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
# net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# @pan_face_detection.route('/detect-face', methods=['POST'])
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