from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

# Dictionary to store known face vectors and corresponding names
known_faces = {}

# Load known faces at the start of the application
def load_known_faces():
    print('hi')
    known_faces["John"] = face_recognition.face_encodings(face_recognition.load_image_file("john.jpg"))[0]
    print(known_faces["John"])
    # Add more entries as needed

load_known_faces()

def recognize_face(img):
    # Find face locations using face_recognition library
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # Process the results
    result = {'result': 'No face detected', 'name': None, 'vectors': None}

    if len(face_locations) > 0:
        # Flag to check if any face is recognized
        recognized = False

        # Compare each detected face encoding with known face encodings
        for face_encoding in face_encodings:
            for name, known_face_encoding in known_faces.items():
                # Compare the face encodings using face_recognition.compare_faces
                if face_recognition.compare_faces([known_face_encoding], face_encoding)[0]:
                    result = {'result': 'Face recognized', 'name': name, 'vectors': face_encoding.tolist()}
                    recognized = True
                    break

            # If any face is recognized, break out of the loop
            if recognized:
                break

        # If no face is recognized, update the result
        if not recognized:
            result = {'result': 'Face detected but not recognized', 'name': None, 'vectors': face_encodings[0].tolist()}

    return result

@app.route('/recognize_face', methods=['POST'])
def recognize_face_endpoint():
    try:
        # Receive image from Flutter app
        image_data = request.files['image'].read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform face recognition
        result = recognize_face(img)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
