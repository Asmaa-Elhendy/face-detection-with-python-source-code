from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import traceback
import requests
from io import BytesIO
import dlib
import json  # Ensure this import is included

app = Flask(__name__)



def download_image(url):
    response = requests.get(url)
    return BytesIO(response.content)

def perform_face_comparison(main_img, img_url):
    try:
        img_data_url = download_image(img_url)
        nparr_url = np.frombuffer(img_data_url.read(), np.uint8)
        img_url = cv2.imdecode(nparr_url, cv2.IMREAD_COLOR)

        rgb_main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        rgb_img_url = cv2.cvtColor(img_url, cv2.COLOR_BGR2RGB)

        face_locations_main_img = face_recognition.face_locations(rgb_main_img)
        face_locations_url_img = face_recognition.face_locations(rgb_img_url)

        face_encodings_main_img = face_recognition.face_encodings(rgb_main_img, face_locations_main_img)
        face_encodings_url_img = face_recognition.face_encodings(rgb_img_url, face_locations_url_img)

        for face_encoding_main_img in face_encodings_main_img:
            for face_encoding_url_img in face_encodings_url_img:
                match = face_recognition.compare_faces([face_encoding_main_img], face_encoding_url_img)
                if match[0]:
                    return True
        return False
    except Exception as e:
        print("Error in face comparison:", traceback.format_exc())
        return False
    #this api perform comparing between main image provided and image urls of each person in json data(employess)
#and return matched persons
@app.route('/compare_faces_list', methods=['POST'])
def compare_faces_list_endpoint():
    try:
        main_image_file = request.files.get('image')
        if not main_image_file:
            return jsonify({'error': 'No main image file provided'})
        main_image_data = main_image_file.read()
        nparr_main = np.frombuffer(main_image_data, np.uint8)
        main_img = cv2.imdecode(nparr_main, cv2.IMREAD_COLOR)

        json_data = request.form.get('json')
        if not json_data:
            return jsonify({'error': 'No JSON data provided'})

        data = json.loads(json_data)
        face_list = data.get('face_list', [])
        if not face_list:
            return jsonify({'error': 'Invalid or missing "face_list" in JSON data'})

        results = []
        for person in face_list:
            if 'image_url' in person:
                match_status = perform_face_comparison(main_img, person['image_url'])
                if match_status:  # Only append if match status is true
                    results.append({
                        'id': person.get('id'),
                        'name': person.get('name'),
                        'match': match_status,
                        'image_url': person['image_url'],
                        'position': person.get('position')  # Include the position from the JSON input
                    })
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})
    #new face encodonds of provided person image and return list of floats
@app.route('/recognizepersonface', methods=['POST'])
def recognize_person_face():
    try:
        main_image_file = request.files.get('image')
        if not main_image_file:
            return jsonify({'error': 'No main image file provided'})
        main_image_data = main_image_file.read()
        nparr_main = np.frombuffer(main_image_data, np.uint8)
        main_img = cv2.imdecode(nparr_main, cv2.IMREAD_COLOR)

        # Convert image to RGB and find face locations and encodings
        rgb_main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_main_img)
        face_encodings = face_recognition.face_encodings(rgb_main_img, face_locations)

        if face_encodings:
            return jsonify({'face_encoding': face_encodings[0].tolist()})  # Return the first face encoding
        else:
            return jsonify({'error': 'No faces detected'})
    except Exception as e:
        return jsonify({'error': str(e)})

    #new    compare provided image file with face encodings of persons to save time 


@app.route('/comparefaceArrays', methods=['POST'])
def compare_face_arrays():
    try:
        main_image_file = request.files.get('image')
        if not main_image_file:
            return jsonify({'error': 'No main image file provided'}), 400
        main_image_data = main_image_file.read()
        nparr_main = np.frombuffer(main_image_data, np.uint8)
        main_img = cv2.imdecode(nparr_main, cv2.IMREAD_COLOR)

        json_data = request.form.get('json')
        if not json_data:
            return jsonify({'error': 'No JSON data provided or "face_arrays" missing'}), 400

        data = json.loads(json_data)
        face_arrays = data.get('face_arrays', [])

        rgb_main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        face_locations_main_img = face_recognition.face_locations(rgb_main_img)
        face_encodings_main_img = face_recognition.face_encodings(rgb_main_img, face_locations_main_img)

        results = []
        if face_encodings_main_img:
            main_face_encoding = face_encodings_main_img[0]  # Use the first face found
            for person in face_arrays:
                matches = face_recognition.compare_faces([np.array(person['face_encoding'])], main_face_encoding)
                match = any(matches)  # Convert numpy bool_ to Python bool
                if match:
                    results.append({
                        'id': person['id'],
                        'name': person['name'],
                        'match': bool(match),  # Explicitly convert numpy bool_ to Python bool
                        'image_url': person['image_url'],
                        'position': person['position']
                    })
            return jsonify(results)
        else:
            return jsonify({'error': 'No faces found in main image'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
