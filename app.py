import json
import cv2
import os
from imutils import paths
import face_recognition
import pickle
from flask import Flask, request, jsonify, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from face_detection import compare_faces
UPLOAD_FOLDER = 'Images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def generate_dataset():
    imagePaths = list(paths.list_images('student_cards'))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    #save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    f = open("dataset/face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()         
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Student card file is required')
            return redirect(request.url)
        file = request.files['file']
        studentId = request.form.get('student_id')
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No student card file selected file')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imagePath = app.config['UPLOAD_FOLDER']+ '/'+ filename
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(500, 500)) 
            print("Found {0} Faces!".format(len(faces)))
            for (x, y, w, h) in faces: 
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = image[y:y + h, x:x + w] 
                print("[INFO] Object found. Saving locally.") 
                path = 'student_cards/'+ studentId
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                cv2.imwrite('student_cards/'+ studentId +'/face.jpg', roi_color) 
                os.remove(imagePath)
            generate_dataset()
            #status = cv2.imwrite('student_cards/faces_detected.jpg', image)
            #print ("Image faces_detected.jpg written to filesystem: ",status)
	     
            
    return render_template('upload.html')

@app.route('/webcam', methods=['GET', 'POST'])
def upload_cam_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'cam_image_file' not in request.files:
            flash('Webcam image is required')
            return redirect(request.url)
        cam_file = request.files['cam_image_file']
        studentId = request.form.get('student_id')
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if cam_file.filename == '':
            flash('No webcam image selected file')
            return redirect(request.url)
        if allowed_file(cam_file.filename):
            compare_faces(cam_file, studentId)    
            #status = cv2.imwrite('student_cards/faces_detected.jpg', image)
            #print ("Image faces_detected.jpg written to filesystem: ",status)
	     
            
    return render_template('cam_template.html')





app.run(debug=True)
