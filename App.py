import os
import cv2
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
import pickle

App = Flask(__name__)

with open('brain_model','rb') as f:
    model = pickle.load(f)
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img):
    img = cv2.imread(img,0)
    img1 = cv2.resize(img,(64,64))
    img2 = img1.reshape(1,-1)/255
    result = model.predict(img2)
    return result[0]

@App.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@App.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file'] 
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    App.run(debug=True)