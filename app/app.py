import numpy as np
import cv2
import os
from keras.models import load_model
from flask import Flask, request, url_for, redirect, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# generalized response formats
def success_response(data, code=200):
    return json.dumps({"success": True, "data": data}), code

def failure_response(message, code=404):
    return json.dumps({"success": False, "error": message}), code

### ROUTES
@app.route('/')
def homepage():
    # return index.html
    return render_template('index.html')

@app.route('/', methods=['POST'])
def uploadphoto():
    # redirect to prediction
    file = request.files['file']
    filename = secure_filename(file.filename)
    imgurl = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(imgurl)
    return redirect(url_for('prediction', filename=filename))

@app.route('/prediction/<filename>')
def prediction(filename):
    # read and reshape image
    imgurl = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(imgurl)
    resize_img = cv2.resize(img, (224, 224))

    # get prediction
    model = load_model('model79.h5')
    probs = model.predict(np.expand_dims(np.array(resize_img),0))
    classes = probs.argmax(axis=-1)
    encoding = {'NORMAL': 0, 'PNEUMONIA': 1, 'COVID-19': 2}
    encoded_list = list(encoding.values())
    class_list = list(encoding.keys())
    y = class_list[encoded_list.index(classes)]

    return render_template('prediction.html', imgurl=imgurl, predictions=str(y))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
