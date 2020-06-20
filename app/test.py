import cv2
import numpy as np
import os
from keras.models import load_model

model = load_model('model79.h5')
img = cv2.imread(os.path.join('uploads', 'ss.png'))
resize_img = cv2.resize(img, (224, 224))
probs = model.predict(np.expand_dims(np.array(resize_img),0))
print(probs)
