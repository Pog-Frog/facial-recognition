import os
import numpy as np
from PIL import Image
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

tmp_id = 0
person_id = {}
yLabel = []
x_train = []

##converitng each image in the image directory to a numeric array of each pixel in each picture after converting the pricture ao gray scale (numpy array)
## in order to be usable to train the model
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            tmp = os.path.basename(root).replace(" ", "-").lower()
            print(tmp, path)
            if tmp in person_id:
                pass
            else:
                person_id[tmp] = tmp_id
                tmp_id += 1
            id_ = person_id[tmp]
            print(person_id)
            pil_img = Image.open(path).convert("L")
            img_arr = np.array(pil_img, "uint8")
            print(img_arr)
            faces = face_cascade.detectMultiScale(img_arr, scaleFactor = 1.5, minNeighbors = 5)
            for(x, y, w, h) in faces:
                region_of_interest = img_arr[y:y+h, x:x+w]
                x_train.append(region_of_interest)
                yLabel.append(id_)

#print(yLabel)
#print(x_train)

with open("labels.pkl", 'wb') as f:
    pickle.dump(person_id, f)