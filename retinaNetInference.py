# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt1
import cv2
import os
import numpy as np
import time
import csv


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = r"E:\Projects\Stanford\CS221\Project\TAPProj\Snapshot\resnet50_csv_10_inf.h5"
labelPath = r"E:\Projects\Stanford\CS221\Project\TAPProj\Output\retinaNetTrainClass.csv"

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {}
with open(labelPath) as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        # An dictionary of "ID, found, location"
        labels_to_names[int(row[1])] = row[0]

# load image - NP array
image = read_image_bgr(r'F:\TrainingImages\train_00\dec9908f78a65aae.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
res = model.predict_on_batch(np.expand_dims(image, axis=0))
boxes, scores, labels  = res
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)

    caption = "Missing Label"
    if label in labels_to_names:
        caption = "{} {:.3f}".format(labels_to_names[label], score)

    draw_caption(draw, b, caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()