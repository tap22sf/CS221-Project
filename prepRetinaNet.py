import os
#import wget

# import keras
import keras

import sys
sys.path.append('.\\keras-retinanet')

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# Download weights for Retinanet
def downloadWeights(url, model):
    fullPath = os.path.join(os.path.join(os.getcwd(), "pretrained_models"))
    if not os.path.exists(fullPath):
        os.mkdir(fullPath)
        
    print ('Downloading trained weights: {} to : {}'.format(model, fullPath))
    #wget.download(url, fullPath)
    
def reFetchRetinaNet():
    # Check to see if any weigths need to be downloaded
    RCNN_COCO_MODEL_URL = "https://github.com/fizyr/keras-retinanet/releases/download/0.5.0/resnet50_coco_best_v2.1.0.h5"
    RCNN_COCO_MODEL = "resnet50_coco_best_v2.1.0.h5"
    downloadWeights(RCNN_COCO_MODEL_URL, RCNN_COCO_MODEL)


def createSession():
    def get_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    # use this environment flag to change which GPU to use
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    
def loadModel():
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    model_path = os.path.join('.', 'pretrained_models', 'resnet50_coco_best_v2.1.0.h5')

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    #model = models.convert_model(model)

    print(model.summary())
    return model

