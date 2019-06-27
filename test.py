import keras
import json
from keras.models import load_model
from keras.preprocessing import image
import sys, getopt
import numpy as np

def get_model(model_path):
    model = load_model(model_path)
    return model

if __name__ == "__main__":
    image_file = ""
    model_file = ""

    label = ['miku','rin','saber']
    label_num = 3

    # parse args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:m:",["image=","model="])
    except getopt.GetoptError:
        print("python test.py -i <image> -m <model.hdf5>")
    for opt, arg in opts:
        if opt == '-h':
            print("python test.py -i <image> -m <model.hdf5>")
            sys.exit()
        elif opt in ('-i', '--image'):
            image_file = arg
        elif opt in ('-m', '--model'):
            model_file = arg

    # load image
    img = image.load_img(image_file, target_size=(256, 256))

    # preprocessing img
    x = image.img_to_array(img)
    x = 1./255 * x
    x = np.expand_dims(x, axis=0)

    # get model
    m = get_model(model_file)

    # predict
    probs = m.predict(x)[0]

    res = []

    for i in range(label_num):
        res.append({label[i]: float(probs[i])})

    print(json.dumps(res))