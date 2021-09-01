# Load libraries
import os
import io
import pickle
import numpy as np
import flask
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils


''' Global value '''
MODEL_PATH = None
METRIC = None
CLASS = None
IMAGE_SIZE = (224, 224)

with open("metric.pkl", "rb") as f:
    METRIC = pickle.load(f)

with open("class.pkl", "rb") as f:
    CLASS = pickle.load(f)

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    MODEL_PATH = './model.h5'

MODEL = load_model(MODEL_PATH, custom_objects=METRIC)

app = flask.Flask(__name__)


def prepare_image(image, target):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image, mode='tf')

    return image


def decode_predictions(preds, top=3, **kwargs):
    results = {CLASS[idx]: value for idx, value in enumerate(preds)}
    results = sorted(results.items(), key=(lambda x: x[1]), reverse=True)
    return results[:top]


@ app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}

    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, target=IMAGE_SIZE)

        preds = MODEL.predict(image)[0]
        results = decode_predictions(preds)
        data["results"] = [{'kind': key, 'value': int(value * 100)} for key, value in results]
        data["success"] = True

        return flask.jsonify(data)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    app.run(host='0.0.0.0')
