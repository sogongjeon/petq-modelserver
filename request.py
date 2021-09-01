import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "./image.jpg"

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()

if r["success"]:
    for idx, item in enumerate(r["results"]):
        print(f'{idx}: {item["kind"]} - {item["value"]}%')
else:
    print("Request failed")
