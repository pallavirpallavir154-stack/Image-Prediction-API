import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

model = MobileNetV2(weights="imagenet")

def predict_image(image):
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    results = decode_predictions(predictions, top=5)[0]

    return [
        {
            "label": label,
            "confidence": round(float(score) * 100, 2)
        }
        for _, label, score in results
    ]
    