import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_amd(image_path):
    model = load_model("models/model.h5")

    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=1)  # ➕ becomes (1, 1, 128, 128, 3)

    preds = model.predict(img_array)
    class_names = ['Dry AMD', 'Healthy', 'Wet AMD']
    predicted_class = np.argmax(preds)
    confidence = float(np.max(preds)) * 100

    return f"{class_names[predicted_class]} ({confidence:.2f}%)"
