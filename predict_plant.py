import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("best_model.h5")

# Image parameters
IMG_SIZE = (224, 224)

# Load class labels (same order as during training)
# You must match this with the training generator's class_indices
class_labels = sorted(os.listdir("dataset/Train_Set_Folder"))

def predict_species(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    predicted_class = class_labels[class_idx]
    print(f"\n🌿 Predicted Species: {predicted_class} ({confidence:.2%} confidence)")

# Example usage
if __name__ == "__main__":
    img_path = input("Enter path to plant image: ")
    predict_species(img_path)
