import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

# --------------------------
# SETTINGS
# --------------------------
MODEL_PATH = "character_cnn_model.h5"
IMAGE_SIZE = (64, 64)  # same as training
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A_caps','B_caps','C_caps','D_caps','E_caps','F_caps','G_caps','H_caps','I_caps','J_caps','K_caps','L_caps','M_caps','N_caps','O_caps','P_caps','Q_caps','R_caps','S_caps','T_caps','U_caps','V_caps','W_caps','X_caps','Y_caps','Z_caps',
               'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# --------------------------
# LOAD MODEL
# --------------------------
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --------------------------
# IMAGE PREPROCESSING FUNCTION
# --------------------------
def preprocess_image(img_path):
    if not os.path.exists(img_path):
        print("Error: Image file does not exist.")
        sys.exit(1)

    img = image.load_img(img_path, color_mode='grayscale', target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# --------------------------
# PREDICTION
# --------------------------
def predict_character(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    print(f"Predicted Character: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    predict_character(img_path)
