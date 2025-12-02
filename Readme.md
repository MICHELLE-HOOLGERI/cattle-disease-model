# 🐄 Cattle Disease Detection Model

This repository contains the trained deep learning model for classifying cattle into:

- **Foot-and-Mouth Disease**
- **Lumpy Skin Disease**
- **Healthy**

The model is trained using **MobileNetV2** (Transfer Learning) on a custom dataset of cattle images.

---

## 📁 Repository Structure

```
model/
    cattle_disease_model.h5      → Trained MobileNetV2 model file
    label_encoder.pkl            → Label encoder for mapping class names
    class_names.txt              → List of class names used during training

notebooks/
    training.ipynb               → Jupyter/Colab notebook for training the model

README.md                        → Project documentation
```

---

## 🧠 Model Information

- **Architecture:** MobileNetV2  
- **Input Size:** 224 × 224  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Output Classes:** 3  
- **Dataset Type:** Folder-based (images in class-named directories)  

---

## 🚀 How to Load the Model

```python
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("model/cattle_disease_model.h5")

# Load label encoder
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
```

---

## 🖼 Prediction Example

```python
import cv2
import numpy as np

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    class_id = np.argmax(preds)
    label = label_encoder.inverse_transform([class_id])[0]

    return label
```

