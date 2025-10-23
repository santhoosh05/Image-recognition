# Image-recognition
identifies basic objects  in the given picture (multiple objects in a single image )
# Install dependencies if you don't have them
# pip install tensorflow pillow

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# 1️⃣ Load a pretrained ResNet50 model
model = ResNet50(weights='imagenet')

# 2️⃣ Path to your image
img_path = 'example.jpg'  # replace with your image file

# 3️⃣ Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 4️⃣ Run prediction
predictions = model.predict(x)

# 5️⃣ Decode and show top 3 predicted labels
decoded = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded):
    print(f"{i+1}. {label} ({score:.2f})")

