from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model("dog_cat_rna_basic.h5")

img_path = "img_1.png"
img = image.load_img(img_path, target_size=(64, 64))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("dog")
else:
    print("cat")
