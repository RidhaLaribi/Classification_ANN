from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Chargement du modèle CNN
model = load_model("dog_cat_cnn_improved.h5")
IMG_SIZE = (128, 128)  # Même taille que l'entraînement

def browse_image():
    global img_label, result_label, confidence_label

    img_path = filedialog.askopenfilename(
        title="Select a Picture",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not img_path:
        return

    # Chargement et affichage de l'image
    img = Image.open(img_path)
    display_img = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(display_img)

    img_label.config(image=tk_img)
    img_label.image = tk_img

    # Prédiction avec le modèle CNN
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation

    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    if confidence > 0.5:
        result_label.config(text="🐶 DOG", fg="green")
        confidence_label.config(text=f"Confiance: {confidence:.2%}")
    elif confidence < 0.5:
        result_label.config(text="🐱 CAT", fg="blue")
        confidence_label.config(text=f"Confiance: {(1-confidence):.2%}")


# Interface utilisateur améliorée
root = Tk()
root.title("Dog vs Cat Classifier - CNN")
root.geometry("400x500")
root.resizable(False, False)

# Style
header = Label(root, text="Cat vs Dog Detector - CNN",
               font=("Arial", 18, "bold"), pady=10)
header.pack()

btn = Button(root, text="Select Image", font=("Arial", 12),
             command=browse_image, bg="#4CAF50", fg="white", height=2)
btn.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 20, "bold"))
result_label.pack(pady=5)

confidence_label = Label(root, text="", font=("Arial", 14))
confidence_label.pack(pady=5)

root.mainloop()