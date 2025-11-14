from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# Load model
model = load_model("dog_cat_rna_basic.h5")

IMG_SIZE = (64, 64)

def browse_image():
    global img_label, result_label

    img_path = filedialog.askopenfilename(
        title="Select a Picture",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not img_path:
        return

    # Load + show image
    img = Image.open(img_path)
    display_img = img.resize((250, 250))   # display size
    tk_img = ImageTk.PhotoImage(display_img)

    img_label.config(image=tk_img)
    img_label.image = tk_img

    # ---- Your original prediction logic ----
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result_label.config(text=f"🐶 DOG  ({prediction[0][0]:.2f})", fg="green")
    else:
        result_label.config(text=f"🐱 CAT  ({prediction[0][0]:.2f})", fg="blue")


# ---- UI ----
root = Tk()
root.title("Dog vs Cat Classifier")
root.geometry("350x420")
root.resizable(False, False)

header = Label(root, text="Cat vs Dog Detector", font=("Arial", 18, "bold"))
header.pack(pady=10)

btn = Button(root, text="Select Image", font=("Arial", 12), command=browse_image)
btn.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
