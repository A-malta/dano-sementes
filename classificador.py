import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_path = 'damage_classification_model.h5'
model = tf.keras.models.load_model(model_path)
img_label = None
result_label = None

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classify_image():
    global img_label, result_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    result_text = "Com dano" if predicted_class == 1 else "Sem dano"
    result_label.config(text=f"Resultado: {result_text}", fg="green" if predicted_class == 1 else "blue")

def main():
    global img_label, result_label
    root = tk.Tk()
    root.title("Classificador de Imagens - Dano ou Não")
    
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    select_button = Button(frame, text="Selecionar Imagem", command=classify_image, font=("Arial", 14))
    select_button.pack()
    
    img_label = Label(frame)
    img_label.pack(pady=10)
    
    result_label = Label(frame, text="", font=("Arial", 16))
    result_label.pack(pady=10)
    
    root.mainloop()