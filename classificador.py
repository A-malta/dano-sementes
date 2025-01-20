import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Label, Style, Frame
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def classify_image(model, img_label, result_label):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return

    try:
        img = Image.open(file_path)


        max_size = 500
        original_width, original_height = img.size
        if original_width > max_size or original_height > max_size:
            if original_width > original_height:
                new_width = max_size
                new_height = int((max_size / original_width) * original_height)
            else:
                new_height = max_size
                new_width = int((max_size / original_height) * original_width)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)

        img_tk = ImageTk.PhotoImage(img)

        img_label.config(image=img_tk)
        img_label.image = img_tk

        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        prediction_confidence = prediction[0][predicted_class]

        result_text = f"Com dano" if predicted_class == 1 else f"Sem dano"
        result_label.config(text=result_text, foreground="green" if predicted_class == 1 else "blue")
    except Exception as e:
        result_label.config(text=f"Erro ao processar: {e}", foreground="red")


def main(model_path):
    model = tf.keras.models.load_model(model_path)

    root = tk.Tk()
    root.title("Classificador de Imagens - Dano ou Não")
    root.geometry("800x900")
    root.configure(bg="#f7f7f7")

    style = Style()
    style.configure("TFrame", background="#f7f7f7")
    style.configure("TLabel", background="#f7f7f7", foreground="#333333", font=("Arial", 12))

    frame = Frame(root)
    frame.pack(pady=20)

    title_label = Label(frame, text="Classificador de Imagens", font=("Arial", 20, "bold"), foreground="#333333")
    title_label.pack(pady=10)

    desc_label = Label(frame, text="Carregue uma imagem para classificar como 'Com dano' ou 'Sem dano'.", font=("Arial", 14), foreground="#555555")
    desc_label.pack(pady=10)

    img_label = Label(frame)
    img_label.pack(pady=20)

    result_label = Label(frame, text="", font=("Arial", 16, "bold"), foreground="#333333")
    result_label.pack(pady=20)

    select_button = tk.Button(
        frame,
        text="Selecionar Imagem",
        command=lambda: classify_image(model, img_label, result_label),
        font=("Arial", 14),
        bg="#4CAF50",
        fg="white",
        bd=0,
        relief="flat"
    )
    select_button.pack(pady=20)

    root.mainloop()