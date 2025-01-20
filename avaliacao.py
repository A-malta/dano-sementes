import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix


def load_test_data(image_dir, csv_path):
    dataset = pd.read_csv(csv_path)
    images = []
    labels = []
    for _, row in dataset.iterrows():
        img_path = os.path.join(image_dir, row['image'])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(0 if row['rotulo'] == 'n' else 1)
    return np.array(images), np.array(labels)

def main(model_path):
    model = tf.keras.models.load_model(model_path)
    image_dir = 'dataset'  
    csv_path = 'rotulo.csv'  
    x_test, y_test = load_test_data(image_dir, csv_path)

    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    print("Relatório de Classificação:")
    print(classification_report(y_test, predicted_classes, target_names=['Sem dano', 'Com dano']))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, predicted_classes))