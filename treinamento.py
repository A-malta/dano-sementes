import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

input_shape = (128, 128, 1)  
num_classes = 2 

def create_model():

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data_from_csv(image_dir, csv_path):
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

def main():
    image_dir = 'dataset'  
    csv_path = 'rotulo.csv'  
    images, labels = load_data_from_csv(image_dir, csv_path)

    data_gen = ImageDataGenerator(
        rotation_range=100,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    for train_index, val_index in kf.split(images):
        print(f'Treinando no fold {fold}...')
        x_train, x_val = images[train_index], images[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = create_model()

        train_generator = data_gen.flow(x_train, y_train, batch_size=32)

        history = model.fit(
            train_generator,
            epochs=100,
            validation_data=(x_val, y_val)
        )

        print(f'Fold {fold} concluído.')
        fold += 1

    model.save('damage_classification_model.h5')