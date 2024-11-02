import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Параметры
img_size = (224, 224)
batch_size = 16
img_shape = (img_size[0], img_size[1], 3)
data_dir_cancer = 'data/colon_aca'  # Путь к изображениям с опухолью
data_dir_normal = 'data/colon_n'    # Путь к изображениям с нормой

# Создание DataFrame с метками
train_paths = []
train_labels = []

for img_file in os.listdir(data_dir_cancer):
    train_paths.append(os.path.join(data_dir_cancer, img_file))
    train_labels.append(1)

for img_file in os.listdir(data_dir_normal):
    train_paths.append(os.path.join(data_dir_normal, img_file))
    train_labels.append(0)

train_df = pd.DataFrame({'filepaths': train_paths, 'labels': train_labels})
train_df, test_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=42)

# Генераторы данных
tr_gen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_gen = tr_gen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='raw',
    shuffle=True, batch_size=batch_size, subset='training'
)
valid_gen = tr_gen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='raw',
    shuffle=True, batch_size=batch_size, subset='validation'
)

# Модель EfficientNetB0
input_layer = Input(shape=img_shape)
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=input_layer, pooling='max')
x = BatchNormalization()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.45)(x)
output_layer = Dense(2, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)

# Компиляция и обучение
model.compile(optimizer=Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=10, validation_data=valid_gen)

# Сохранение модели
model.save('model/efficientnetb0_model.h5')
print("Модель сохранена в папке model")
