import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Путь к сохраненной модели
model_path = 'model/efficientnetb0_model.h5'

# Аргументы командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help="Путь к папке с изображениями для предсказаний")
args = parser.parse_args()

# Генератор данных
data_gen = ImageDataGenerator(rescale=1./255)
test_gen = data_gen.flow_from_directory(
    args.data_path,
    target_size=(224, 224),
    class_mode=None,
    shuffle=False
)

# Загрузка модели и предсказания
model = load_model(model_path)
predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)

# Сохранение предсказаний в CSV
filenames = test_gen.filenames
results = pd.DataFrame({"Filename": filenames, "Prediction": predicted_classes})
results.to_csv("predictions.csv", index=False)
print("Предсказания сохранены в 'predictions.csv'")
