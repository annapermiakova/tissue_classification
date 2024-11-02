

# Решение для классификации изображений тканей

## Описание проекта
Это решение использует модель на основе `EfficientNetB0` для классификации изображений тканей на опухолевые и нормальные. Решение обучалось на изображениях, и итоговая модель достигает высокой точности и F1 Score на тестовой выборке.

## Структура репозитория
- `train_model.py` - Код для обучения модели.
- `predict_model.py` - Код для предсказаний и сохранения их в CSV.
- `model/efficientnetb0_model.h5` - Сохраненные веса обученной модели.
- `requirements.txt` - Список используемых библиотек.

## Запуск решения

### 1. Установка зависимостей
- `pip install -r requirements.tx`

2. Обучение модели
Для запуска обучения выполните:
python train_model.py

4. Генерация предсказаний
Чтобы получить предсказания на новых данных и сохранить их в CSV:
python predict_model.py --data_path /path/to/test_data

Использованные данные
Модель обучена на изображениях тканей, разделённых на опухолевые и нормальные.

### Шаг 3: Сохраните файл




