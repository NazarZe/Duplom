# Image Classification Pipeline with CNN Feature Extractors and Classical ML Models

Цей проєкт реалізує повний пайплайн для класифікації зображень. Основна ідея — використання попередньо натренованих convolutional neural networks (CNN) як екстракторів ознак, а також моделей машинного навчання для класифікації.

## Використані технології

### Бібліотеки для глибокого навчання:

* TensorFlow / Keras — для використання моделей ResNet50, MobileNetV2, EfficientNetB0
* Вбудовані функції preprocess\_input для передобробки зображень

### Класичне машинне навчання (scikit-learn):

* SVC (Support Vector Classifier)
* MLPClassifier (нейронна мережа)
* RandomForestClassifier
* StandardScaler — для нормалізації ознак
* classification\_report, accuracy\_score

### Інші інструменти:

* NumPy, Pandas — для роботи з масивами та таблицями
* Matplotlib, Seaborn — для побудови графіків
* glob, os, pickle, logging — для обробки даних, збереження моделей, логування
* SciPy — для обробки `.mat` файлів (Stanford features)

## Що робить проєкт

1. Завантажує зображення з каталогів
2. Витягує ознаки з використанням CNN
3. Тренує класичні ML-моделі
4. Оцінює точність та час роботи
5. Підтримує `.mat` ознаки зі Stanford
6. Зберігає моделі у `.pkl`
7. Створює графіки та CSV-результати
8. Може класифікувати окремі зображення

## Як користуватись

1. Встановити залежності:



2. Створити структуру каталогів:

```
images/
├── class0/
│   ├── img1.jpg
│   └── ...
├── class1/
│   └── ...
...
```

3. Запустити скрипт:

```bash
python main.py
```

4. Після запуску будуть створені:

* `pipeline_log.txt`
* `*.pkl` файли моделей та скейлерів
* `model_comparison_results.csv`
* `accuracy_comparison.png`, `train_time_comparison.png`, `feature_time_comparison.png`

5. Класифікувати зображення:

```python
predict_all_models("MyDog.jpg", class_names)
```
