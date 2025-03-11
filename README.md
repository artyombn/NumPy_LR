# Логистический регрессор

### Описание 
Логистический регрессор, который классифицирует отзывы из Amazon методом градиентного спуска.

### Установка 
Проект использует Poetry для управления зависимостями: 

```sh
poetry install
```

### Использование 

##### Пример генерации данных:  
```python
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
```

##### Обучение модели:  
```python
model = LogisticRegression()
model.train(X_train, y_train, learning_rate=0.01, num_iters=1000)
```

##### Предсказание:  
```python
prediction = model.predict(X_train)
```

##### Пример использования:  
`/predict_example.py`  
`/model_learning.py`
`/homework.ipynb`


##### _Для корректной работы `plt.show()` из библиотеки `matplotlib`, запускайте код через консоль `python predict.py`_