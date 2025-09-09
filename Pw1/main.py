import random
import math
import matplotlib.pyplot as plt

# Сигмоїдальна функція активації
def sigmoid(x):
    return 10 / (1 + math.exp(-x))

# Похідна сигмоїдальної функції 
def der_sigmoid(x):
    return (math.exp(-x) / (1 + math.exp(-x)) ** 2) * 10

# Тренування
def train_neuron(data, learning_rate=0.01, max_epochs=10_000, error_threshold=0.0001):
    # Випадкова генерація синаптичних ваг
    weights = [random.uniform(-1, 1) for _ in range(3)]
    
    # Кількість навчальних прикладів
    n = len(data) - 3
    previous_total_error = float('inf')
    errors = []
    # цикл тренування
    for epoch in range(max_epochs):

        total_error = 0
        deltas = [0, 0, 0]

        for i in range(n):
            
            x = [data[i], data[i+1], data[i+2]]

            y = data[i+3]
            # Зважувальна сума(вхід нейрона)
            w_sum = sum(x[j] * weights[j] for j in range(3))

            # Прогнозоване значенння(сигмоїдальна функція активації)
            y_pred = sigmoid(w_sum)

            # Квадратична помилка
            error = (y_pred - y) ** 2
            total_error += error

            # Похідна помилки для зворотного поширення (градієнт)
            delta = (y_pred - y) * der_sigmoid(w_sum)

            # Дельти для кожної ваги
            for j in range(3):
                deltas[j] += -learning_rate * delta * x[j]

        # Оновлені ваги
        for k in range(3):
            weights[k] += deltas[k] / n

        errors.append(total_error)

        # Якщо зміна помилки менше 0.01 * кікість навчальних наборів
        if abs(previous_total_error - total_error) < error_threshold * n:
            break

        previous_total_error = total_error

    plt.plot(range(0, len(errors)), errors, label='Сумарна помилка')
    plt.xlabel('Кількість епох')
    plt.ylabel('Помилка')
    plt.title('Динаміка помилки під час тренування')
    plt.legend()
    plt.show()
    # Навчені вагові коефіцієнти
    return weights

# Функція прогнозу
def predict_next(data, weights):
    x1, x2, x3 = data[-3], data[-2], data[-1]
    s = x1 * weights[0] + x2 * weights[1] + x3 * weights[2]
    return sigmoid(s)

# 1 - 13 дані для тренування, 14 і 15 - тестові
data = [0.58, 3.38, 0.91, 5.80, 0.91, 5.01, 1.17, 4.67, 0.60, 4.81, 0.53, 4.75, 1.01, 5.04, 1.07]

# Тренування нейрону, поверне ваги
weights = train_neuron(data[:13])

# Прогнозуєм
predicted_x14 = predict_next(data[:13], weights)
predicted_x15 = predict_next(data[1:14], weights)

print(f"Прогнозуємо x14: {predicted_x14}")
print(f"Дійсне x14: {data[13]}")
print(f"Прогнозуємо x15: {predicted_x15}")
print(f"Дійсне x15: {data[14]}")

# Обчислюємо похибку
print(f"Похибка на тестових даних: {round((abs(predicted_x14 - data[13])/data[13] + abs(predicted_x15 - data[14])/data[14])*50, 2)}%")
