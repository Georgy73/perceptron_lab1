import random
import pandas
import math
import numpy


class Perceptron:
    def __init__(self, input_size, learn_speed):
        self.inputs = None
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.learn_speed = learn_speed
        self.epochs = 1000

    def activation(self):  # тут должна быть сигмоида
        total = 0
        for i, j in zip(self.inputs, self.weights):
            total += (i * j)
        if total > 0:
            return 1.0
        else:
            return 0.0

    def train_gradient(self, train_x, train_y):
        length = len(train_x)
        iterations = 0
        while iterations <= self.epochs:
            global_error = 0
            for i in (range(length)):
                self.inputs = train_x[i]
                out = self.activation()

                error = ((train_y[i] - out) ** 2) / 2  # E = 1/2(d-y)^2
                global_error += abs(error / length)

                out_derivative = out * (1 - out)  # производная выхода
                grad = out_derivative * error
                for it in range(len(self.inputs)):
                    self.weights[it] += self.learn_speed * grad * self.inputs[it]

            iterations += 1

    def train(self, train_x, train_y):
        length = len(train_x)
        iterations = 0
        while True:
            global_error = 0
            for i in (range(length)):
                self.inputs = train_x[i]
                out = self.activation()
                error = train_y[i] - out
                update = self.learn_speed * error

                for it in range(len(self.inputs)):
                    self.weights[it] += update * self.inputs[it]

                if error != 0:
                    global_error += 1
                # if out == train_y[i]:
                #     print("ss")
                # elif out != train_y[i] and train_y[i] == 0:
                #     for it in range(len(self.inputs)):
                #         if self.inputs[it] > 0:  # if input is active
                #             self.weights[it] += self.inputs[it]
                # elif out != train_y[i] and train_y[i] == 1:
                #     for it in range(len(self.inputs)):
                #         if self.inputs[it] > 0:  # if input is active
                #             self.weights[it] -= self.inputs[it]

                # error = ((train_y[i] - out) ** 2) / 2  # E = 1/2(d-y)^2
                # global_error += abs(error / length)
                #
                # out_derivative = out * (1 - out)  # производная выхода
                # grad = out_derivative * error

            if global_error == 0:
                break
            iterations += 1

    def predict(self, input_data):
        self.inputs = input_data
        return self.activation()


def accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


# tr_data_x = [  # OR
#     [0, 1],
#     [0, 0],
#     [1, 0],
#     [1, 1]
# ]
#
# tr_data_y = [1, 0, 1, 1]  # OR
#
# test_data_x = [
#     [0, 1],
#     [1, 1],
#     [1, 0],
#     [0, 0]
# ]
#
# true_data_y = [1, 1, 1, 0]

df2 = pandas.read_csv("../data.csv")

tr_data_x = df2[:16].iloc[:, :2].to_numpy()
tr_data_y = df2[:16]['class'].to_numpy()

test_data_x = df2[16:20].iloc[:, :2].to_numpy()
true_data_y = df2[16:20]['class'].to_numpy()

print('Test data', test_data_x, 'True result', true_data_y)

test_data_y = []

perc = Perceptron(input_size=2, learn_speed=0.1)
perc.train(tr_data_x, tr_data_y)

for x in test_data_x:
    result = perc.predict(x)
    test_data_y.append(result)

print('Result:', test_data_y)
cor = accuracy(true_data_y, test_data_y)
print('Accuracy:', cor)
