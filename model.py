"""
Модуль с математической моделью колебательного
контура и её внутренним состоянием
"""


import numpy as np


class LCOscillatorModel:
    """
    Класс математической модели LC-контура
    """
    def __init__(
            self,
            inductance: float,
            capacitance: float,
            initial_current: float,
            initial_voltage: float
    ):
        """
        Конструктор класса LCOscillatorModel, инициализирующий
        инстанс класса колебательного контура

        :param inductance: индуктивность катушки в Гн
        :param capacitance: ёмкость конденсатора в Ф
        :param initial_current: ток в контуре в начальный момент времени
        :param initial_voltage: заряд конденсатора в начальный момент времени
        """
        self.inductance = inductance
        self.capacitance = capacitance
        self.initial_current = initial_current
        self.initial_voltage = initial_voltage

        self.initial_state = np.array([
            self.initial_current,
            self.initial_voltage
        ])

        self.state = np.array([self.initial_state, ])
        self.solving_time = False

    def add_new_state(self, new_state_y: np.array):
        """
        Метод добавления значений нового состояния системы
        в параметр state, хранящий результаты моделирования

        :param new_state_y: значения состояния системы в новый момент времени
        :return: None
        """
        new_state = np.array([[
            new_state_y[0],
            -self.inductance * new_state_y[1]
        ]])
        self.state = np.append(self.state, new_state, axis=0)

    def get_last_state(self) -> np.array:
        """
        Метод, возвращающий текущее значение состояния системы

        :return: текущее состояние системы
        """
        return self.state[-1]

    def get_last_state_y(self) -> np.array:
        """
        Метод, возвращающий текущее значение состояния
        системы в координатах y

        :return: текущее состояние системы
        """
        last_state_y = np.array([
            self.state[-1, 0],
            -self.state[-1, 1] / self.inductance
        ])
        return last_state_y

    def last_equation_y(self, ) -> float:
        """
        Метод, возвращающий текущее значение
        последней функции в системе уравнений задачи Коши

        :return: текущее состояние системы последней функции
        """
        last_state = self.get_last_state_y()
        return -last_state[0] / (self.inductance * self.capacitance)


if __name__ == "__main__":
    m = LCOscillatorModel(
        inductance=1,
        capacitance=1,
        initial_current=0,
        initial_voltage=1
    )
    print(m.state)
