"""
Модуль с решателями для задачи моделирования LC-контура
"""


import numpy as np
from model import LCOscillatorModel
from abc import ABC, abstractmethod
from time import time


class SolverInterface(ABC):
    """
    Абстрактный метод решателя
    """
    @abstractmethod
    def __init__(
            self,
            step_size: float,
            simulation_time: float,
            model: LCOscillatorModel
    ):
        """
        Конструктор класса решателя

        :param step_size: шаг интегрирования
        :param simulation_time: время интегрирования
        :param model: класс модели
        """
        self.step_size = step_size
        self.simulation_time = simulation_time
        self.model = model
        self.time_steps = np.arange(0, self.simulation_time, self.step_size)

    @abstractmethod
    def solve(self):
        """
        Абстрактный метод интегрирования модели LC-контура

        :return:
        """
        pass


class SimpleEulerSolver(SolverInterface):
    """
    Решатель методом Эйлера
    """
    def __init__(
            self,
            step_size: float,
            simulation_time: float,
            model: LCOscillatorModel
    ):
        """
        Конструктор класса решателя методом Эйлера

        :param step_size: шаг интегрирования
        :param simulation_time: время интегрирования
        :param model: модель
        """
        super().__init__(
            step_size,
            simulation_time,
            model
        )

    def solve(self):
        """
        Метод интегрирования модели LC-контура
        решателя методом Эйлера.

        Решает систему уравнений и обновляет внутренний параметр
        модели state, добавляя новые значения
        на каждом шаге интегрирования.

        В конце расчета обновляет
        затраченное на него время в параметре модели.

        :return: None
        """
        start_time = time()

        for _ in self.time_steps:
            last_state = self.model.get_last_state_y()

            new_state = np.array([
                last_state[0] + last_state[1] * self.step_size,
                last_state[1] + self.model.last_equation_y() * self.step_size
            ])
            self.model.add_new_state(new_state)

        end_time = time()
        solving_time = end_time - start_time
        self.model.solving_time = solving_time


class BackwardEulerSolver(SolverInterface):
    """
    Решатель неявным методом Эйлера
    """
    def __init__(
            self,
            step_size: float,
            simulation_time: float,
            model: LCOscillatorModel
    ):
        """
        Конструктор класса решателя неявным методом Эйлера

        :param step_size: шаг интегрирования
        :param simulation_time: время интегрирования
        :param model: модель
        """
        super().__init__(
            step_size,
            simulation_time,
            model
        )

    def solve(self):
        """
        Метод интегрирования модели LC-контура
        решателя неявным методом Эйлера.

        Решает систему уравнений и обновляет внутренний параметр
        модели state, добавляя новые значения
        на каждом шаге интегрирования.

        В конце расчета обновляет
        затраченное на него время в параметре модели.

        :return: None
        """
        def y_2_new_func(y_1_old: float, y_2_old: float) -> float:
            """
            Функция расчета значения параметра y_2
            на очередном шаге интегрирования

            :param y_1_old: значение параметра y_1 на предыдущем шаге
            :param y_2_old: значение параметра y_2 на предыдущем шаге
            :return: значения параметра y_2 на очередном шаге интегрирования
            """
            a = 1 / (
                    1 + (self.step_size ** 2) /
                    (self.model.inductance * self.model.capacitance)
            )
            b = self.step_size / (
                    self.model.inductance * self.model.capacitance
            )
            return a * y_2_old - b * y_1_old

        start_time = time()

        for _ in self.time_steps:
            last_state = self.model.get_last_state_y()
            y_2_new = y_2_new_func(last_state[0], last_state[1])
            y_1_new = last_state[0] + y_2_new * self.step_size

            new_state = np.array([
                y_1_new,
                y_2_new
            ])
            self.model.add_new_state(new_state)

        end_time = time()
        solving_time = end_time - start_time
        self.model.solving_time = solving_time


class SimpleRungeKutta(SolverInterface):
    """
    Решатель методом Рунге-Кутта
    """
    def __init__(
            self,
            step_size: float,
            simulation_time: float,
            model: LCOscillatorModel
    ):
        """
        Конструктор класса решателя методом Рунге-Кутта

        :param step_size: шаг интегрирования
        :param simulation_time: время интегрирования
        :param model: модель
        """
        super().__init__(
            step_size,
            simulation_time,
            model
        )

    def solve(self):
        """
        Метод интегрирования модели LC-контура
        решателя неявным методом Рунге-Кутта.

        Решает систему уравнений и обновляет внутренний параметр
        модели state, добавляя новые значения
        на каждом шаге интегрирования.

        В конце расчета обновляет
        затраченное на него время в параметре модели.

        :return: None
        """
        def y_new(y_old: float, y_func: float) -> float:
            """
            Функция расчета значения параметров
            на очередном шаге интегрирования

            :param y_old: значение параметра на предыдущем шаге
            :param y_func: значение параметра на предыдущем шаге
            :return: состояние параметра на очередном шаге интегрирования
            """
            k_1 = y_func
            k_2 = y_func + self.step_size * k_1 / 2
            k_3 = y_func + self.step_size * k_2 / 2
            k_4 = y_func + self.step_size * k_3

            y_delta = self.step_size * (
                k_1 + 2 * k_2 + 2 * k_3 + k_4
            ) / 6
            return y_old + y_delta

        start_time = time()

        for _ in self.time_steps:
            last_state = self.model.get_last_state_y()
            new_state = np.array([
                y_new(last_state[0], last_state[1]),
                y_new(
                    last_state[1],
                    - last_state[0] / (
                            self.model.inductance *
                            self.model.capacitance
                    )
                )
            ])
            self.model.add_new_state(new_state)

        end_time = time()
        solving_time = end_time - start_time
        self.model.solving_time = solving_time


if __name__ == "__main__":
    from model import LCOscillatorModel

    m = LCOscillatorModel(
        inductance=1,
        capacitance=1,
        initial_current=0,
        initial_voltage=1
    )
    s = SimpleEulerSolver(
        step_size=0.001,
        simulation_time=10,
        model=m
    )
    s.solve()
    print(m.solving_time)
