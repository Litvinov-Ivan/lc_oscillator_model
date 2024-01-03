"""
Модуль проведения симуляции и анализа модели
"""


from model import LCOscillatorModel
from solver import SimpleEulerSolver, BackwardEulerSolver, SimpleRungeKutta
import matplotlib.pyplot as plt
from typing import Union, Type


class Simulation:
    """
    Класс симуляции - проведения интегрирования задачи Коши
    """
    def __init__(
            self,
            inductance: float,
            capacitance: float,
            initial_current: float,
            initial_voltage: float,
            step_size: float,
            simulation_time: float,
            solver: Union[
                Type[SimpleEulerSolver],
                Type[BackwardEulerSolver],
                Type[SimpleRungeKutta]
            ]
    ):
        """
        Конструктор класса моделирования

        :param inductance: индуктивность катушки в Гн
        :param capacitance: ёмкость конденсатора в Ф
        :param initial_current: ток в контуре в начальный момент времени
        :param initial_voltage: заряд конденсатора в начальный момент времени
        :param step_size: шаг интегрирования
        :param simulation_time: время интегрирования
        :param solver: решатель
        """
        self.inductance = inductance
        self.capacitance = capacitance
        self.initial_current = initial_current
        self.initial_voltage = initial_voltage
        self.step_size = step_size
        self.simulation_time = simulation_time

        self.model = LCOscillatorModel(
            inductance=self.inductance,
            capacitance=self.capacitance,
            initial_current=self.initial_current,
            initial_voltage=self.initial_voltage
        )

        self.solver = solver(
            step_size=self.step_size,
            simulation_time=self.simulation_time,
            model=self.model
        )

    def start_simulation(self):
        """
        Метод проведения симуляции по введённым параметрам

        :return: None
        """
        self.solver.solve()

    def show_plots(self):
        """
        Метод вывода графиков внутреннего состояния
        системы I(t) и U(t) после интегрирования.

        :return: None
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            layout='constrained',
            sharex=True
        )
        ax1.plot(
            self.solver.time_steps,
            self.model.state[1:, 0]
        )
        ax1.set_title('Capacitor current')
        ax1.set_xlabel('time, sec')
        ax1.set_ylabel('I, A')
        ax1.grid()

        ax2.plot(
            self.solver.time_steps,
            self.model.state[1:, 1]
        )
        ax2.set_title('Capacitor voltage')
        ax2.set_xlabel('time, sec')
        ax2.set_ylabel('U, V')
        ax2.grid()

        fig.suptitle('LC circuit simulation', fontsize=16)
        plt.show()


if __name__ == "__main__":
    sim = Simulation(
        inductance=1,
        capacitance=1,
        initial_current=0,
        initial_voltage=1,
        step_size=0.001,
        simulation_time=100,
        solver=SimpleRungeKutta
    )

    sim.start_simulation()

    sim.show_plots()
