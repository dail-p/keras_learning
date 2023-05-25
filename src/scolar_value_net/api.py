import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import (
    Axes,
)

from scolar_value_net.calc_methods import (
    ScolarValueNetHandler,
    PVINetHandler,
QNetHandler,
FNetHandler
)


class ScolarValueFacade:
    """

    """
    calc_method = ScolarValueNetHandler

    data_file_name = 'KIN_98.csv'

    metric_title = 'PVI, Q, F'
    metric_name = 'PVI_Q_F'

    def __init__(self):
        self.x_data, self.y_data = self._get_data()
        self._calc()

    @classmethod
    def load_data(cls):
        """

        Returns:

        """

        def get_curvature_value(i):
            return 0.5 + 0.0625 * i

        def get_length_value_list(length_index_list):
            value_by_index = {
                1: 2.23,
                2: 2.45,
                4: 3.74,
                5: 4.58,
                6: 5.53,
                7: 6.68,
            }

            return list(map(lambda x: value_by_index.get(x, x), length_index_list))

        data_path = os.path.join('files')

        tree = os.walk(data_path)

        result = None

        for current_path, folders, files in tree:
            if cls.data_file_name in files:
                data = np.loadtxt(
                    os.path.join(current_path, cls.data_file_name),
                    delimiter='\t',
                )

                data[:, 0] = get_length_value_list(data[:, 0])

                curvature_index = int(current_path[-1])
                curvature_value = get_curvature_value(curvature_index)

                data = np.insert(data, 0, 10 * [curvature_value], axis=1)

                if result is None:
                    result = data
                else:
                    result = np.vstack([result, data])

        return result

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        Args:
            data: Матрица дынных

        Returns: Векторы входных и выходных данных
        """
        np.random.shuffle(data)
        return data[:, :2], data[:, 2:]

    def _get_data(self):
        """

        """
        data = self.load_data()

        return self._prepare_data(data)

    def _calc(self):
        """

        """
        self.handler = self.calc_method(self.x_data, self.y_data)
        self.handler.run()
        self.handler.get_metrics_values()
        self.print_metrics()
        #self.print_predict()

    def _get_value(self, key):
        """

        Args:
            key:

        Returns:

        """
        return self.handler.model.predict(key)

    def _get_weights(self):
        """

        Returns:

        """
        return self.handler.model.get_weights()

    def print_predict(self):
        """
        Рисует значения
        """
        def build_figure(index, label):
            """

            Args:
                index:
                label

            Returns:

            """
            fig, axe = plt.subplots()
            axe.plot(y, predict_data[:, 10, index],)
            axe.scatter(self.x_data[:, 1], self.y_data[:, index])
            axe.set_xlabel('l')
            axe.set_ylabel(label)
            axe.set_title('a)')

            fig.savefig(os.path.join('scolar_value_net', 'figure', f'l_predict_{index}'))

            fig = plt.figure()
            plot3d_axe = fig.add_subplot(projection='3d')
            plot3d_axe.plot_surface(
                xgrid,
                ygrid,
                predict_data[:, :, index],
                cmap='inferno'
            )
            plot3d_axe.set_xlabel('l')
            plot3d_axe.set_ylabel('a')
            plot3d_axe.set_zlabel(label)
            plot3d_axe.set_title('b)')

            fig.savefig(os.path.join('scolar_value_net', 'figure', f'predict3d_{index}'))

        x = np.arange(min(self.x_data[:, 0]), max(self.x_data[:, 0]), 0.01)
        y = np.arange(min(self.x_data[:, 1]), max(self.x_data[:, 1]), 0.15)
        xgrid, ygrid = np.meshgrid(x, y)

        predict_data = np.zeros(xgrid.shape + (3,))
        for i, (x_item, y_item) in enumerate(zip(xgrid, ygrid)):
            predict_data[i] = self._get_value(np.column_stack([x_item, y_item]))

        build_figure(0, 'PVI')
        build_figure(1, 'q')
        build_figure(2, 'F')

    def print_metrics(self):
        """
        Выводит графики метрик после обучения
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        axe_mse, axe_mae, axe_mape, axe_msle = axes.flatten()

        _data = self.handler.history.history
        data_mse, data_mae, data_mape, data_msle = _data['mse'], _data['mae'], _data['mape'], _data['msle']

        self.setup_metric_axe(axe_mse, data_mse, 'mse')
        self.setup_metric_axe(axe_mae, data_mae, 'mae')
        self.setup_metric_axe(axe_mape, data_mape, 'mape')
        self.setup_metric_axe(axe_msle, data_msle, 'msle')
        fig.suptitle(self.metric_title, fontsize=16)

        fig.savefig(os.path.join(
            'scolar_value_net',
            'figure',
            f'metrics_{self.metric_name}'
        ))
        plt.show()

    def setup_metric_axe(self, axe: Axes, data, axe_name):
        """
        Задает настройки графику метрики.

        Args:
            axe:
            data:
            axe_name:

        Returns:

        """
        axe.plot(data)
        axe.set_ylim(0, data[10])
        axe.grid(True)
        axe.set_title(axe_name)


class PVINetFacade(ScolarValueFacade):
    """

    """
    calc_method = PVINetHandler

    metric_title = 'PVI'
    metric_name = 'PVI'

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = super()._prepare_data(data)

        return x, y[:, 0]


class QNetFacade(ScolarValueFacade):
    """

    """
    calc_method = QNetHandler

    metric_title = 'Q'
    metric_name = 'Q'

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = super()._prepare_data(data)

        return x, y[:, 1]


class FNetFacade(ScolarValueFacade):
    """

    """
    calc_method = FNetHandler

    metric_title = 'F'
    metric_name = 'F'

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = super()._prepare_data(data)

        return x, y[:, 2]
