import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import (
    Axes,
)

from halpers import (
    load_data,
    prepare_data,
)
from scolar_value_net.calc_metods import (
    ScolarValueNetHandler,
)


class ScolarValueFacade:
    """

    """
    def __init__(self):
        self.x_data, self.y_data = self._get_data()
        self._calc()

    def _get_data(self):
        """

        """
        data = load_data()

        return prepare_data(data)

    def _calc(self):
        """

        """
        self.handler = ScolarValueNetHandler(self.x_data, self.y_data)
        self.handler.run()
        self.handler.get_metrics_values()
        self.print_metrics()
        self.print_predict()

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
        def build_fiure(index, label):
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

        build_fiure(0, 'PVI')
        build_fiure(1, 'q')
        build_fiure(2, 'F')

    def print_metrics(self):
        """
        Выводит графики метрик после обучения
        """
        def setup_axe(axe: Axes, data, axe_name, ylim_index=5):
            """
            Задает настройки графику метрики.

            Args:
                axe:
                data:
                axe_name:
                ylim_index:

            Returns:

            """
            axe.plot(data)
            axe.set_ylim(0, data[10])
            axe.grid(True)
            axe.set_title(axe_name)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        axe_mse, axe_mae, axe_mape, axe_msle = axes.flatten()

        _data = self.handler.history.history
        data_mse, data_mae, data_mape, data_msle = _data['mse'], _data['mae'], _data['mape'], _data['msle']

        setup_axe(axe_mse, data_mse, 'mse')
        setup_axe(axe_mae, data_mae, 'mae')
        setup_axe(axe_mape, data_mape, 'mape', ylim_index=25)
        setup_axe(axe_msle, data_msle, 'msle')
        fig.tight_layout()

        fig.savefig(os.path.join('scolar_value_net', 'figure', 'metrics'))
        plt.show()

