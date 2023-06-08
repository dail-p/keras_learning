import os
from collections import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import (
    Axes,
)

from scolar_value_net.calc_methods import (
    ENetHandler,
    PVINetHandler,
    QNetHandler,
    ScolarValueNetHandler,
)


def nearest_value_index(items, value):
    """Поиск ближайшего значения до value в списке items"""
    found = items[0]
    ind = 0
    for index, item in enumerate(items):
        if abs(item - value) < abs(found - value):
            found = item
            ind = index

    return ind


class ScolarValueFacade:
    """
    Фасад для работы с результатами обучения
    """
    # Класс, описывающий сеть
    calc_method = ScolarValueNetHandler
    # Наименование файла с данными
    data_file_name = 'KIN_98.csv'

    # Заголовок картинки с метриками
    metric_title = 'E, PVI, Q'
    # Наименование файла-картинки с метриками
    metric_name = 'E_PVI_Q'

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

                # Преобразуем интекс положения в l1
                data[:, 0] = get_length_value_list(data[:, 0])

                # Добавим столбец высоты изгиба в начало
                curvature_index = int(current_path[-1])
                curvature_value = get_curvature_value(curvature_index)
                data = np.insert(data, 0, 10 * [curvature_value], axis=1)

                # Заменим последний столбец на значения Q98 из файлов lzwrk
                q_list = [0] * 10
                for file in files:
                    if file[0].isdigit():
                        indx = int(file[0:-6]) - 1
                        q_data = np.loadtxt(
                            os.path.join(current_path, file),
                            delimiter='\t',
                        )

                        q_list[indx] = q_data[-1, 2]

                data[:, -1] = q_list

                if result is None:
                    result = data
                else:
                    result = np.vstack([result, data])

        return result

    def _normalize(self, data, prefix, init=False):
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                column_data = data[:, i]
                if init:
                    setattr(self, f'{prefix}_normalize_{i}', (column_data.min(), column_data.max()))
                    _min, _max = column_data.min(), column_data.max()
                else:
                    _min, _max = getattr(self, f'{prefix}_normalize_{i}')

                data[:, i] = (column_data - _min) / (_max - _min)
            result = data
        else:
            if init:
                setattr(self, f'{prefix}_normalize_{0}', (data.min(), data.max()))
                _min, _max = data.min(), data.max()
            else:
                _min, _max = getattr(self, f'{prefix}_normalize_{0}')

            result = (data - _min) / (_max - _min)

        return result

    def _denormalize(self, data, prefix):
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                column_data = data[:, i]
                _min, _max = getattr(self, f'{prefix}_normalize_{i}')
                data[:, i] = column_data * (_max - _min) + _min
            result = data
        else:
            _min, _max = getattr(self, f'{prefix}_normalize_{0}')
            result =  data * (_max - _min) + _min

        return result

    def _get_real_data(self):
        x = self.x_data.copy()
        y = self.y_data.copy()

        x = self._denormalize(x, 'x')
        y = self._denormalize(y, 'y')

        return x, y

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
        x, y = self._prepare_data(data)

        x = self._normalize(x, 'x', init=True)
        y = self._normalize(y, 'y', init=True)

        return x, y

    def _calc(self):
        """

        """
        self.handler = self.calc_method(self.x_data, self.y_data)
        self.handler.run()
        self.print_metrics()
        self.print_predict()

    def _get_value(self, key):
        """

        Args:
            key:

        Returns:

        """
        key = self._normalize(key, 'x')

        predict = self.handler.model.predict(key)

        predict = self._denormalize(predict, 'y')

        return predict

    def _get_weights(self):
        """

        Returns:

        """
        return self.handler.model.get_weights()

    def print_predict(self):
        """
        Рисует значения
        """
        x_data, y_data = self._get_real_data()

        x = np.arange(min(x_data[:, 0]), max(x_data[:, 0]), 0.01)
        y = np.arange(min(x_data[:, 1]), max(x_data[:, 1]), 0.15)
        xgrid, ygrid = np.meshgrid(x, y)

        if len(y_data.shape) > 1:
            param_count = y_data.shape[1]
        else:
            param_count = 1

        predict_data = np.zeros(xgrid.shape + (param_count,))
        for i, (x_item, y_item) in enumerate(zip(xgrid, ygrid)):
            predict_data[i] = self._get_value(np.column_stack([x_item, y_item]))

        self.build_predict_all_figures(predict_data)

    def build_predict_all_figures(self, predict_data):
        """
        Собирает все графики по прогнозам
        """
        self.build_predict_figures(predict_data, 0, 'E')
        self.build_predict_figures(predict_data, 1, 'PVI')
        self.build_predict_figures(predict_data, 2, 'q')

    def build_predict_figures(self, predict_data, index, label):
        """

        Args:
            predict_data:
            index:
            label

        Returns:

        """
        x_data, y_data = self._get_real_data()
        x = np.arange(min(x_data[:, 0]), max(x_data[:, 0]), 0.01)
        y = np.arange(min(x_data[:, 1]), max(x_data[:, 1]), 0.15)
        xgrid, ygrid = np.meshgrid(x, y)

        fig, axe = plt.subplots()
        for h, color in zip(sorted(set(x_data[:, 0])), ['c', 'b', 'g', 'r', 'k']):
            index_h = nearest_value_index(x, h)
            axe.plot(y, predict_data[:, index_h, index], label=f'h = {h}', color=color)

            x_point, y_point = [], []
            for x_item, y_item in zip(x_data, y_data):
                if x_item[0] == h:
                    x_point.append(x_item[1])
                    if isinstance(y_item, Iterable):
                        y_point.append(y_item[index])
                    else:
                        y_point.append(y_item)

            axe.scatter(x_point, y_point, color=color)

        axe.legend()

        axe.set_xlabel('l')
        axe.set_ylabel(label)
        axe.set_title('a)')

        fig.savefig(os.path.join('scolar_value_net', 'figure', f'{self.metric_name}_l_predict_{index}'))

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

        fig.savefig(os.path.join('scolar_value_net', 'figure', f'{self.metric_name}_predict3d_{index}'))

    def print_metrics(self):
        """
        Выводит графики метрик после обучения
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        axe_mse, axe_mae, axe_mape, axe_msle = axes.flatten()

        _data = self.handler.history.history
        data_mse, data_mae, data_mape, data_msle = _data.get('mse', []), _data.get('mae', []), _data.get('mape', []), _data.get('msle', [])

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
        #plt.show()

    def setup_metric_axe(self, axe: Axes, data, axe_name):
        """
        Задает настройки графику метрики.

        Args:
            axe:
            data:
            axe_name:
        """
        axe.plot(data)
        axe.set_ylim(0, data[5])
        axe.grid(True)
        axe.set_title(axe_name)


class PVINetFacade(ScolarValueFacade):
    """
    Фасад для работы с результатами обучения
    """
    calc_method = PVINetHandler

    metric_title = 'PVI'
    metric_name = 'PVI'

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = super()._prepare_data(data)

        return x, y[:, 1]

    def build_predict_all_figures(self, predict_data):
        """
        Собирает все графики по прогнозам
        """
        self.build_predict_figures(predict_data, 0, 'PVI')


class QNetFacade(ScolarValueFacade):
    """
    Фасад для работы с результатами обучения
    """
    calc_method = QNetHandler

    metric_title = 'Q'
    metric_name = 'Q'

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = super()._prepare_data(data)

        return x, y[:, 2]

    def build_predict_all_figures(self, predict_data):
        """
        Собирает все графики по прогнозам
        """
        self.build_predict_figures(predict_data, 0, 'q')


class ENetFacade(ScolarValueFacade):
    """
    Фасад для работы с результатами обучения
    """
    calc_method = ENetHandler

    metric_title = 'E'
    metric_name = 'E'

    def _prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = super()._prepare_data(data)

        return x, y[:, 0]

    def build_predict_all_figures(self, predict_data):
        """
        Собирает все графики по прогнозам
        """
        self.build_predict_figures(predict_data, 0, 'E')
