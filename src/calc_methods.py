from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Iterable,
    Tuple,
)

import numpy as np
from keras.models import (
    Model,
    Sequential,
    load_model,
)


class BaseNetworkHandler(ABC):
    """
    Базовый класс для работы с нейронной сетью и ее настройками
    """
    # Модель сети для обучения
    model_cls = Sequential
    # Какую часть данных брать для обучения
    share_of_data_for_train = 0.8

    def __init__(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        """
        Args:
            x: списиок выходных параметров
            y: список выходных параметров
        """
        super().__init__(*args, **kwargs)
        self.model = self.create_model()
        self.x_train, self.y_train, self.x_test, self.y_test = self._share_data(x, y)

    @abstractmethod
    def _get_model_path(self) -> str:
        """
        Возвращает путь к файлу подели для сохранения/загрузки
        """
        pass

    @abstractmethod
    def _get_layers(self) -> Iterable:
        """
        Возвращает слои сети
        """

    @abstractmethod
    def _get_config_model(self) -> dict:
        """
        Возвращает конфигурацию сети
        """

    @abstractmethod
    def _get_config_trains(self) -> dict:
        """
        Возвращает конфигурацию модели обучения
        """

    def _share_data(self, x: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Разделяет входные данные на выборку для обучения и лдя тестирования
        """
        index_train = int(self.share_of_data_for_train * len(x))

        return x[:index_train], y[:index_train], x[index_train:], y[index_train:]

    def create_model(self) -> Model:
        """
        Создает модель нейроной сети
        """
        model = self.model_cls(self._get_layers())

        model.compile(**self._get_config_model())

        return model

    def save(self) -> None:
        """
        Сохраняет модель нейронной сети в файл
        """
        self.model.save(self._get_model_path())

    def load(self) -> None:
        """
        Загружает модель нейронной сети из файла
        """
        self.model = load_model(self._get_model_path())

    def run(self):
        """
        Обучает нейронную сеть
        """
        self.history = self.model.fit(self.x_train, self.y_train, **self._get_config_trains())

        self.save()

    def get_metrics_values(self):
        """
        Возвращает метрики обученной сети на тестовой выборке
        """
        return self.model.evaluate(self.x_test, self.y_test, return_dict=True)
