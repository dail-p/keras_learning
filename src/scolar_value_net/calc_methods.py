import os
from typing import (
    Iterable,
)

from keras.layers import (
    Dense,
    Input,
)
from keras.optimizers import (
    Adam,
)

from calc_methods import (
    BaseNetworkHandler,
)


class ScolarValueNetHandler(BaseNetworkHandler):
    """
    Обработчик сети с сколярными выходными параметрами
    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'scolar_value')

    def _get_layers(self) -> Iterable:
        return [
            Input(shape=(2,)),
            Dense(units=5, activation='sigmoid'),
            Dense(units=3, activation='linear'),
        ]

    def _get_config_model(self) -> dict:
        return {
            'loss': 'mse',
            'optimizer': Adam(0.05),
            'metrics': [
                'mse',
                'mae',
                'mape',
                'msle'
            ]
        }

    def _get_config_trains(self) -> dict:
        return {
            'epochs': 400,
            'verbose': False,
        }


class PVINetHandler(ScolarValueNetHandler):
    """
    Обработчик сети с одним выходным параметром PVI
    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'scolar_value_PVI')

    def _get_layers(self) -> Iterable:
        return [
            Input(shape=(2,)),
            Dense(units=4, activation='sigmoid'),
            Dense(units=1, activation='linear'),
        ]


class QNetHandler(ScolarValueNetHandler):
    """
    Обработчик сети с одним выходным параметром Q
    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'scolar_value_Q')

    def _get_layers(self) -> Iterable:
        return [
            Input(shape=(2,)),
            Dense(units=4, activation='sigmoid'),
            Dense(units=1, activation='linear'),
        ]


class ENetHandler(ScolarValueNetHandler):
    """
    Обработчик сети с одним выходным параметром E
    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'scolar_value_E')

    def _get_layers(self) -> Iterable:
        return [
            Input(shape=(2,)),
            Dense(units=4, activation='sigmoid'),
            Dense(units=1, activation='linear'),
        ]
