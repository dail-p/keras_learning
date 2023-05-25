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
            'loss': 'mae',
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
            'epochs': 300,
            'verbose': False,
        }


class PVINetHandler(ScolarValueNetHandler):
    """

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

    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'scolar_value_Q')

    def _get_layers(self) -> Iterable:
        return [
            Input(shape=(2,)),
            Dense(units=4, activation='sigmoid'),
            Dense(units=1, activation='linear'),
        ]


class FNetHandler(ScolarValueNetHandler):
    """

    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'scolar_value_F')

    def _get_layers(self) -> Iterable:
        return [
            Input(shape=(2,)),
            Dense(units=4, activation='sigmoid'),
            Dense(units=1, activation='linear'),
        ]
