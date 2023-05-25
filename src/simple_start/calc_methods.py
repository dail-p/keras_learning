import os
from typing import (
    Iterable,
)

from keras.layers import (
    Dense,
)
from keras.metrics import (
    Precision,
)
from keras.optimizers import (
    Adam,
)

from calc_methods import (
    BaseNetworkHandler,
)


class SimpleStartHandler(BaseNetworkHandler):
    """
    Класс для решения простой задачи переводы градусов из С в F с помощью нейронной сети
    """
    def _get_model_path(self) -> str:
        return os.path.join('models', 'simple_start')

    def _get_layers(self) -> Iterable:
        return [
            Dense(units=1, input_shape=(1,), activation='linear')
        ]

    def _get_config_model(self) -> dict:
        return {
            'loss': 'mean_squared_error',
            'optimizer': Adam(0.1),
            'metrics': [Precision()]
        }

    def _get_config_trains(self) -> dict:
        return {
            'epochs': 200,
            'verbose': True,
        }
