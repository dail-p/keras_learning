import matplotlib.pyplot as plt
import numpy as np

from simple_start.calc_methods import (
    SimpleStartHandler,
)


def celsius_to_fahrenheit():
    """
    Обучает сеть на перевод градусов из С в F.
    F = C * 1.8 + 32
    """
    c = np.random.sample(200) * 200 - 100
    f = 1.8 * c + 32

    handler = SimpleStartHandler(c, f)
    history = handler.run()
    print("Обучение завершено")

    #print(handler.model.predict([100]))
    print(handler.model.get_weights())

    handler.get_metrics_values()

    plt.plot(history.history['loss'])
    plt.ylim([0, 100])
    plt.grid(True)
    plt.show()
