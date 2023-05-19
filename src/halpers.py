import os
import numpy as np


DATA_FILE_NAME = 'KIN_98.csv'


def load_data():
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
        if DATA_FILE_NAME in files:
            data = np.loadtxt(
                os.path.join(current_path, DATA_FILE_NAME),
                delimiter='\t',
            )

            data[:, 0] = get_length_value_list(data[:, 0])

            curvature_index = int(current_path[-1])
            curvature_value = get_curvature_value(curvature_index)

            data = np.insert(data, 0, 10*[curvature_value], axis=1)

            if result is None:
                result = data
            else:
                result = np.vstack([result, data])

    return result


def prepare_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    Args:
        data: Матрица дынных

    Returns: Векторы входных и выходных данных
    """
    np.random.shuffle(data)
    return data[:, :2], data[:, 2:]
