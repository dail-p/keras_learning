import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scolar_value_net.api import ScolarValueFacade


if __name__ == '__main__':
    ScolarValueFacade()
