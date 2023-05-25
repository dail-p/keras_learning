import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scolar_value_net.api import (
    FNetFacade,
    PVINetFacade,
    QNetFacade,
    ScolarValueFacade,
)


if __name__ == '__main__':
    ScolarValueFacade()
    PVINetFacade()
    QNetFacade()
    FNetFacade()
