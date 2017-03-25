import os

import pytest

from analysis import utils


@pytest.fixture(scope='session')
def models():
    db_name = os.environ['DB_NAME']
    utils.setup_spreadr(db_name)
    return utils.import_spreadr_models()
