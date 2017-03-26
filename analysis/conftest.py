import os

import pytest

from analysis import utils, linguistics


@pytest.fixture(scope='session')
def models():
    db_name = os.environ['DB_NAME']
    utils.setup_spreadr(db_name)
    utils.equip_spreadr_models()
    linguistics.equip_spreadr_models()
    return utils.import_spreadr_models()
