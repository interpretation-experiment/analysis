import os

import pytest

import analysis


@pytest.fixture(scope='session')
def models():
    db_name = os.environ['DB_NAME']
    return analysis.setup(db_name)


@pytest.fixture(scope='session')
def nlp():
    return analysis.utils.get_nlp()
