from os.path import join, abspath, dirname


BASE_FOLDER = dirname(dirname(abspath(__file__)))

NOTEBOOKS_FOLDER = join(BASE_FOLDER, 'notebooks')
DATA_FOLDER = join(BASE_FOLDER, 'data')

CODINGS_FOLDER = join(DATA_FOLDER, 'codings')

AOA = join(DATA_FOLDER, 'Kuperman-BRM-data-2012.csv')
CLEARPOND = join(DATA_FOLDER, 'englishCPdatabase2.txt')
FREQUENCY = join(DATA_FOLDER, 'SUBTLEX-UK.txt')
CLUSTERING = join(DATA_FOLDER, 'fa_clustering.pickle')

MODELS_FOLDER = join(DATA_FOLDER, 'models')
MODEL_TEMPLATE = join(MODELS_FOLDER, '{n}-gram_{type}.pickle')

ALIGNMENT_GAP_CHAR = '-'
# Obtained through optimisation in notebooks/exp_3/optimise_alignment.ipynb
ALIGNMENT_PARAMETERS = {
    'COMPARE_FACTOR': 1.0,
    'COMPARE_ORIGIN': -0.88888888888888884,
    'GAP_OPEN': -0.29222222222222227,
    'GAP_EXTEND': -0.11888888888888893,
    'EXCHANGE': -.5,
}
