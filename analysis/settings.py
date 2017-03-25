from os.path import join, abspath, dirname


BASE_FOLDER = dirname(dirname(abspath(__file__)))
NOTEBOOKS_FOLDER = join(BASE_FOLDER, 'notebooks')

DATA_FOLDER = join(BASE_FOLDER, 'data')
AOA = join(DATA_FOLDER, 'Kuperman-BRM-data-2012.csv')
CLEARPOND = join(DATA_FOLDER, 'englishCPdatabase2.txt')
FREQUENCY = join(DATA_FOLDER, 'SUBTLEX-UK.txt')

MODELS_FOLDER = join(DATA_FOLDER, 'models')
MODEL_TEMPLATE = join(MODELS_FOLDER, '{n}-gram_{type}.pickle')
