from os.path import join, abspath, dirname


base_folder = dirname(dirname(abspath(__file__)))

DATA_FOLDER = join(base_folder, 'data')
AOA = join(DATA_FOLDER, 'Kuperman-BRM-data-2012.csv')
CLEARPOND = join(DATA_FOLDER, 'englishCPdatabase2.txt')
FREQUENCY = join(DATA_FOLDER, 'SUBTLEX-UK.txt')

MODELS_FOLDER = join(DATA_FOLDER, 'models')
MODEL_TEMPLATE = join(MODELS_FOLDER, '{n}-gram_{type}.pickle')
