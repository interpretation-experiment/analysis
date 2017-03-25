from os.path import join, abspath, dirname


settings_folder = dirname(abspath(__file__))

MODELS_FOLDER = join(settings_folder, 'models')
MODEL_TEMPLATE = join(MODELS_FOLDER, '{n}-gram_{type}.pickle')

DATA_FOLDER = join(settings_folder, 'data')
AOA = join(DATA_FOLDER, 'Kuperman-BRM-data-2012.csv')
CLEARPOND = join(DATA_FOLDER, 'englishCPdatabase2.txt')
FREQUENCY = join(DATA_FOLDER, 'SUBTLEX-UK.txt')
