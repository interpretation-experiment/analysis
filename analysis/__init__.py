from . import utils
from . import contents
from . import shaping
from . import features
from . import transformations


def setup(db_name):
    utils.setup_spreadr(db_name)
    models = utils.import_spreadr_models()

    shaping.equip_sentence_shaping(models)
    shaping.equip_model_managers_with_bucket_type(models)
    shaping.equip_sentence_with_head_depth(models)
    shaping.equip_sentence_branch_sentences(models)

    contents.equip_sentence_words(models)
    contents.equip_sentence_distances(models)
    contents.equip_profile_transformation_rate(models)

    transformations.equip_sentence_alignments(models)

    features.equip_sentence_features(models)

    return models
