import numpy as np


# TODO: refresh tests


def test_equip_sentence_content_words(models):
    from django.conf import settings as django_settings

    # Test
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.get(id=1).content_words == \
            ['young', 'boy', 'sudden', 'hit', 'littl', 'girl']
        assert models.Sentence.objects.get(id=2).content_words == \
            ['forget', 'leav', 'door', 'open', 'leav', 'offic']


def test_equip_sentence_distances(models):
    from django.conf import settings as django_settings

    # Testing distance is hard (we don't have predictable data for it),
    # so we mostly test for stupid values only
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.get(id=1).raw_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert models.Sentence.objects.get(id=1).ordered_content_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert models.Sentence.objects.get(id=1).unordered_content_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert np.abs(models.Sentence.objects.get(id=1).raw_distance(
                models.Sentence.objects.get(id=2)) - .754098) <= 1e-6
        assert models.Sentence.objects.get(id=1).ordered_content_distance(
                models.Sentence.objects.get(id=2)) == 1.0
        assert models.Sentence.objects.get(id=1).unordered_content_distance(
                models.Sentence.objects.get(id=2)) == 1.0

    # cum_root_distance is also hard to test, so testing stupid values
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.get(id=580).cum_root_distance(
            'raw') == 0
        assert models.Sentence.objects.get(id=580).cum_root_distance(
            'raw', normalized=False) == 0
        assert models.Sentence.objects.get(id=580).cum_root_distance(
            'ordered_content') == 0
        assert models.Sentence.objects.get(id=580).cum_root_distance(
            'ordered_content', normalized=False) == 0
        assert models.Sentence.objects.get(id=580).cum_root_distance(
            'unordered_content') == 0
        assert models.Sentence.objects.get(id=580).cum_root_distance(
            'unordered_content', normalized=False) == 0
        assert models.Sentence.objects.get(id=823).cum_root_distance(
            'raw') == .02
        assert models.Sentence.objects.get(id=823).cum_root_distance(
            'raw', normalized=False) == 1
        assert models.Sentence.objects.get(id=823).cum_root_distance(
            'ordered_content') == 0
        assert models.Sentence.objects.get(id=823).cum_root_distance(
            'ordered_content', normalized=False) == 0
        assert models.Sentence.objects.get(id=823).cum_root_distance(
            'unordered_content') == 0
        assert models.Sentence.objects.get(id=823).cum_root_distance(
            'unordered_content', normalized=False) == 0
        assert abs(models.Sentence.objects.get(id=1115).cum_root_distance(
            'raw') - .308333) < 1e-6
        assert models.Sentence.objects.get(id=1115).cum_root_distance(
            'raw', normalized=False) == 15
        assert abs(models.Sentence.objects.get(id=1115)
                   .cum_root_distance('ordered_content') - .166666) < 1e-6
        assert models.Sentence.objects.get(id=1115).cum_root_distance(
            'ordered_content', normalized=False) == 1
        assert abs(models.Sentence.objects.get(id=1115)
                   .cum_root_distance('unordered_content') - .166666) < 1e-6
        assert abs(models.Sentence.objects.get(id=1115)
                   .cum_root_distance('unordered_content', normalized=False) -
                   .166666) < 1e-6


def test_equip_profile_transformation_rate(models):
    try:
        models.Profile.objects.get(
            user__username='sl').transformation_rate('raw')
    except ValueError:
        pass  # Test passed
    else:
        raise Exception("Exception not raised on profile with "
                        "no reformulated sentences")
    # And with with_spam=True
    try:
        models.Profile.objects.get(
            user__username='sl').transformation_rate('raw', with_spam=True)
    except ValueError:
        pass  # Test passed
    else:
        raise Exception("Exception not raised on profile with "
                        "no reformulated sentences")
