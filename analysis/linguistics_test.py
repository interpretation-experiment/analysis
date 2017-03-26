import numpy as np


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


def test_equip_sentence_codings(models):
    from django.conf import settings as django_settings

    # Test spam (hard to test, so only checking that what we entered as first
    # sentences is not spam)
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert len(models.Sentence.objects.get(id=1).spam_detail[0]) == 2
        assert not models.Sentence.objects.get(id=1).spam_detail[0][0]
        assert not models.Sentence.objects.get(id=1).spam
        assert len(models.Sentence.objects.get(id=2).spam_detail[0]) == 2
        assert not models.Sentence.objects.get(id=2).spam_detail[0][0]
        assert not models.Sentence.objects.get(id=2).spam
        assert len(models.Sentence.objects.get(id=3).spam_detail[0]) == 2
        assert not models.Sentence.objects.get(id=3).spam_detail[0][0]
        assert not models.Sentence.objects.get(id=3).spam

    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.get(id=901).spam
        assert models.Sentence.objects.get(id=2608).spam
        assert not models.Sentence.objects.get(id=244).spam

    # Test nonspam
    try:
        models.Profile.objects.nonspam
    except ValueError:
        pass  # Test passed
    else:
        raise Exception('ValueError not raised on Profile.objects.nonspam')
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.nonspam.get(id=1) is not None
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.nonspam.get(id=6) is not None

    # Test doublepost
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.get(id=258).doublepost
        assert models.Sentence.objects.get(id=1557).doublepost
        assert models.Sentence.objects.get(id=442).doublepost
        assert models.Sentence.objects.get(id=677).doublepost
        assert models.Sentence.objects.get(id=1453).doublepost

    # Test nondoublepost
    try:
        models.Profile.objects.nondoublepost
    except ValueError:
        pass  # Test passed
    else:
        raise Exception('ValueError not raised on '
                        'Profile.objects.nondoublepost')
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.nondoublepost.get(id=984) is not None

    # Test rogue
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.get(id=486).rogue
        assert not models.Sentence.objects.get(id=489).rogue
        assert not models.Sentence.objects.get(id=2081).rogue
        assert models.Sentence.objects.get(id=2084).rogue

    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert not models.Sentence.objects.get(id=1452).rogue
        assert models.Sentence.objects.get(id=1453).rogue

    # Test nonrogue
    try:
        models.Profile.objects.nonrogue
    except ValueError:
        pass  # Test passed
    else:
        raise Exception('ValueError not raised on '
                        'Profile.objects.nonrogue')
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.nonrogue.get(id=984) is not None

    # Test kept
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.kept.get(id=1) is not None
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.kept.get(id=6) is not None
        assert models.Sentence.objects.kept.get(id=1618) is not None

    # Test with_dropped
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.with_dropped(True).get(id=1) is not None
        assert (models.Sentence.objects.with_dropped(False).get(id=1)
                is not None)


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
