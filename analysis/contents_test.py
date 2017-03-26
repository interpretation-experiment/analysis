from numpy.testing import assert_approx_equal


def test_equip_sentence_words(models):
    from django.conf import settings as django_settings

    # Test
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.get(id=1).words == \
            ("it", "was", "that", "night", "i", "discovered", "that", "most",
             "things", "you", "consider", "evil", "or", "wicked", "are",
             "simply", "lonely", "and", "lacking", "in", "the", "social",
             "niceties")
        assert models.Sentence.objects.get(id=2).words == \
            ("see", "most", "men", "they", "'ll", "tell", "a", "story",
             "straight", "through", "and", "it", "wo", "n't", "be",
             "complicated", "but", "it", "wo", "n't", "be", "interesting",
             "either")
        assert models.Sentence.objects.get(id=1).content_words == \
            ('night', 'discovered', 'things', 'consider', 'evil', 'wicked',
             'simply', 'lonely', 'lacking', 'social', 'niceties')
        # spaCy doesn't recognise "wo" in "won't" as "will" (why not? seems
        # pretty standard), leading it to show up in content words. But this
        # only adds a little more strictness to content word measures, which is
        # no fundamental problem
        assert models.Sentence.objects.get(id=2).content_words == \
            ('men', 'tell', 'story', 'straight', 'wo', 'complicated', 'wo',
             'interesting')
        assert models.Sentence.objects.get(id=1).content_lemmas == \
            ('night', 'discover', 'thing', 'consider', 'evil', 'wicked',
             'simply', 'lonely', 'lack', 'social', 'nicety')
        assert models.Sentence.objects.get(id=2).content_lemmas == \
            ('man', 'tell', 'story', 'straight', 'wo', 'complicated', 'wo',
             'interesting')
        assert models.Sentence.objects.get(id=1).content_ids == \
            (3, 5, 8, 10, 11, 13, 15, 16, 18, 21, 22)
        assert models.Sentence.objects.get(id=2).content_ids == \
            (2, 5, 7, 8, 12, 15, 18, 21)


def test_equip_sentence_distances(models):
    from django.conf import settings as django_settings

    # Testing distance is hard (we don't have predictable data for it),
    # so we mostly test for stupid values only
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.get(id=1).raw_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert models.Sentence.objects.get(id=1).ow_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert models.Sentence.objects.get(id=1).oc_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert models.Sentence.objects.get(id=1).uc_distance(
                models.Sentence.objects.get(id=1)) == 0.0
        assert_approx_equal(models.Sentence.objects.get(id=26)
                            .raw_distance(models.Sentence.objects.get(id=160)),
                            .32941176470588235)
        assert models.Sentence.objects.get(id=26).ow_distance(
                models.Sentence.objects.get(id=160)) == .375
        assert_approx_equal(models.Sentence.objects.get(id=26)
                            .oc_distance(models.Sentence.objects.get(id=160)),
                            .16666666666666666)
        assert_approx_equal(models.Sentence.objects.get(id=26)
                            .uc_distance(models.Sentence.objects.get(id=160)),
                            .16666666666666666)

    # cum_root_distance is also hard to test, so testing stupid values
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert_approx_equal(models.Sentence.objects.get(id=2060)
                            .cum_root_distance('raw'), .012987012987012988)
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'raw', normalized=False) == 1
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'ow') == 0
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'ow', normalized=False) == 0
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'oc') == 0
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'oc', normalized=False) == 0
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'uc') == 0
        assert models.Sentence.objects.get(id=2060).cum_root_distance(
            'uc', normalized=False) == 0
        assert_approx_equal(models.Sentence.objects.get(id=1347)
                            .cum_root_distance('raw'), .37662337662337664)
        assert models.Sentence.objects.get(id=1347).cum_root_distance(
            'raw', normalized=False) == 29
        assert models.Sentence.objects.get(id=1347).cum_root_distance(
            'ow') == .4375
        assert models.Sentence.objects.get(id=1347).cum_root_distance(
            'ow', normalized=False) == 7
        assert models.Sentence.objects.get(id=1347).cum_root_distance(
            'oc') == 0.6
        assert models.Sentence.objects.get(id=1347).cum_root_distance(
            'oc', normalized=False) == 3
        assert_approx_equal(models.Sentence.objects.get(id=1347)
                            .cum_root_distance('uc'), .6666666666666666)
        # UC distance is always normalised
        assert_approx_equal(models.Sentence.objects.get(id=1347)
                            .cum_root_distance('uc', normalized=False),
                            .6666666666666666)

    try:
        models.Sentence.objects.get(id=1).cum_root_distance('unknown')
    except ValueError:
        pass  # Test passed
    else:
        raise Exception('Exception not raised on sentence for unknown '
                        'cum_root_distance')


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
