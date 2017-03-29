def test_equip_sentence_shaping(models):
    from django.core.exceptions import ObjectDoesNotExist
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
    assert models.Sentence.objects.with_dropped(True).get(id=1) is not None
    assert models.Sentence.objects.with_dropped(False).get(id=1) is not None
    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert (models.Sentence.objects.with_dropped(True).get(id=901)
                is not None)
        try:
            models.Sentence.objects.with_dropped(False).get(id=901)
        except ObjectDoesNotExist:
            pass  # Test passed
        else:
            raise Exception('DoesNotExist not raised')


def test_equip_model_managers_with_bucket_type(models):
    from django.conf import settings as django_settings

    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_2':
        assert models.Sentence.objects.training.count() == 255
        assert models.Tree.objects.training.count() == 5

    assert models.Sentence.objects.experiment.count() == \
        (models.Sentence.objects.count()
         - models.Sentence.objects.training.count()
         - models.Sentence.objects.game.count())
    assert models.Tree.objects.experiment.count() == \
        (models.Tree.objects.count()
         - models.Tree.objects.training.count()
         - models.Tree.objects.game.count())

    try:
        models.Profile.objects.training
    except ValueError:
        pass  # Test passed
    else:
        raise Exception('ValueError not raised on '
                        'Profile.objects.training')


def test_equip_sentence_with_head_depth(models):
    from django.db.models import Count

    tree = models.Tree.objects\
        .annotate(sentences_count=Count('sentences'))\
        .filter(sentences_count__gte=10).first()
    heads = set(tree.root.children.all())

    def _add_with_children(sentence, children, depth):
        children.append((sentence, depth))
        for child in sentence.children.all():
            _add_with_children(child, children, depth + 1)

    def walk_children(sentence, depth):
        res = []
        _add_with_children(sentence, res, depth)
        return res

    branches = {}
    for head in heads:
        branches[head] = walk_children(head, 1)

    for sentence in tree.sentences.all():
        if sentence == tree.root:
            assert sentence.depth == 0
            try:
                sentence.head
            except ValueError:
                pass  # Test passed
            else:
                raise Exception('Exception not raised on root')
        else:
            assert (sentence, sentence.depth) in branches[sentence.head]
