def test_equip_model_managers_with_bucket_type(models):
    from django.conf import settings as django_settings

    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
        assert models.Sentence.objects.training.count() == 6
        assert models.Sentence.objects.experiment.count() == \
            (models.Sentence.objects.count() - 6 -
             models.Sentence.objects.game.count())
        assert models.Tree.objects.training.count() == 6
        assert models.Tree.objects.experiment.count() == \
            models.Tree.objects.count() - 6 - models.Tree.objects.game.count()
        try:
            models.Profile.objects.training
        except ValueError:
            pass  # Test passed
        else:
            raise Exception('ValueError not raised on '
                            'Profile.objects.training')


def test_equip_sentence_with_head_depth(models):
    from django.db.models import Count
    from django.conf import settings as django_settings

    if django_settings.DATABASES['default']['NAME'] == 'spreadr_exp_1':
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
