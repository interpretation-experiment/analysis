import collections
import functools
from itertools import zip_longest

import django
from django.conf import settings
from django.db.models import Count


DB_USER = 'spreadr_analysis'


def setup_spreadr(db_name):
    import os, sys
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'spreadr'))
    from spreadr import settings as base_spreadr_settings

    spreadr_settings = base_spreadr_settings.__dict__.copy()
    spreadr_settings['DATABASES'] = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': db_name,
            'USER': DB_USER
        }
    }
    settings.configure(**spreadr_settings)
    django.setup()


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated)."""

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def mpl_palette(n_colors, variation='Set2'):  # or variation='colorblind'
    import seaborn as sb
    palette = sb.color_palette(variation, n_colors, desat=0.8)
    return (sb.blend_palette(palette, n_colors=n_colors, as_cmap=True),
            sb.blend_palette(palette, n_colors=n_colors))


def import_spreadr_models():
    try:
        from gists import models
    except ImportError:
        raise ImportError('`gists` models not found for import, '
                          'you might need to run `spreadr_setup() first')
    return models


def equip_model_managers_with_bucket_type(models):
    # For sentences
    models.Sentence.objects.__class__.training = property(lambda self: \
            self.get_queryset().filter(bucket__exact='training'))
    models.Sentence.objects.__class__.experiment = property(lambda self: \
            self.get_queryset().filter(bucket__exact='experiment'))
    models.Sentence.objects.__class__.game = property(lambda self: \
            self.get_queryset().filter(bucket__exact='game'))

    # Test
    assert models.Sentence.objects.training.count() == 6
    assert models.Sentence.objects.experiment.count() == \
            models.Sentence.objects.count() - 6 - models.Sentence.objects.game.count()

    # For trees
    models.Tree.objects.__class__.training = property(lambda self: \
            self.get_queryset().filter(root__bucket__exact='training'))
    models.Tree.objects.__class__.experiment = property(lambda self: \
            self.get_queryset().filter(root__bucket__exact='experiment'))
    models.Tree.objects.__class__.game = property(lambda self: \
            self.get_queryset().filter(root__bucket__exact='game'))

    # Test
    assert models.Tree.objects.training.count() == 6
    assert models.Tree.objects.experiment.count() == \
            models.Tree.objects.count() - 6 - models.Tree.objects.game.count()


def equip_sentence_with_head(models):

    def get_head(self):
        if self.parent is None:
            raise ValueError('Already at root')
        if self.parent.parent is None:
            return self
        return self.parent.head

    models.Sentence.head = property(get_head)

    # Test
    tree = models.Tree.objects.annotate(sentences_count=Count('sentences')).filter(\
            sentences_count__gte=10).first()
    heads = set(tree.root.children.all())
    def _add_with_children(sentence, children):
        children.append(sentence)
        for child in sentence.children.all():
            _add_with_children(child, children)
    def walk_children(sentence):
        res = []
        _add_with_children(sentence, res)
        return res
    branches = {}
    for head in heads:
        branches[head] = walk_children(head)

    for sentence in tree.sentences.all():
        if sentence == tree.root:
            try:
                sentence.head
            except ValueError:
                pass  # Test passed
            else:
                raise Exception('Exception not raised on root')
        else:
            assert sentence in branches[sentence.head]


def equip_spreadr_models():
    models = import_spreadr_models()
    equip_model_managers_with_bucket_type(models)
    equip_sentence_with_head(models)
