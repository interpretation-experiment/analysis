import pickle
import functools
from itertools import zip_longest

from . import settings


def setup_spreadr(db_name):
    """Setup environment for using spreadr models and database.

    This loads the `spreadr` folder into `sys.path`, connects to `db_name`
    on mysql which should be a spreadr database, and does initial django setup.

    """

    import os
    import sys
    notebook_base = os.path.join(settings.NOTEBOOKS_FOLDER, db_name[8:])
    # The actual spreadr source
    sys.path.append(os.path.join(notebook_base, 'spreadr'))
    # And the corresponding environment
    spreadr_lib = os.path.join(notebook_base, 'spreadr_env', 'lib')
    pythonVersion = [f for f in os.listdir(spreadr_lib)
                     if f.startswith('python')][0]
    sys.path.append(os.path.join(spreadr_lib, pythonVersion, 'site-packages'))

    import django
    from django.conf import settings as django_settings
    from spreadr import settings_analysis as spreadr_settings

    spreadr_settings = spreadr_settings.__dict__.copy()
    spreadr_settings['DATABASES']['default']['NAME'] = db_name
    django_settings.configure(**spreadr_settings)
    django.setup()


def grouper(iterable, n, fillvalue=None):
    """Iterate over `n`-wide slices of `iterable`, filling the
    last slice with `fillvalue`."""

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class memoized(object):
    """Decorate a function to cache its return value each time it is called.

    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    """

    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, self.func)

    def __call__(self, *args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value

    def __get__(self, obj, objtype):
        """Support instance methods."""
        func = functools.partial(self.__call__, obj)
        functools.update_wrapper(func, self)
        func.drop_cache = self.drop_cache
        return func

    def drop_cache(self):
        self.cache = {}


@memoized
def unpickle(filename):
    """Load a pickle file at path `filename`.

    This function is :func:`memoized` so a file is only loaded the first time.

    """

    with open(filename, 'rb') as file:
        return pickle.load(file)


def mpl_palette(n_colors, variation='Set2'):  # or variation='colorblind'
    """Get any seaborn palette as a usable matplotlib colormap."""

    import seaborn as sb
    palette = sb.color_palette(variation, n_colors, desat=0.8)
    return (sb.blend_palette(palette, n_colors=n_colors, as_cmap=True),
            sb.blend_palette(palette, n_colors=n_colors))


def import_spreadr_models():
    """Get spreadr models, and bail if spreadr has not yet been setup."""

    try:
        from gists import models
    except ImportError:
        raise ImportError('`gists` models not found for import, '
                          'you might need to run `setup_spreadr() first')
    return models


def equip_model_managers_with_bucket_type(models):
    """Add `training`, `experiment`, `game` to `Sentence` and `Tree` model
    managers.

    These let you quickly narrow down a queryset to the named bucket.

    """

    from django.db.models.manager import Manager

    def filter_bucket(self, bucket_name):
        qs = self.get_queryset()
        if self.model == models.Sentence:
            return qs.filter(bucket__exact=bucket_name)
        elif self.model == models.Tree:
            return qs.filter(root__bucket__exact=bucket_name)
        else:
            raise ValueError('Only available on Sentence and Tree')

    # This will work for Sentence and Tree
    Manager.training = property(lambda self: filter_bucket(self, 'training'))
    Manager.experiment = property(
        lambda self: filter_bucket(self, 'experiment'))
    Manager.game = property(lambda self: filter_bucket(self, 'game'))


def equip_sentence_with_head_depth(models):
    """Define `head` and `depth` on `Sentence`s."""

    @memoized
    def get_head(self):
        """Get the head of the branch this sentence is in,
        bailing if the sentence is root."""

        if self.parent is None:
            raise ValueError('Already at root')
        if self.parent.parent is None:
            return self
        return self.parent.head

    models.Sentence.head = property(get_head)

    @memoized
    def get_depth(self):
        """Get the depth of the sentence in its branch."""

        if self.parent is None:
            return 0
        return 1 + self.parent.depth

    models.Sentence.depth = property(get_depth)


def equip_spreadr_models():
    """Equip spreadr models with useful tools.

    Tools:
    * Bucket selection on `Sentence` and `Tree` model managers
    * Head of branch and depth in branch for a `Sentence`

    """

    models = import_spreadr_models()
    equip_model_managers_with_bucket_type(models)
    equip_sentence_with_head_depth(models)
