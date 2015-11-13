import collections
import functools
from itertools import zip_longest

import django
from django.conf import settings


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
