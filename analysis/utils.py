import os
import csv
import pickle
import functools
from itertools import zip_longest

import spacy
import numpy as np

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


def quantile_interval(values, target):
    """Get the span of `target` in the distribution `values`.

    This is the actual quantile occupied by `target` in the `values`
    distribution, expressed as an interval in [0; 1].

    """

    if np.isnan(target) or (target not in values):
        return np.nan, np.nan
    finite_values = values[np.isfinite(values)]
    sorted_values = np.array(sorted(finite_values))
    length = len(sorted_values)
    ours = np.where(sorted_values == target)[0]
    return ours[0] / length, (ours[-1] + 1) / length


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


def load_codings(db, coding, mapper):
    """Load the files for codings of `coding`, for the given `db`, extracting
    key `coding` from each csv file, and processing the values with
    `mapper`."""

    # Get the list of files to load
    folder = os.path.join(settings.CODINGS_FOLDER, db, coding)
    filepaths = [os.path.join(folder, name)
                 for name in next(os.walk(folder))[2]]

    # Load the files
    codings = {}
    for filepath in filepaths:
        coder = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = int(row['id'])
                code = mapper(row[coding])
                if id not in codings:
                    codings[id] = []
                codings[id].append((code, coder))

    return codings


def import_spreadr_models():
    """Get spreadr models, and bail if spreadr has not yet been setup."""

    try:
        from gists import models
    except ImportError:
        raise ImportError('`gists` models not found for import, '
                          'you might need to run `setup_spreadr()` first')
    return models


@memoized
def get_nlp():
    return spacy.load('en_core_web_md')
