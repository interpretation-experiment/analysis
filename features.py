"""Features for words.

This module defines the :class:`Feature` which gives access to feature values
and related computed values (e.g. sentence-relative feature values).

A few other utility functions that load data for the features are also defined.

"""


import logging
from csv import DictReader, reader as csvreader
import warnings
import functools

import numpy as np
from nltk.corpus import cmudict

from utils import memoized
import settings


logger = logging.getLogger(__name__)


@memoized
def _get_pronunciations():
    """Get the CMU pronunciation data as a dict.

    The method is :func:`~.utils.memoized` since it is called so often.

    Returns
    -------
    dict
        Association of words to their list of possible pronunciations.

    """

    logger.debug('Loading CMU data')
    return cmudict.dict()


@memoized
def _get_aoa():
    """Get the Age-of-Acquisition data as a dict.

    The method is :func:`~.utils.memoized` since it is called so often.

    Returns
    -------
    dict
        Association of words to their average age of acquisition. `NA` values
        in the originating data set are ignored.

    """

    logger.debug('Loading Age-of-Acquisition data')

    aoa = {}
    with open(settings.AOA) as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            word = row['Word'].lower()
            mean = row['Rating.Mean']
            if mean == 'NA':
                continue
            if word in aoa:
                raise Exception("'{}' is already in the AoA dictionary"
                                .format(word))
            aoa[word] = float(mean)
    return aoa


@memoized
def _get_zipf_frequency():
    """Get the Frequency data as a dict.

    The method is :func:`~.utils.memoized` since it is called so often.

    Returns
    -------
    dict
        Association of words to their Zipf frequency.

    """

    logger.debug('Loading Frequency data')

    freqs = {}
    with open(settings.FREQUENCY) as csvfile:
        reader = DictReader(csvfile, delimiter='\t')
        for row in reader:
            word = row['Spelling'].lower()
            if len(word) == 0:
                # A few rows have empty data. Skip it.
                continue
            freq = row['LogFreq(Zipf)']
            if word in freqs:
                raise Exception("'{}' is already in the frequencies dictionary"
                                .format(word))
            freqs[word] = float(freq)
    return freqs


@memoized
def _get_clearpond():
    """Get CLEARPOND neighbourhood density data as a dict.

    The method is :func:`~.utils.memoized` since it is called so often.

    Returns
    -------
    dict
        `Dict` with two keys: `orthographic` and `phonological`. `orthographic`
        contains a dict associating words to their orthographic neighbourhood
        density (CLEARPOND's `OTAN` column). `phonological` contains a dict
        associating words to their phonological neighbourhood density
        (CLEARPOND's `PTAN` column).

    """

    logger.debug('Loading Clearpond data')

    clearpond_orthographic = {}
    clearpond_phonological = {}
    with open(settings.CLEARPOND, encoding='iso-8859-2') as csvfile:
        reader = csvreader(csvfile, delimiter='\t')
        for row in reader:
            word = row[0].lower()
            if word in clearpond_phonological:
                raise Exception("'{}' is already in the Clearpond "
                                'phonological dictionary'.format(word))
            if word in clearpond_orthographic:
                raise Exception("'{}' is already in the Clearpond "
                                'orthographic dictionary'.format(word))
            clearpond_orthographic[word] = int(row[5])
            clearpond_phonological[word] = int(row[29])
    return {'orthographic': clearpond_orthographic,
            'phonological': clearpond_phonological}


class Features:

    """Feature loading and computing.

    Methods in this class fall into 3 categories:

    * Raw feature methods: they are :func:`~.utils.memoized` class methods of
      the form `cls._feature_name(cls, word=None)`. Calling them with a `word`
      returns either the feature value of that word, or `np.nan` if the word is
      not encoded. Calling them with `word=None` returns the set of words
      encoded by that feature (which is used to compute e.g. averages over the
      pool of words encoded by that feature). Their docstring (which you will
      see below if you're reading this in a web browser) is the short name used
      to identify e.g. the feature's column in analyses in notebooks. These
      methods are used internally by the class, to provide the next category of
      methods.
    * Useful feature methods that can be used in analyses: :meth:`features`,
      :meth:`sentence_features`. These methods use the raw feature methods
      (previous category) and the utility methods (next category) to compute
      feature values (possibly relative to sentence).
    * Private utility methods: :meth:`_transformed_feature`. These methods are
      used by the previous category of methods.

    Read the source of the first category (raw features) to know how exactly an
    individual feature is computed. Read the docstrings (and source) of the
    second category (useful methods for analyses) to learn how to use this
    class in analyses. Read the docstrings (and source) of the third category
    (private utility methods) to learn how the whole class assembles its
    different parts together.

    """

    #: Association of available features to `(source_type, transform)` tuples:
    #: `source_type` defines if a feature is computed on tokens or lemmas, and
    #: `transform` defines how a feature value is transformed (for now either
    #: identity or log) because of the shape of its distribution (see the
    #: brainscopypaste `notebook/feature_distributions.ipynb` notebook for more
    #: details).
    __features__ = {
        # feature_name:           (source_type, transform)
        'letters_count':          ('orth_', lambda x: x),
        'aoa':                    ('lemma_', lambda x: x),
        'zipf_frequency':         ('orth_', lambda x: x),
        'orthographic_density':   ('orth_', np.log),
    }

    @memoized
    def sentence_features(self, name, doc, sentence_relative=None):
        """Compute the feature values for all words in `doc`, possibly
        sentence-relative.

        Feature values are transformed as explained in
        :meth:`_transformed_feature`.

        If `sentence_relative` is not `None`, it indicates a NumPy function
        used to aggregate word features in the sentence; this method then
        returns the sentence feature values minus the corresponding aggregate
        value. For instance, if `sentence_relative='median'`, this method
        returns the sentence features minus the median of the sentence (words
        valued at `np.nan` are ignored).

        The method is :func:`~.utils.memoized` since it is called so often.

        Parameters
        ----------
        name : str
            Name of the feature for which to compute source and destination
            values.
        doc : spacy.Doc
            The spaCy doc representing the sentence.
        sentence_relative : str, optional
            If not `None` (which is the default), return features relative to
            values of their corresponding sentence aggregated by this function;
            must be a name for which `np.nan<sentence_relative>` exists.

        Returns
        -------
        sentence_features : array of float
            Array of feature values (possibly sentence-relative) for each word
            in the sentence. Non-coded words appear as `np.nan`.

        """

        if name not in self.__features__:
            raise ValueError("Unknown feature: '{}'".format(name))

        # Get the sentence orth_'s or lemma_'s,
        # depending on the requested feature.
        source_type, _ = self.__features__[name]
        sentence_words = [getattr(w, source_type).lower() for w in doc]

        # Compute the features.
        feature = self._transformed_feature(name)
        sentence_features = np.array([feature(word) for word
                                      in sentence_words],
                                     dtype=np.float_)

        if sentence_relative is not None:
            pool = getattr(np, 'nan' + sentence_relative)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                sentence_features -= pool(sentence_features)

        return sentence_features

    @memoized
    def feature(self, name, doc, i, sentence_relative=None):
        """Compute feature `name` for the `i`-th word of `doc`, possibly
        sentence-relative.

        Feature values are transformed as explained in
        :meth:`_transformed_feature`.

        If `sentence_relative` is not `None`, it indicates a NumPy function
        used to aggregate word features in the sentence; this method then
        returns the word feature value minus the corresponding aggregate value.
        For instance, if `sentence_relative='median'`, this method returns the
        word feature minus the median of the sentence (words valued at `np.nan`
        are ignored).

        The method is :func:`~.utils.memoized` since it is called so often.

        Parameters
        ----------
        name : str
            Name of the feature for which to compute the value.
        doc : spacy.Doc
            The spaCy doc representing the sentence.
        i : int
            Index of the word in `doc` to look at.
        sentence_relative : str, optional
            If not `None` (which is the default), return feature relative to
            values of the sentence aggregated by this function; must be a name
            for which `np.nan<sentence_relative>` exists.

        Returns
        -------
        tuple of float
            Feature value (possibly sentence-relative) of the word.

        """

        return self.sentence_features(name, doc,
                                      sentence_relative=sentence_relative)[i]

    @classmethod
    @memoized
    def _transformed_feature(cls, name):
        """Get a function computing feature `name`, transformed as defined by
        :attr:`__features__`.

        Some features have a very skewed distribution (e.g. exponential, where
        a few words are valued orders of magnitude more than the vast majority
        of words), so we use their log-transformed values in the analysis to
        make them comparable to more regular features. The :attr:`__features__`
        attribute (which appears in the source code but not in the web version
        of these docs) defines which features are transformed how. Given a
        feature `name`, this method will generate a function that proxies calls
        to the raw feature method, and transforms the value if necessary.

        This method is :func:`~.utils.memoized` for speed, since other methods
        call it all the time.

        Parameters
        ----------
        name : str
            Name of the feature for which to create a function, without
            preceding underscore; for instance, call
            `cls._transformed_feature('aoa')` to get a function that uses the
            :meth:`_aoa` class method.

        Returns
        -------
        feature : function
            The feature function, with signature `feature(word=None)`. Call
            `feature()` to get the set of words encoded by that feature. Call
            `feature(word)` to get the transformed feature value of `word` (or
            `np.nan` if `word` is not coded by that feature).

        """

        if name not in cls.__features__:
            raise ValueError("Unknown feature: '{}'".format(name))
        _feature = getattr(cls, '_' + name)
        _, transform = cls.__features__[name]

        def feature(word=None):
            if word is None:
                return _feature()
            else:
                return transform(_feature(word))

        # Set the right docstring and name on the transformed feature function.
        functools.update_wrapper(feature, _feature)
        if transform is np.log:
            feature.__name__ = '_log' + feature.__name__
            feature.__doc__ = 'log(' + feature.__doc__ + ')'

        return feature

    @classmethod
    @memoized
    def _letters_count(cls, word=None):
        """#letters"""
        if word is None:
            # Return the keys of the frequency data
            return _get_zipf_frequency().keys()
        return len(word)

    @classmethod
    @memoized
    def _aoa(cls, word=None):
        """age of acquisition"""
        aoa = _get_aoa()
        if word is None:
            return aoa.keys()
        return aoa.get(word, np.nan)

    @classmethod
    @memoized
    def _zipf_frequency(cls, word=None):
        """zipf frequency"""
        frequency = _get_zipf_frequency()
        if word is None:
            return frequency.keys()
        return frequency.get(word, np.nan)

    @classmethod
    @memoized
    def _orthographic_density(cls, word=None):
        """orthographic nd"""
        clearpond_orthographic = _get_clearpond()['orthographic']
        if word is None:
            return clearpond_orthographic.keys()
        return clearpond_orthographic.get(word, np.nan) or np.nan
