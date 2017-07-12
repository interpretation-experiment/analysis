"""Features for words.

This module defines the :class:`Feature` which gives access to feature values
and related computed values (e.g. sentence-relative feature values).

A few other utility functions that load data for the features are also defined.

"""


import itertools
import logging
from csv import DictReader, reader as csvreader
import warnings
import functools

import numpy as np
from nltk.corpus import cmudict, wordnet
import spacy

from .contents import doc_tokens
from .utils import memoized, unpickle
from . import settings


logger = logging.getLogger(__name__)


def equip_sentence_features(models):
    """Redefine the sentence class to add the :class:`Feature` mixin to it."""
    models.Sentence.__bases__ = (Features,) + models.Sentence.__bases__


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
                # A few rows have empty data. Skip them.
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


@memoized
def _depth_under(tok):
    """Depth of the sentence dependency tree under `tok`."""

    children_depths = [_depth_under(child) + (child.dep != spacy.symbols.punct)
                       for child in tok.children]
    return 0 if len(children_depths) == 0 else np.max(children_depths)


@memoized
def _depth_above(tok):
    """Depth of `tok` in its sentence's dependency tree."""

    return (0 if tok.head == tok
            else ((tok.dep != spacy.symbols.punct) + _depth_above(tok.head)))


@memoized
def _depth_prop(tok):
    """Depth of `tok` compared to the maximun depth of the sentence `tok` is
    in.

    Note that the document `tok` comes from can span several sentences, but
    we're only returning the depth of `tok` as a fraction of the full depth of
    the particular sentence `tok` is in.

    """

    # Find the head of the sentence this token is in
    # (careful that tok.doc could span several sentences,
    # so we need our specific head)
    head = tok
    while head.head != head:
        head = head.head
    depth_under_head = _depth_under(head)
    return (_depth_above(tok) / depth_under_head
            if depth_under_head != 0 else np.nan)


@memoized
def _depth_subtree_prop(tok):
    """Depth of `tok` compared to the maximun depth of tokens under it."""
    num = _depth_above(tok)
    denom = (_depth_above(tok) + _depth_under(tok))
    if denom == 0:
        # This implies num == 0, so return 0.0
        return 0.0
    else:
        return num / denom


class PoolError(Exception):
    """Raised when a coding pool is requested for a feature that doesn't
    have any."""

    pass


def _identity(arg):
    return arg


class Features:

    """Feature loading and computing.

    Methods in this class fall into 3 categories:

    * Raw feature methods: they are :func:`~.utils.memoized` class methods of
      the form `cls._feature_name(cls, target=None)`. In general, calling them
      with `target=(tokens, tok, i)`, where `i` is the position of `tok` in
      `tokens`, will give you `tok`'s feature. For some feature, you can call
      directly with a `target=word`. In both those cases, `np.nan` is returned
      if the target is not coded.  For some features, calling with
      `target=None` returns the set of words encoded by that feature (which is
      used to compute e.g. averages over the pool of words encoded by that
      feature). Their docstring (which you will see below if you're reading
      this in a web browser) is the short name used to identify e.g. the
      feature's column in analyses in notebooks. These methods are used
      internally by the class, to provide the next category of methods.
    * Useful feature methods that can be used in analyses: :meth:`feature`,
      :meth:`features`. These methods use the raw feature methods (previous
      category) and the utility methods (next category) to compute feature
      values (possibly relative to sentence).
    * Private utility methods: :meth:`_transformed_word_feature`. These methods
      are used by the previous category of methods.

    Read the source of the first category (raw features) to know exactly how an
    individual feature is computed. Read the docstrings (and source) of the
    second category (useful methods for analyses) to learn how to use this
    class in analyses. Read the docstrings (and source) of the third category
    (private utility methods) to learn how the whole class assembles its
    different parts together.

    """

    #: Association of available word features to `transform` operation,
    #: defining how a feature value is transformed (for now either identity or
    #: log).
    WORD_FEATURES = {
        # feature_name:           transform
        'letters_count':          _identity,
        'aoa':                    _identity,
        'zipf_frequency':         _identity,
        'orthographic_density':   np.log,
        '1_gram_word':            _identity,
        '2_gram_word':            _identity,
        '3_gram_word':            _identity,
        '1_gram_tag':             _identity,
        '2_gram_tag':             _identity,
        '3_gram_tag':             _identity,
        'depth_under':            _identity,
        'depth_above':            _identity,
        'depth_prop':             _identity,
        'depth_subtree_prop':     _identity,
        'sentence_prop':          _identity,
    }

    #: List of categorical features defined on words.
    CATEGORICAL_WORD_FEATURES = {
        'pos',
        'dep',
    }

    #: List of features defined on sentences.
    SENTENCE_FEATURES = {
        'relatedness',
        'dep_depth',
    }

    _SW_INCLUDE = 'include'
    _SW_NAN = 'nan'
    _SW_EXCLUDE = 'exclude'

    @memoized
    def feature(self, name, stopwords=_SW_INCLUDE):
        """Compute feature `name` for this sentence.

        Parameters
        ----------
        name : str
            Name of the sentence feature to compute; can be a feature defined
            directly on the sentence (one of :attr:`SENTENCE_FEATURES`), or
            defined on the sentence's words and averaged (one of
            :attr:`WORD_FEATURES`).
        stopwords : {'include', 'exclude'}, optional
            Whether or not to include stopwords in the computation of the
            feature; note that some features (e.g. "dep depth") ignore this
            setting.

        Returns
        -------
        scalar
            Feature value.

        """

        # Check arguments
        assert name in (set(self.WORD_FEATURES.keys())
                        .union(self.SENTENCE_FEATURES)), name
        from_words = name in self.WORD_FEATURES
        # self._SW_NAN is redundant with self._SW_EXCLUDE here, so we don't
        # allow it
        assert stopwords in [self._SW_INCLUDE, self._SW_EXCLUDE], stopwords

        if from_words:
            return np.nanmean(self.features(name, stopwords=stopwords))

        tokens = (self.content_tokens
                  if stopwords == self._SW_EXCLUDE else self.tokens)
        feature = getattr(self, '_' + name)
        return feature(tokens)

    @memoized
    def token_average(self, position, name, rel=None, restrict_synonyms=False,
                      stopwords=_SW_INCLUDE):
        """Compute an average feature value for the position a token occupies
        in the sentence.

        Use this method to compute null-values for a feature for a given
        position in the sentence.

        Parameters
        ----------
        position : int
            Index of the token for which to compute an average feature value;
            the meaning of this value depends on the `stopwords` argument.
        name : str
            Feature name.
        rel : str, optional
            If not `None` (which is the default), return average relative to
            values of the sentence (with or without stopwords depending on
            `stopwords`) aggregated by this function; must be a name for which
            `np.nan<rel>` exists.
        restrict_synonyms : bool, optional
            If `True`, the average returned is the average value of synonyms of
            the token at position `position`; if `False` (the default), the
            average is computed over all words coded in the feature's coding
            pool (raising `PoolError` if there is none).
        stopwords : {'include', 'nan', 'exclude'}, optional
            How stopwords should be treated in computing features on the
            sentence. 'include' keeps the stopwords; 'nan' replaces them with
            `np.nan`, meaning they're not included in making the average
            relative to the sentence, but `position` is still relative to all
            words in the sentence (if `position` falls on a stopword the method
            returns `np.nan`); 'exclude' removes the stopwords, such that
            `position` is relative to content words only, and making the
            average relative to the sentence is the same as for
            `stopwords=nan`.

        Returns
        -------
        avg : scalar
            The average feature value as defined by the parameters.

        """

        # Check arguments
        assert name in set(self.WORD_FEATURES.keys()), name
        assert rel is None or hasattr(np, 'nan' + rel), rel
        assert (stopwords in
                [self._SW_INCLUDE, self._SW_NAN, self._SW_EXCLUDE]), stopwords

        if not restrict_synonyms:
            # If we average on the whole pool, we want to fail for a missing
            # pool before returning nan for a nan-ed stopword. This will raise
            # a PoolError if the feature has no coding pool.
            feature = self._transformed_word_feature(name)
            coded_words = feature()

        if stopwords == self._SW_NAN and position not in self.content_ids:
            # `position` is a stopword, which we convert to np.nan.
            return np.nan

        if restrict_synonyms:
            target_tok = (self.content_tokens
                          if stopwords == self._SW_EXCLUDE
                          else self.tokens)[position]
            # Compute the feature value for all synonyms of the target_tok
            values_to_avg = []
            for syn in self._strict_synonyms(target_tok.lemma_):
                # Reconstitute a sentence to compute features on (this is
                # necessary as some features are context-sensitive)
                syn_text = (self.text[:target_tok.idx] + syn
                            + self.text[target_tok.idx + len(target_tok):])
                syn_sentence = self.__class__(id=np.nan, text=syn_text)
                # Never compute sentence-relative, as we take care of
                # that below. Also, deal with stopwords ourselves, as the
                # reconstituted sentence could infer different stopwords from
                # our sentence's.
                syn_sentence_values = syn_sentence.features(name)
                if stopwords == self._SW_EXCLUDE:
                    syn_value = syn_sentence_values[self.content_ids[position]]
                elif (stopwords == self._SW_NAN and
                      position not in self.content_ids):
                    # This should never happen as we test for above
                    syn_value = np.nan
                else:
                    # stopwords is _SW_INCLUDE
                    syn_value = syn_sentence_values[position]
                values_to_avg.append(syn_value)
        else:
            # Average the feature over all coded words.
            values_to_avg = [feature(word) for word in coded_words]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            avg = np.nanmean(values_to_avg)

        if rel is not None:
            pool = getattr(np, 'nan' + rel)

            # Get sentence feature values, never relative to sentence
            # as we take care of that below
            values = self.features(name, stopwords=stopwords).copy()
            values[position] = avg

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                avg -= pool(values)

        return avg

    @memoized
    def features(self, name, rel=None, stopwords=_SW_INCLUDE):
        """Compute the feature values for all words in this sentence, possibly
        sentence-relative, with or without stopwords.

        Feature values are transformed as explained in
        :meth:`_transformed_word_feature`.

        If `rel` is not `None`, it indicates a NumPy function used to aggregate
        word features in the sentence; this method then returns the feature
        values minus the corresponding aggregate value. For instance, if
        `rel='median'`, this method returns the feature values minus the median
        of the sentence (words valued at `np.nan` are ignored). If `name`
        indicates a categorical feature, `rel` must be `None` (else an
        exception is raised).

        The method is :func:`~.utils.memoized` since it is called so often.

        Parameters
        ----------
        name : str
            Name of the word feature to compute; must be one of
            :attr:`WORD_FEATURES` or :attr:`CATEGORICAL_WORD_FEATURES`.
        rel : str, optional
            If not `None` (which is the default), return features relative to
            values of the sentence (with or without stopwords depending on
            `stopwords`) aggregated by this function; must be a name for which
            `np.nan<rel>` exists.
        stopwords : {'include', 'nan', 'exclude'}, optional
            'include' keeps the stopwords; 'nan' replaces them with `np.nan`
            values; 'exclude' removes their values from the final array.

        Returns
        -------
        features : array of float
            Array of feature values (possibly sentence-relative) for each word
            in the sentence (with or without stopwords). Non-coded words appear
            as `np.nan`.

        """

        # Check arguments
        assert name in (set(self.WORD_FEATURES.keys())
                        .union(self.CATEGORICAL_WORD_FEATURES)), name
        categorical = name in self.CATEGORICAL_WORD_FEATURES
        assert rel is None or hasattr(np, 'nan' + rel), rel
        assert (stopwords in
                [self._SW_INCLUDE, self._SW_NAN, self._SW_EXCLUDE]), stopwords

        # Compute the features
        tokens = (self.content_tokens
                  if stopwords == self._SW_EXCLUDE else self.tokens)
        feature = (self._transformed_word_feature(name)
                   if not categorical else getattr(self, '_' + name))
        values = np.array([feature((tokens, tok, i))
                           for i, tok in enumerate(tokens)],
                          dtype=np.float_)
        if stopwords == self._SW_NAN:
            values = np.array([v if i in self.content_ids else np.nan
                               for i, v in enumerate(values)])

        if rel is not None:
            assert not categorical, ("Categorical features can't "
                                     "be sentence-relative")
            pool = getattr(np, 'nan' + rel)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                values -= pool(values)

        return values

    @classmethod
    @memoized
    def _transformed_word_feature(cls, name):
        """Get a function computing word feature `name`, transformed as defined
        by :attr:`WORD_FEATURES`.

        Some features have a very skewed distribution (e.g. exponential, where
        a few words are valued orders of magnitude more than the vast majority
        of words), so we use their log-transformed values in the analysis to
        make them comparable to more regular features. :attr:`WORD_FEATURES`
        defines which features are transformed how
        (:attr:`CATEGORICAL_WORD_FEATURES` cannot be transformed). Given a
        feature `name`, this method will generate a function that proxies calls
        to the raw feature method, and transforms the value if necessary.

        This method is :func:`~.utils.memoized` for speed, since other methods
        call it all the time.

        Parameters
        ----------
        name : str
            Name of the feature for which to create a function, without
            preceding underscore; for instance, call
            `cls._transformed_word_feature('aoa')` to get a function that uses
            the :meth:`_aoa` class method; must be one of
            :attr:`WORD_FEATURES`.

        Returns
        -------
        feature : function
            The feature function, with signature `feature(target=None)`. For
            some features, call `feature()` to get the set of words encoded by
            that feature. In general, call `feature((tokens, tok, i))`, where
            `i` is the index of `tok` in `tokens`, to get the transformed
            feature value of `tok`. For some features (those that don't depend
            on the sentence), you can call directly `feature(word)` to get the
            transformed feature value of `word`. In all cases, `np.nan` is
            returned if `word` or `tok` is not coded by that feature.

        """

        assert name in cls.WORD_FEATURES, name
        _feature = getattr(cls, '_' + name)
        transform = cls.WORD_FEATURES[name]

        def feature(target=None):
            if target is None:
                return _feature()
            else:
                return transform(_feature(target))

        # Set the right docstring and name on the transformed feature function.
        functools.update_wrapper(feature, _feature)
        if transform is np.log:
            feature.__name__ = '_log' + feature.__name__
            feature.__doc__ = 'log(' + feature.__doc__ + ')'

        return feature

    @classmethod
    def _strict_synonyms(cls, word, compounds=False):
        """Get the set of synonyms of `word` through WordNet, excluding `word`
        itself; empty if nothing is found.

        If `compounds=True`, also include compound synonyms.

        """

        # wordnet.synsets() lemmatizes words, so we might as well control it.
        # This also lets us check the lemma is present in the generated
        # synonym list further down.
        lemma = wordnet.morphy(word)
        if lemma is None:
            return set()

        # Skip multi-word synonyms
        synonyms = set(word.lower() for synset in wordnet.synsets(lemma)
                       for word in synset.lemma_names()
                       if compounds or ('_' not in word and '-' not in word))
        if len(synonyms) > 0:
            assert lemma in synonyms
            synonyms.remove(lemma)
        return synonyms

    #
    # WORD FEATURES
    #

    @classmethod
    @memoized
    def _letters_count(cls, target=None):
        """#letters"""
        if target is None:
            # Return the keys of the frequency data
            return _get_zipf_frequency().keys()
        if isinstance(target, str):
            word = target
        else:
            _, tok, _ = target
            word = tok.lower_
        return len(word)

    @classmethod
    @memoized
    def _aoa(cls, target=None):
        """age of acquisition"""
        aoa = _get_aoa()
        if target is None:
            return aoa.keys()
        if isinstance(target, str):
            word = target
        else:
            _, tok, _ = target
            word = tok.lower_
        return aoa.get(word, np.nan)

    @classmethod
    @memoized
    def _zipf_frequency(cls, target=None):
        """zipf frequency"""
        frequency = _get_zipf_frequency()
        if target is None:
            return frequency.keys()
        if isinstance(target, str):
            word = target
        else:
            _, tok, _ = target
            word = tok.lower_
        return frequency.get(word, np.nan)

    @classmethod
    @memoized
    def _orthographic_density(cls, target=None):
        """orthographic nd"""
        clearpond_orthographic = _get_clearpond()['orthographic']
        if target is None:
            return clearpond_orthographic.keys()
        if isinstance(target, str):
            word = target
        else:
            _, tok, _ = target
            word = tok.lemma_
        return clearpond_orthographic.get(word, np.nan) or np.nan

    @classmethod
    @memoized
    def _ngram_logprob(cls, model_n, model_type, target):
        assert model_n in (1, 2, 3), model_n
        assert model_type in ('word', 'tag'), model_type
        tags = (model_type == 'tag')
        model = unpickle(settings.MODEL_TEMPLATE.format(n=model_n,
                                                        type=model_type))
        if target is None:
            raise PoolError("No coding pool for ngrams probs")
        _, target_tok, target_position = target

        # Find the position of target_tok in doc_tokens, which will be the list
        # of tokens we use (i.e., never stopword-filtered, since the models are
        # trained with all words).
        tokens = doc_tokens(target_tok.doc)
        position = tokens.index(target_tok)

        # Since most Gistr sentences actually are single sentences,
        # treat the doc as a single sentence
        # (vs. averaging the scores of multiple sentences)
        data = []
        for tok in tokens[:position + 1]:
            if tags:
                data.append(tok.pos_.upper())
            else:
                data.append(tok.lower_)

        data = list(model._lpad) + data
        data_position = len(model._lpad) + position
        context = tuple(data[data_position - model_n + 1:data_position])
        final_tok = data[data_position]
        assert final_tok == (target_tok.pos_.upper() if tags
                             else target_tok.lower_), (target_tok, final_tok)
        return (- model.logprob(final_tok, context))

    @classmethod
    def _1_gram_word(cls, target=None):
        """1-gram word logprob"""
        return cls._ngram_logprob(1, 'word', target)

    @classmethod
    def _2_gram_word(cls, target=None):
        """2-gram word logprob"""
        return cls._ngram_logprob(2, 'word', target)

    @classmethod
    def _3_gram_word(cls, target=None):
        """3-gram word logprob"""
        return cls._ngram_logprob(3, 'word', target)

    @classmethod
    def _1_gram_tag(cls, target=None):
        """1-gram tag logprob"""
        return cls._ngram_logprob(1, 'tag', target)

    @classmethod
    def _2_gram_tag(cls, target=None):
        """2-gram tag logprob"""
        return cls._ngram_logprob(2, 'tag', target)

    @classmethod
    def _3_gram_tag(cls, target=None):
        """3-gram tag logprob"""
        return cls._ngram_logprob(3, 'tag', target)

    @classmethod
    def _depth_under(cls, target=None):
        """depth under"""
        if target is None:
            raise PoolError("No coding pool for depth_under")
        _, tok, _ = target
        return _depth_under(tok)

    @classmethod
    def _depth_above(cls, target=None):
        """depth above"""
        if target is None:
            raise PoolError("No coding pool for depth_above")
        _, tok, _ = target
        return _depth_above(tok)

    @classmethod
    def _depth_prop(cls, target=None):
        """depth %"""
        if target is None:
            raise PoolError("No coding pool for depth_prop")
        _, tok, _ = target
        return _depth_prop(tok)

    @classmethod
    def _depth_subtree_prop(cls, target=None):
        """depth subtree %"""
        if target is None:
            raise PoolError("No coding pool for depth_subtree_prop")
        _, tok, _ = target
        return _depth_subtree_prop(tok)

    @classmethod
    def _sentence_prop(cls, target=None):
        """sentence %"""
        if target is None:
            raise PoolError("No coding pool for sentence_prop")
        tokens, _, position = target
        return np.nan if len(tokens) <= 1 else position / (len(tokens) - 1)

    #
    # CATEGORICAL WORD FEATURES
    #

    @classmethod
    def _pos(cls, target=None):
        """POS"""
        if target is None:
            raise PoolError("No coding pool for POS")
        _, tok, _ = target
        return tok.pos

    @classmethod
    def _dep(cls, target=None):
        """dep"""
        if target is None:
            raise PoolError("No coding pool for dep")
        _, tok, _ = target
        return tok.dep

    #
    # SENTENCE FEATURES
    #

    @classmethod
    @memoized
    def _relatedness(cls, tokens):
        """relatedness"""
        tokens = [t for t in tokens if t.has_vector]
        return np.mean([t1.similarity(t2)
                        for t1, t2 in itertools.combinations(tokens, 2)])

    @classmethod
    @memoized
    def _dep_depth(cls, tokens):
        """dep depth"""
        # Recover the parent doc
        doc = tokens[0].doc
        # Average depth of all doc heads (can span several sentences)
        return np.mean([_depth_under(tok) for tok in doc if tok.head == tok])
