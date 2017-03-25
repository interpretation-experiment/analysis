# Adapted from http://www.nltk.org/_modules/nltk/model/ngram.html
# Original copyright header:
# ----------------------------------------------------------------------------
# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2014 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Daniel Blanchard <dblanchard@ets.org>
#          Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
# ----------------------------------------------------------------------------
from collections import defaultdict
from math import log
from warnings import warn

from nltk.probability import FreqDist, LidstoneProbDist
from nltk.util import ngrams


class NgramModel:
    """
    A processing interface for assigning a probability to the next word.
    """

    def __init__(self, n, train, pad_left=True, pad_right=False,
                 estimator=None, estimator_args=[], estimator_kwargs={}):
        """
        Create an ngram language model to capture patterns in n consecutive
        words of training text.  An estimator smooths the probabilities derived
        from the text and may allow generation of ngrams not seen during
        training.

            >>> from nltk.corpus import brown
            >>> from nltk.probability import LidstoneProbDist
            >>> est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
            >>> lm = NgramModel(3, brown.words(categories='news'),
            ...                 estimator=est)
            >>> lm
            <NgramModel with 91603 3-grams>
            >>> lm._backoff
            <NgramModel with 62888 2-grams>
            >>> lm.entropy(['The', 'Fulton', 'County', 'Grand', 'Jury', 'said',
            ... 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent',
            ... 'primary', 'election', 'produced', '``', 'no', 'evidence',
            ... "''", 'that', 'any', 'irregularities', 'took', 'place', '.'])
            ... # doctest: +ELLIPSIS
            0.5776...

        :param n: the order of the language model (ngram size)
        :type n: int
        :param train: the training text
        :type train: list(str) or list(list(str))
        :param pad_left: whether to pad the left of each sentence with an
            (n-1)-gram of empty strings
        :type pad_left: bool
        :param pad_right: whether to pad the right of each sentence with an
            (n-1)-gram of empty strings
        :type pad_right: bool
        :param estimator: a function for generating a probability distribution
        :type estimator: a function that takes a FreqDist and returns a
            ProbDist
        :param estimator_args: Extra arguments for estimator.
            These arguments are usually used to specify extra properties for
            the probability distributions of individual conditions, such as the
            number of bins they contain.
        :type estimator_args: list(any)
        :param estimator_kwargs: Extra keyword arguments for the estimator
        :type estimator_kwargs: dict(str, any)
        """

        # protection from cryptic behavior for calling programs
        # that use the pre-2.0.2 interface
        assert(isinstance(pad_left, bool))
        assert(isinstance(pad_right, bool))

        # make sure n is greater than zero, otherwise print it
        assert (n > 0), n

        # For explicitness save the check whether this is a unigram model
        self.is_unigram_model = (n == 1)
        # save the ngram order number
        self._n = n
        # save left and right padding
        self._lpad = ('',) * (n - 1) if pad_left else ()
        self._rpad = ('',) * (n - 1) if pad_right else ()

        # Save the estimator we use
        self._estimator = estimator or LidstoneProbDist
        self._estimator_args = estimator_args
        self._estimator_kwargs = estimator_kwargs

        cfd = FreqDist()

        # Save observed ngrams and context-word pairs
        self._ngrams = set()
        self._words_following = defaultdict(set)

        # If given a list of strings instead of a list of lists, create
        # enclosing list
        if (train is not None) and isinstance(train[0], str):
            train = [train]

        for sent in train:
            raw_ngrams = ngrams(sent, n, pad_left, pad_right,
                                left_pad_symbol='', right_pad_symbol='')
            for ngram in raw_ngrams:
                self._ngrams.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                cfd[(context, token)] += 1
                self._words_following[context].add(token)

        self._probdist = self._estimator(cfd, *estimator_args,
                                         **estimator_kwargs)

        # recursively construct the lower-order models
        if not self.is_unigram_model:
            self._backoff = NgramModel(n - 1, train,
                                       pad_left, pad_right,
                                       self._estimator,
                                       estimator_args,
                                       estimator_kwargs)

            self._backoff_alphas = dict()
            # For each condition (or context)
            for ctxt in self._words_following.keys():
                backoff_ctxt = ctxt[1:]
                backoff_total_pr = 0.0
                total_observed_pr = 0.0

                # this is the subset of words that we OBSERVED following
                # this context.
                # i.e. Count(word | context) > 0
                for word in self._words_following[ctxt]:
                    total_observed_pr += self.prob(word, ctxt)
                    # we also need the total (n-1)-gram probability of
                    # words observed in this n-gram context
                    backoff_total_pr += self._backoff.prob(word, backoff_ctxt)

                assert (0 <= total_observed_pr <= 1), total_observed_pr
                # beta is the remaining probability weight after we factor out
                # the probability of observed words.
                # As a sanity check, both total_observed_pr and
                # backoff_total_pr must be GE 0, since probabilities are never
                # negative
                beta = 1.0 - total_observed_pr

                # backoff total has to be less than one, otherwise we get
                # an error when we try subtracting it from 1 in the denominator
                assert (0 <= backoff_total_pr), backoff_total_pr
                if backoff_total_pr >= 1:
                    warn("Found a backoff_total_pr >= 1 ({}), maybe because "
                         "of numerical imprecision. Skipping this alpha."
                         .format(backoff_total_pr))
                    continue
                alpha_ctxt = beta / (1.0 - backoff_total_pr)

                self._backoff_alphas[ctxt] = alpha_ctxt

    def prob(self, word, context):
        """
        Evaluate the probability of this word in this context using Katz
        Backoff.

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        context = tuple(context)
        if (context + (word,) in self._ngrams) or (self.is_unigram_model):
            return self._probdist.prob((context, word))
        else:
            return self._alpha(context) * self._backoff.prob(word, context[1:])

    def _alpha(self, context):
        """Get the backoff alpha value for the given context
        """
        error_message = "Alphas and backoff are not defined for unigram models"
        assert not self.is_unigram_model, error_message

        if context in self._backoff_alphas:
            return self._backoff_alphas[context]
        else:
            return 1

    def logprob(self, word, context):
        """
        Evaluate the log probability of this word in this context (a negative
        value).

        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """
        return log(self.prob(word, context), 2)

    @property
    def ngrams(self):
        return self._ngrams

    @property
    def backoff(self):
        return self._backoff

    @property
    def probdist(self):
        return self._probdist

    def choose_random_word(self, context):
        '''
        Randomly select a word that is likely to appear in this context.

        :param context: the context the word is in
        :type context: list(str)
        '''

        return self.generate(1, context)[-1]

    # NB, this will always start with same word if the model
    # was trained on a single text

    def generate(self, num_words, context=()):
        '''
        Generate random text based on the language model.

        :param num_words: number of words to generate
        :type num_words: int
        :param context: initial words in generated string
        :type context: list(str)
        '''

        text = list(context)
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text

    def _generate_one(self, context):
        if self._n == 1:
            context = ()
        else:
            context = (self._lpad + tuple(context))[- self._n + 1:]
        if context in self:
            return self[context].generate()
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return '.'

    def entropy(self, text):
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given evaluation text.
        This is the absolute value of the average log probability of each word
        in the text.

        :param text: words to use for evaluation
        :type text: list(str)
        """

        e = 0.0
        text = list(self._lpad) + text + list(self._rpad)
        for i in range(self._n - 1, len(text)):
            context = tuple(text[i - self._n + 1:i])
            token = text[i]
            e += (- self.logprob(token, context))
        return e / float(len(text) - (self._n - 1))

    def perplexity(self, text):
        """
        Calculates the perplexity of the given text.
        This is simply 2 ** cross-entropy for the text.

        :param text: words to calculate perplexity of
        :type text: list(str)
        """

        return pow(2.0, self.entropy(text))

    def __contains__(self, item):
        return tuple(item) in self._words_following

    def __getitem__(self, item):
        context = tuple(item)
        cfd = self._probdist.freqdist()

        counts = {}
        for word in self._words_following[context]:
            counts[word] = cfd[(context, word)]

        return self._estimator(FreqDist(counts), *self._estimator_args,
                               **self._estimator_kwargs)

    def __repr__(self):
        return '<NgramModel with %d %d-grams>' % (len(self._ngrams), self._n)
