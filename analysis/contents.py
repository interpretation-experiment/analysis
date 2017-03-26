from warnings import warn

from nltk.corpus import stopwords as nltk_stopwords
from nltk.metrics import jaccard_distance, edit_distance
from nltk.stem.snowball import EnglishStemmer as SnowballStemmer
from nltk.tokenize import word_tokenize as nltk_word_tokenize
import numpy as np

from .utils import memoized


# TODO: fix docs


# TODO: use spacy
def equip_sentence_content_words(models):
    """Define sentence `content_words`."""

    def filter_backtick_apostrophe(text):
        return text.replace('`', "'")

    def filter_lowercase(words):
        return map(lambda w: w.lower(), words)

    def _filter(words, exclude_list):
        return filter(lambda w: w not in exclude_list, words)

    def filter_punctuation(words):
        return _filter(words, [',', '.', ';', '!', '?'])

    def filter_length(words):
        return filter(lambda w: len(w) > 2, words)

    stopwords = set(nltk_stopwords.words('english'))
    # Missing from the corpus, and appears with tokenization
    stopwords.add("n't")

    def filter_stopwords(words):
        return _filter(words, stopwords)

    stemmer = SnowballStemmer(ignore_stopwords=True)

    def filter_stem(words):
        return map(lambda w: stemmer.stem(w), words)

    filters = [filter_backtick_apostrophe,
               nltk_word_tokenize,
               filter_lowercase,
               filter_punctuation,
               filter_length,
               filter_stopwords,
               filter_stem]

    @memoized
    def get_content_words(self):
        """Get content words of this sentence.

        This is done with the following steps:
        * tokenize the sentence text
        * set tokens to lowercase
        * remove punctuation
        * remove words $\leq$ 2 characters
        * remove stopwords
        * stem

        """

        processed = self.text
        for f in filters:
            processed = f(processed)
        return list(processed)

    models.Sentence.content_words = property(get_content_words)


def equip_sentence_distances(models):
    """Define distances between sentences.

    Distances defined:
    * `raw_distance`
    * `ordered_content_distance`
    * `unordered_content_distance`
    * `cum_root_distance`: cumulative distance from root, for any of the above
      distances

    Also add `Sentence.DISTANCE_TYPES` that lists available distances.

    """

    @memoized
    def raw_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance between `self.text` and
        `sentence.text`."""

        self_text = self.text
        sentence_text = sentence.text
        distance = edit_distance(self_text, sentence_text)
        norm = max(len(self_text), len(sentence_text))
        return distance / norm if normalized else distance

    @memoized
    def ordered_content_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on (ordered) content words
        between `self` and `sentence`."""

        self_content_words = self.content_words
        sentence_content_words = sentence.content_words
        distance = edit_distance(self_content_words, sentence_content_words)
        norm = max(len(self_content_words), len(sentence_content_words))
        return distance / norm if normalized else distance

    @memoized
    def unordered_content_distance(self, sentence):
        """Jaccard distance on (unordered) content words between `self` and
        `sentence`."""
        return jaccard_distance(set(self.content_words),
                                set(sentence.content_words))

    models.Sentence.raw_distance = raw_distance
    models.Sentence.ordered_content_distance = ordered_content_distance
    models.Sentence.unordered_content_distance = unordered_content_distance
    models.Sentence.DISTANCE_TYPES = ['raw', 'ordered_content',
                                      'unordered_content']

    # Add cumulative distance from root
    @memoized
    def cum_root_distance(self, distance_type, normalized=True):
        """Cumulative distance from root, for the distance `distance_type`."""

        if distance_type not in models.Sentence.DISTANCE_TYPES:
            raise ValueError("Unkown distance type (not one of {}): {}".format(
                    distance_type, models.Sentence.DISTANCE_TYPES))
        distance_name = distance_type + '_distance'
        if distance_type == 'unordered_content':
            kwargs = {}
            if not normalized:
                warn("'unordered_content' distance is always normalized, so "
                     "we're ignoring normalized=False")
        else:
            kwargs = {'normalized': normalized}

        if self.parent is None:
            return 0
        else:
            return (getattr(self, distance_name)(self.parent, **kwargs) +
                    self.parent.cum_root_distance(distance_type, **kwargs))

    models.Sentence.cum_root_distance = cum_root_distance


def equip_profile_transformation_rate(models):
    """Define `Profile.transformation_rate`; requires `content_words` and
    `spam`."""

    @memoized
    def transformation_rate(self, distance_type, with_spam=False):
        """Compute profile transformation for distance `distance_type`.

        The transformation rate is the average of distances between a sentence
        a profile transformed and its parent. It can be computed for all
        available distance types (defined on `Sentence`).

        Unless you provide `with_spam=True`, it is computed only on non-spam
        sentences. Note that we always keep rogue sentences (i.e. non-spam
        transformations that profiles made, but that are rejected in the trees
        because they initiated a branch that died early): they are valid
        transformations that the profiles made, just not included in the final
        trees because of branching.

        """

        if distance_type not in models.Sentence.DISTANCE_TYPES:
            raise ValueError("Unkown distance type (not one of {}): {}".format(
                    distance_type, models.Sentence.DISTANCE_TYPES))
        distance_name = distance_type + '_distance'

        sentences = (self.sentences.nondoublepost if with_spam
                     else self.sentences.nondoublepost.nonspam)
        transformed_sentences = sentences.filter(parent__isnull=False).all()
        if len(transformed_sentences) == 0:
            raise ValueError("Profile has no reformulated sentences "
                             "(with_spam={})".format(with_spam))

        return np.array([getattr(s.parent, distance_name)(s)
                         for s in transformed_sentences]).mean()

    models.Profile.transformation_rate = transformation_rate
