from warnings import warn

from nltk.metrics import jaccard_distance, edit_distance
import numpy as np

from .utils import memoized, get_nlp


@memoized
def is_stopword(tok):
    nlp = get_nlp()
    return (tok.lower_ in nlp.Defaults.stop_words
            or tok.lemma_ in nlp.Defaults.stop_words)


@memoized
def doc_tokens(doc):
    tokens = [tok for tok in doc
              if not tok.is_punct and not tok.is_space]
    return tuple(tokens)


def equip_sentence_words(models):
    """Define sentence content-related word accessors.

    Properties defined:
    * `tokens`: sentence's spaCy `Token`s, without punctuation or space
    * `words`: lowercase spelling of `tokens`
    * `content_tokens`: `tokens` that are not stopwords
    * `content_words`: lowercase spelling of `content_tokens`
    * `content_lemmas`: lowercase lemmas of `content_tokens`
    * `content_ids`: indexes of `content_tokens` in `tokens`

    """

    nlp = get_nlp()

    @memoized
    def get_tokens(self):
        return tuple(doc_tokens(nlp(self.text)))

    def get_words(self):
        return tuple(tok.lower_ for tok in self.tokens)

    @memoized
    def _get_content_ids_tokens(self):
        return tuple((i, tok) for i, tok in enumerate(self.tokens)
                     if not is_stopword(tok))

    def get_content_tokens(self):
        return tuple(tok for _, tok in _get_content_ids_tokens(self))

    def get_content_words(self):
        return tuple(tok.lower_ for _, tok in _get_content_ids_tokens(self))

    def get_content_lemmas(self):
        return tuple(tok.lemma_ for _, tok in _get_content_ids_tokens(self))

    def get_content_ids(self):
        return tuple(i for i, _ in _get_content_ids_tokens(self))

    models.Sentence.tokens = property(get_tokens)
    models.Sentence.words = property(get_words)
    models.Sentence.content_tokens = property(get_content_tokens)
    models.Sentence.content_words = property(get_content_words)
    models.Sentence.content_lemmas = property(get_content_lemmas)
    models.Sentence.content_ids = property(get_content_ids)


def equip_sentence_distances(models):
    """Define distances between sentences.

    Distances defined:
    * `raw_distance`: distance on raw sentence text
    * `ow_distance`: ordered distance on all words
    * `oc_distance`: ordered distance on content lemmas
    * `uc_distance`: unordered distance on content lemmas
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
    def ow_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on all (ordered) words between
        `self` and `sentence`."""

        self_words = self.words
        sentence_words = sentence.words
        distance = edit_distance(self_words, sentence_words)
        norm = max(len(self_words), len(sentence_words))
        return distance / norm if normalized else distance

    @memoized
    def ncow_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on all (ordered) words between
        `self` and `sentence`, modulo sentence cropping."""

        self_words = self.words
        sentence_words = sentence.words
        length_diff = np.abs(len(self_words) - len(sentence_words))
        distance = edit_distance(self_words, sentence_words) - length_diff
        norm = min(len(self_words), len(sentence_words))
        return distance / norm if normalized else distance

    @memoized
    def oc_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on (ordered) content lemmas between
        `self` and `sentence`."""

        self_content_lemmas = self.content_lemmas
        sentence_content_lemmas = sentence.content_lemmas
        distance = edit_distance(self_content_lemmas, sentence_content_lemmas)
        norm = max(len(self_content_lemmas), len(sentence_content_lemmas))
        return distance / norm if normalized else distance

    @memoized
    def ncoc_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on (ordered) content lemmas between
        `self` and `sentence`, modulo sentence cropping."""

        self_content_lemmas = self.content_lemmas
        sentence_content_lemmas = sentence.content_lemmas
        length_diff = np.abs(len(self_content_lemmas) - len(sentence_content_lemmas))
        distance = (edit_distance(self_content_lemmas, sentence_content_lemmas)
        	    - length_diff)
        norm = min(len(self_content_lemmas), len(sentence_content_lemmas))
        return distance / norm if normalized else distance

    @memoized
    def uc_distance(self, sentence):
        """Jaccard distance on (unordered) content lemmas between `self` and
        `sentence`."""
        return jaccard_distance(set(self.content_lemmas),
                                set(sentence.content_lemmas))

    models.Sentence.raw_distance = raw_distance
    models.Sentence.ow_distance = ow_distance
    models.Sentence.ncow_distance = ncow_distance
    models.Sentence.oc_distance = oc_distance
    models.Sentence.ncoc_distance = ncoc_distance
    models.Sentence.uc_distance = uc_distance
    models.Sentence.DISTANCE_TYPES = ['raw', 'ow', 'ncow', 'oc', 'ncoc', 'uc']

    # Add cumulative distance from root
    @memoized
    def cum_root_distance(self, distance_type, normalized=True):
        """Cumulative distance from root, for the distance `distance_type`."""

        if distance_type not in models.Sentence.DISTANCE_TYPES:
            raise ValueError("Unkown distance type (not one of {}): {}".format(
                    distance_type, models.Sentence.DISTANCE_TYPES))
        distance_name = distance_type + '_distance'
        if distance_type == 'uc':
            kwargs = {}
            if not normalized:
                warn('unordered content distance is always normalized, so '
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
    """Define `Profile.transformation_rate`; requires sentence distances and
    shaping codings."""

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
        trees because of branching. We also always drop doubleposts.

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
