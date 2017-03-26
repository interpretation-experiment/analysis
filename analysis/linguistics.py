import csv
import os
from warnings import warn

from nltk.corpus import stopwords as nltk_stopwords
from nltk.metrics import jaccard_distance, edit_distance
from nltk.stem.snowball import EnglishStemmer as SnowballStemmer
from nltk.tokenize import word_tokenize as nltk_word_tokenize
import numpy as np

from .utils import memoized, import_spreadr_models
from . import settings


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


def equip_sentence_codings(models):
    """Import external codings on sentences.

    Codings imported:
    * `spam` (with `spam_detail`, and `nonspam` on `SentenceManager`
      and `QuerySet`)
    * `doublepost` (with `nondoublepost` on `SentenceManager` and `QuerySet`)
    * `rogue` (with `nonrogue` on `SentenceManager` and `QuerySet`)

    Also add Sentence.LOADED_CODINGS which lists loaded codings, and `kept`
    and `with_dropped()` on `SentenceManager` and `QuerySet` which filters on
    sentences kept after deciding on all codings.

    """

    from django.db.models.manager import Manager
    from django.db.models.query import QuerySet

    # Get database name
    from django.conf import settings as django_settings
    db = django_settings.DATABASES['default']['NAME']

    # Load spam codings
    spam_codings = load_codings(
        db, 'spam',
        lambda c: c.lower() == 'true' or c.lower() == 'yes' or c == '1')

    # Add them as properties to Sentence
    models.Sentence._spam_codings = spam_codings
    models.Sentence.spam_detail = property(memoized(
        lambda self: self._spam_codings.get(self.id, [(False, None)])
    ))
    models.Sentence._self_spam = property(memoized(
        lambda self: np.mean([spam for (spam, _) in self.spam_detail]) > 0.5
    ))
    models.Sentence.spam = property(memoized(
        lambda self: (self._self_spam or
                      (self.parent is not None and self.parent.spam))
    ))

    # Give easy access to non-spam sentences
    @memoized
    def get_nonspam(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if not s.spam]
        return self.filter(pk__in=ids)

    Manager.nonspam = property(get_nonspam)
    QuerySet.nonspam = property(get_nonspam)

    # Create doublepost codings
    @memoized
    def get_self_doublepost(self):
        if self.parent is None:
            # Root sentences are not doubleposts
            return False

        if self.profile.user.is_staff:
            # Staff-made sentences are not self_doublepost (though they
            # can descend from a self_doublepost)
            return False

        # So we're from a non-staff profile. Are we a self_doublepost?
        others = self.tree.sentences\
            .filter(profile=self.profile)\
            .exclude(id=self.id)\
            .order_by('id')
        # If there is another sentence by this same (non-staff) profile,
        # with smaller id, we're self_doublepost.
        return (others.count() > 0 and others.first().id < self.id)

    models.Sentence._self_doublepost = property(get_self_doublepost)
    models.Sentence.doublepost = property(memoized(
        lambda self: (self._self_doublepost or
                      (self.parent is not None and self.parent.doublepost))
    ))

    # Give easy access to non-doublepost sentences
    @memoized
    def get_nondoublepost(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if not s.doublepost]
        return self.filter(pk__in=ids)

    Manager.nondoublepost = property(get_nondoublepost)
    QuerySet.nondoublepost = property(get_nondoublepost)

    # Create rogue codings
    @memoized
    def get_self_rogue(self):
        if self.parent is None:
            # Root sentences are not rogue
            return False

        if self.children.nonspam.nondoublepost.count() > 0:
            # We're not a leaf, so not self_rogue (though we could have
            # rogue children)
            return False

        # We're a potential leaf, do we compete with other leaves in this
        # branch?
        head = self.head
        depth = self.depth
        competitors = \
            [s for s in self.tree.sentences.nonspam.nondoublepost.all()
             if (s.children.nonspam.nondoublepost.count() == 0
                 and s.head == head)]

        betters = [s for s in competitors if s.depth > depth]
        if len(betters) > 0:
            # Another leaf (or even several) in this branch is strictly
            # deeper
            return True

        equals = [s for s in competitors if s.depth == depth]
        if len(equals) > 0:
            # Another leaf (or even several) in this branch is as deep as
            # we are. Take the sentence with smallest id as non-self_rogue.
            equals = sorted(equals, key=lambda s: s.id)
            return equals[0] != self

        # No better competitor, so we're not self_rogue
        return False

    models.Sentence._self_rogue = property(get_self_rogue)
    models.Sentence.rogue = property(memoized(
        lambda self: (self._self_rogue or
                      (self.children.nonspam.nondoublepost.count() > 0 and
                       np.all([c.rogue for c in
                               self.children.nonspam.nondoublepost.all()])))
    ))

    # Give easy access to non-rogue sentences
    @memoized
    def get_nonrogue(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if not s.rogue]
        return self.filter(pk__in=ids)

    Manager.nonrogue = property(get_nonrogue)
    QuerySet.nonrogue = property(get_nonrogue)

    # Easily access sentences we want to keep
    @memoized
    def get_kept(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        return self.nonspam.nondoublepost.nonrogue

    Manager.kept = property(get_kept)
    QuerySet.kept = property(get_kept)
    Manager.with_dropped = lambda self, with_dropped: \
        self if with_dropped else self.kept
    QuerySet.with_dropped = lambda self, with_dropped: \
        self if with_dropped else self.kept

    models.Sentence.LOADED_CODINGS = ['spam', 'doublepost', 'rogue']


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


def equip_spreadr_models():
    """Equip spreadr models with linguistics tools and codings.

    Tools:
    * Content words on `Sentence`s
    * Distances on `Sentence`s
    * Codings on `Sentence`s
    * Tranformation rate on `Profile`s

    """

    models = import_spreadr_models()
    equip_sentence_content_words(models)
    equip_sentence_distances(models)
    equip_sentence_codings(models)
    equip_profile_transformation_rate(models)
