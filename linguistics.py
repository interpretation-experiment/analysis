import csv
import os
from warnings import warn

from django.db.models.manager import Manager
from nltk.corpus import stopwords as nltk_stopwords
from nltk.metrics import jaccard_distance, edit_distance
from nltk.stem.snowball import EnglishStemmer as SnowballStemmer
from nltk.tokenize import word_tokenize as nltk_word_tokenize
import numpy as np

from utils import memoized, import_spreadr_models


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

    # Test
    assert models.Sentence.objects.get(id=1).content_words == \
        ['young', 'boy', 'sudden', 'hit', 'littl', 'girl']
    assert models.Sentence.objects.get(id=2).content_words == \
        ['forget', 'leav', 'door', 'open', 'leav', 'offic']


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

    # Testing this is hard (we don't have predictable data for it),
    # so we mostly test for stupid values only
    assert models.Sentence.objects.get(id=1).raw_distance(
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).ordered_content_distance(
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).unordered_content_distance(
            models.Sentence.objects.get(id=1)) == 0.0
    assert np.abs(models.Sentence.objects.get(id=1).raw_distance(
            models.Sentence.objects.get(id=2)) - .754098) <= 1e-6
    assert models.Sentence.objects.get(id=1).ordered_content_distance(
            models.Sentence.objects.get(id=2)) == 1.0
    assert models.Sentence.objects.get(id=1).unordered_content_distance(
            models.Sentence.objects.get(id=2)) == 1.0

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

    # Also hard to test, so testing stupid values
    assert models.Sentence.objects.get(id=580).cum_root_distance('raw') == 0
    assert models.Sentence.objects.get(id=580).cum_root_distance(
        'raw', normalized=False) == 0
    assert models.Sentence.objects.get(id=580).cum_root_distance(
        'ordered_content') == 0
    assert models.Sentence.objects.get(id=580).cum_root_distance(
        'ordered_content', normalized=False) == 0
    assert models.Sentence.objects.get(id=580).cum_root_distance(
        'unordered_content') == 0
    assert models.Sentence.objects.get(id=580).cum_root_distance(
        'unordered_content', normalized=False) == 0
    assert models.Sentence.objects.get(id=823).cum_root_distance(
        'raw') == .02
    assert models.Sentence.objects.get(id=823).cum_root_distance(
        'raw', normalized=False) == 1
    assert models.Sentence.objects.get(id=823).cum_root_distance(
        'ordered_content') == 0
    assert models.Sentence.objects.get(id=823).cum_root_distance(
        'ordered_content', normalized=False) == 0
    assert models.Sentence.objects.get(id=823).cum_root_distance(
        'unordered_content') == 0
    assert models.Sentence.objects.get(id=823).cum_root_distance(
        'unordered_content', normalized=False) == 0
    assert abs(models.Sentence.objects.get(id=1115).cum_root_distance('raw') -
               .308333) < 1e-6
    assert models.Sentence.objects.get(id=1115).cum_root_distance(
        'raw', normalized=False) == 15
    assert abs(models.Sentence.objects.get(id=1115)
               .cum_root_distance('ordered_content') - .166666) < 1e-6
    assert models.Sentence.objects.get(id=1115).cum_root_distance(
        'ordered_content', normalized=False) == 1
    assert abs(models.Sentence.objects.get(id=1115)
               .cum_root_distance('unordered_content') - .166666) < 1e-6
    assert abs(models.Sentence.objects.get(id=1115)
               .cum_root_distance('unordered_content', normalized=False) -
               .166666) < 1e-6


def load_codings(db, coding, mapper):
    """Load the files for codings of `coding`, for the given `db`, extracting
    key `coding` from each csv file, and processing the values with
    `mapper`."""

    # Get the list of files to load
    folder = os.path.join(os.path.dirname(__file__), 'codings', db, coding)
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
    * `spam` (with `spam_detail`, and `ham` on `SentenceManager`)
    * `rogue`

    Also add Sentence.LOADED_CODINGS which lists loaded codings, and `kept`
    and `with_dropped()` on `SentenceManager` which filters on sentences kept
    after deciding on all codings.

    """

    # Get database name
    from django.conf import settings
    db = settings.DATABASES['default']['NAME']

    # Load spam codings
    spam_codings = load_codings(
        db, 'spam',
        lambda c: c.lower() == 'true' or c.lower() == 'yes' or c == '1')

    # Add them as properties to Sentence
    models.Sentence._spam_codings = spam_codings
    models.Sentence.spam_detail = property(memoized(
            lambda self: self._spam_codings.get(self.id, [(False, None)])))
    models.Sentence.self_spam = property(memoized(
            lambda self: np.mean([spam for (spam, _)
                                  in self.spam_detail]) > 0.5))
    models.Sentence.spam = property(memoized(
            lambda self: self.self_spam or (self.parent is not None and
                                            self.parent.spam)))

    # Give easy access to non-spam sentences
    @memoized
    def get_ham(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        qs = self.get_queryset()
        ids = [s.id for s in qs if not s.spam]
        return qs.filter(pk__in=ids)

    Manager.ham = property(get_ham)

    # Create rogue codings
    @memoized
    def get_rogue(self):
        if self.parent is None:
            return False

        ham_children = self.children.ham
        if ham_children.count() == 0:
            # Potential leaf
            head = self.head
            depth = self.depth
            competitors = [s for s in self.tree.sentences.ham.all()
                           if s.children.ham.count() == 0 and s.head == head]

            betters = [s for s in competitors if s.depth > depth]
            if len(betters) > 0:
                # Another leaf (or even several) in this branch is strictly
                # deeper
                return True

            equals = [s for s in competitors if s.depth == depth]
            if len(equals) > 0:
                # Another leaf (or even several) in this branch is as deep as
                # we are. Take the sentence created earliest as non-rogue.
                equals = sorted(equals, key=lambda s: s.created)
                return equals[0] != self

            # No better competitor, so we're not rogue
            return False
        else:
            # We're not a leaf, so see if our children are rogue
            return np.all([c.rogue for c in ham_children])

    models.Sentence.rogue = property(get_rogue)

    # Easily access sentences we want to keep
    @memoized
    def get_kept(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ham = self.ham
        ids = [s.id for s in ham if not s.rogue]
        return ham.filter(pk__in=ids)

    Manager.kept = property(get_kept)
    Manager.with_dropped = lambda self, with_dropped: \
        self.get_queryset() if with_dropped else self.kept

    models.Sentence.LOADED_CODINGS = ['spam', 'rogue']

    # Test spam (hard to test, so only checking that what we entered as first
    # sentences is not spam)
    assert len(models.Sentence.objects.get(id=1).spam_detail[0]) == 2
    assert not models.Sentence.objects.get(id=1).spam_detail[0][0]
    assert not models.Sentence.objects.get(id=1).spam
    assert len(models.Sentence.objects.get(id=2).spam_detail[0]) == 2
    assert not models.Sentence.objects.get(id=2).spam_detail[0][0]
    assert not models.Sentence.objects.get(id=2).spam
    assert len(models.Sentence.objects.get(id=3).spam_detail[0]) == 2
    assert not models.Sentence.objects.get(id=3).spam_detail[0][0]
    assert not models.Sentence.objects.get(id=3).spam

    # Test ham
    assert models.Sentence.objects.ham.get(id=1) is not None
    try:
        models.Profile.objects.ham
    except ValueError:
        pass  # Test passed
    else:
        raise Exception('ValueError not raised on Profile.objects.ham')

    # Test rogue
    assert models.Sentence.objects.get(id=486).rogue
    assert not models.Sentence.objects.get(id=489).rogue
    assert not models.Sentence.objects.get(id=2081).rogue
    assert models.Sentence.objects.get(id=2084).rogue

    # Test kept
    assert models.Sentence.objects.kept.get(id=1) is not None

    # Test with_dropped
    assert models.Sentence.objects.with_dropped(True).get(id=1) is not None
    assert models.Sentence.objects.with_dropped(False).get(id=1) is not None


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

        sentences = self.sentences if with_spam else self.sentences.ham
        transformed_sentences = sentences.filter(parent__isnull=False).all()
        if len(transformed_sentences) == 0:
            raise ValueError("Profile has no reformulated sentences "
                             "(with_spam={})".format(with_spam))

        return np.array([getattr(s.parent, distance_name)(s)
                         for s in transformed_sentences]).mean()

    models.Profile.transformation_rate = transformation_rate

    # Test
    try:
        models.Profile.objects.get(
            user__username='sl').transformation_rate('raw')
    except ValueError:
        pass  # Test passed
    else:
        raise Exception("Exception not raised on profile with "
                        "no reformulated sentences")
    # And with with_spam=True
    try:
        models.Profile.objects.get(
            user__username='sl').transformation_rate('raw', with_spam=True)
    except ValueError:
        pass  # Test passed
    else:
        raise Exception("Exception not raised on profile with "
                        "no reformulated sentences")


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
