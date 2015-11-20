import csv
import os

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
    stopwords.add("n't")  # Missing from the corpus, and appears with tokenization

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

    Also add `Sentence.DISTANCE_TYPES` that lists available distances.

    """

    @memoized
    def raw_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance between `self.text` and `sentence.text`."""

        self_text = self.text
        sentence_text = sentence.text
        distance = edit_distance(self_text, sentence_text)
        norm = max(len(self_text), len(sentence_text))
        return  distance / norm if normalized else distance

    @memoized
    def ordered_content_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on (ordered) content words
        between `self` and `sentence`."""

        self_content_words = self.content_words
        sentence_content_words = sentence.content_words
        distance = edit_distance(self_content_words, sentence_content_words)
        norm = max(len(self_content_words), len(sentence_content_words))
        return  distance / norm if normalized else distance

    @memoized
    def unordered_content_distance(self, sentence):
        """Jaccard distance on (unordered) content words between `self` and `sentence`."""
        return jaccard_distance(set(self.content_words), set(sentence.content_words))

    models.Sentence.raw_distance = raw_distance
    models.Sentence.ordered_content_distance = ordered_content_distance
    models.Sentence.unordered_content_distance = unordered_content_distance
    models.Sentence.DISTANCE_TYPES = ['raw', 'ordered_content', 'unordered_content']

    # Testing this is hard (we don't have predictable data for it),
    # so we mostly test for stupid values only
    assert models.Sentence.objects.get(id=1).raw_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).ordered_content_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).unordered_content_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert np.abs(models.Sentence.objects.get(id=1).raw_distance(\
            models.Sentence.objects.get(id=2)) - .754098) <= 1e-6
    assert models.Sentence.objects.get(id=1).ordered_content_distance(\
            models.Sentence.objects.get(id=2)) == 1.0
    assert models.Sentence.objects.get(id=1).unordered_content_distance(\
            models.Sentence.objects.get(id=2)) == 1.0


def load_codings(db, coding, mapper):
    """Load the files for codings of `coding`, for the given `db`, extracting key `coding`
    from each csv file, and processing the values with `mapper`."""

    # Get the list of files to load
    folder = os.path.join(os.path.dirname(__file__), 'codings', db, coding)
    filepaths = [os.path.join(folder, name) for name in next(os.walk(folder))[2]]

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
    * `spam` (with `spam_detail`)

    Also add Sentence.LOADED_CODINGS which lists loaded codings.

    """

    # Get database name
    from django.conf import settings
    db = settings.DATABASES['default']['NAME']

    # Load spam codings
    spam_codings = load_codings(\
            db, 'spam', lambda c: c.lower() == 'true' or c.lower() == 'yes' or c == '1')
    models.Sentence._spam_codings = spam_codings
    models.Sentence.spam_detail = property(memoized(\
            lambda self: self._spam_codings.get(self.id, [(False, None)])))
    models.Sentence.self_spam = property(memoized(\
            lambda self: np.mean([spam for (spam, _) in self.spam_detail]) > 0.5))
    models.Sentence.spam = property(memoized(\
            lambda self: self.self_spam or getattr(self.parent, 'self_spam', False)))

    models.Sentence.LOADED_CODINGS = ['spam']

    # Test (hard to test, so only checking that what we entered as first
    # sentences is not spam)
    assert len(models.Sentence.objects.get(id=1).spam_detail[0]) == 2
    assert models.Sentence.objects.get(id=1).spam_detail[0][0] == False
    assert models.Sentence.objects.get(id=1).spam == False
    assert len(models.Sentence.objects.get(id=2).spam_detail[0]) == 2
    assert models.Sentence.objects.get(id=2).spam_detail[0][0] == False
    assert models.Sentence.objects.get(id=2).spam == False
    assert len(models.Sentence.objects.get(id=3).spam_detail[0]) == 2
    assert models.Sentence.objects.get(id=3).spam_detail[0][0] == False
    assert models.Sentence.objects.get(id=3).spam == False


def equip_profile_transformation_rate(models):
    """Define `Profile.transformation_rate`."""

    @memoized
    def transformation_rate(self, distance_type):
        """Compute profile transformation for distance `distance_type`.

        The transformation rate is the average of distances between a sentence a
        profile transformed and its parent. It can be computed for all
        available distance types (defined on `Sentence`).

        """

        if distance_type not in models.Sentence.DISTANCE_TYPES:
            raise ValueError("Unkown distance type (not one of {}): {}".format(\
                    distance_type, models.Sentence.DISTANCE_TYPES))
        distance_name = distance_type + '_distance'

        transformed_sentences = self.sentences.filter(parent__isnull=False).all()
        if len(transformed_sentences) == 0:
            raise ValueError("Profile has no reformulated sentences")

        return np.array([getattr(s.parent, distance_name)(s)
                        for s in transformed_sentences]).mean()

    models.Profile.transformation_rate = transformation_rate

    # Test
    try:
        models.Profile.objects.get(user__username='sl').transformation_rate('raw')
    except ValueError:
        pass  # Test passed
    else:
        raise Exception("Exception not raised on profile with no reformulated sentences")


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
