from nltk.corpus import stopwords as nltk_stopwords
from nltk.metrics import jaccard_distance, edit_distance
from nltk.stem.snowball import EnglishStemmer as SnowballStemmer
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from numpy import abs

from utils import memoized, import_spreadr_models


def equip_sentence_content_words(models):
    """Define sentence `content_words`."""

    def _filter(words, exclude_list):
        return filter(lambda w: w not in exclude_list, words)

    def filter_punctuation(words):
        return _filter(words, [',', '.', ';', '!', '?'])

    stopwords = set(nltk_stopwords.words('english'))
    stopwords.add("n't")  # Missing from the corpus, and appears with tokenization

    def filter_stopwords(words):
        return _filter(words, stopwords)

    def filter_lowercase(words):
        return map(lambda w: w.lower(), words)

    def filter_length(words):
        return filter(lambda w: len(w) > 2, words)

    stemmer = SnowballStemmer(ignore_stopwords=True)

    def filter_stem(words):
        return map(lambda w: stemmer.stem(w), words)

    filters = [nltk_word_tokenize,
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
    * `ordered_distance`
    * `unordered_distance`

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
    def ordered_distance(self, sentence, normalized=True):
        """Normalized levenshtein distance on (ordered) content words
        between `self` and `sentence`."""

        self_content_words = self.content_words
        sentence_content_words = sentence.content_words
        distance = edit_distance(self_content_words, sentence_content_words)
        norm = max(len(self_content_words), len(sentence_content_words))
        return  distance / norm if normalized else distance

    @memoized
    def unordered_distance(self, sentence):
        """Jaccard distance on (unordered) content words between `self` and `sentence`."""
        return jaccard_distance(set(self.content_words), set(sentence.content_words))

    models.Sentence.raw_distance = raw_distance
    models.Sentence.ordered_distance = ordered_distance
    models.Sentence.unordered_distance = unordered_distance

    # Testing this is hard (we don't have predictable data for it),
    # so we mostly test for stupid values only
    assert models.Sentence.objects.get(id=1).raw_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).ordered_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).unordered_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert abs(models.Sentence.objects.get(id=1).raw_distance(\
            models.Sentence.objects.get(id=2)) - .754098) <= 1e-6
    assert models.Sentence.objects.get(id=1).ordered_distance(\
            models.Sentence.objects.get(id=2)) == 1.0
    assert models.Sentence.objects.get(id=1).unordered_distance(\
            models.Sentence.objects.get(id=2)) == 1.0


def equip_spreadr_models():
    """Equip spreadr models with linguistics tools.

    Tools:
    * Content words on `Sentence`s
    * Distances on `Sentence`s

    """

    models = import_spreadr_models()
    equip_sentence_content_words(models)
    equip_sentence_distances(models)
