from nltk.corpus import stopwords as nltk_stopwords
from nltk.metrics import jaccard_distance, edit_distance
from nltk.stem.snowball import EnglishStemmer as SnowballStemmer
from nltk.tokenize import word_tokenize as nltk_word_tokenize

from utils import memoized, import_spreadr_models


def equip_sentence_content_words(models):
    """Define sentence content words.

    For each sentence text, we want the content words. So:
    * tokenize
    * set to lowercase
    * remove punctuation
    * remove words $\leq$ 2 characters
    * remove stopwords
    * stem

    and set the result as `content_words` on each `Sentence`.
    """

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
    @memoized
    def ordered_distance(self, sentence):
        self_content_words = self.content_words
        sentence_content_words = sentence.content_words
        return edit_distance(self_content_words, sentence_content_words) / \
            max(len(self_content_words), len(sentence_content_words))

    @memoized
    def unordered_distance(self, sentence):
        return jaccard_distance(set(self.content_words), set(sentence.content_words))

    models.Sentence.ordered_distance = ordered_distance
    models.Sentence.unordered_distance = unordered_distance

    # Testing this is hard (we don't have predictable data for it),
    # so we test values for 0 and 1 only
    assert models.Sentence.objects.get(id=1).ordered_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).unordered_distance(\
            models.Sentence.objects.get(id=1)) == 0.0
    assert models.Sentence.objects.get(id=1).ordered_distance(\
            models.Sentence.objects.get(id=2)) == 1.0
    assert models.Sentence.objects.get(id=1).unordered_distance(\
            models.Sentence.objects.get(id=2)) == 1.0


def equip_spreadr_models():
    models = import_spreadr_models()
    equip_sentence_content_words(models)
    equip_sentence_distances(models)
