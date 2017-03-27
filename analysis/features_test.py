import numpy as np
from numpy.testing import assert_approx_equal
import pytest
import spacy
from scipy import spatial

from . import features


class Namespace:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def nanequal(list1, list2):
    array1, array2 = np.array(list1), np.array(list2)
    return ((np.isnan(array1) == np.isnan(array2)).all() and
            (array1[np.isfinite(array1)] == array2[np.isfinite(array2)]).all())


def test_get_pronunciations():
    pronunciations = features._get_pronunciations()
    # We have the right kind of data.
    assert pronunciations['hello'] == [['HH', 'AH0', 'L', 'OW1'],
                                       ['HH', 'EH0', 'L', 'OW1']]
    # And what's loaded is memoized.
    assert pronunciations is features._get_pronunciations()


def test_get_aoa():
    aoa = features._get_aoa()
    # We have the right kind of data.
    assert aoa['time'] == 5.16
    # 'NA' terms were not loaded.
    assert 'wickiup' not in aoa
    assert len(aoa) == 30102
    # And what's loaded is memoized.
    assert aoa is features._get_aoa()


def test_get_zipf_frequency():
    freq = features._get_zipf_frequency()
    # We have the right data
    assert freq['labour'] == 5.35
    # And what's loaded is memoized.
    assert freq is features._get_zipf_frequency()


def test_get_clearpond():
    clearpond = features._get_clearpond()
    # We have the right kind of data.
    assert clearpond['phonological']['dog'] == 25
    assert clearpond['phonological']['cat'] == 50
    assert clearpond['phonological']['ghost'] == 14
    assert clearpond['phonological']['you'] == 49
    assert clearpond['orthographic']['dog'] == 20
    assert clearpond['orthographic']['cat'] == 33
    assert clearpond['orthographic']['ghost'] == 2
    assert clearpond['orthographic']['you'] == 4
    # And what's loaded is memoized.
    assert clearpond is features._get_clearpond()


def test_depth_under(nlp):
    doc = nlp("This is a sentence")
    assert features._depth_under(doc[0]) == 0
    assert features._depth_under(doc[1]) == 2
    assert features._depth_under(doc[2]) == 0
    assert features._depth_under(doc[3]) == 1
    # Punctuation is ignored
    doc = nlp("It is, he, in there")
    assert features._depth_under(doc[0]) == 0
    assert features._depth_under(doc[1]) == 2
    assert features._depth_under(doc[2]) == 0
    assert features._depth_under(doc[3]) == 0
    assert features._depth_under(doc[4]) == 2
    assert features._depth_under(doc[5]) == 0
    assert features._depth_under(doc[6]) == 1


def test_depth_above(nlp):
    doc = nlp("This is a sentence")
    assert features._depth_above(doc[0]) == 1
    assert features._depth_above(doc[1]) == 0
    assert features._depth_above(doc[2]) == 2
    assert features._depth_above(doc[3]) == 1
    # Punctuation is ignored
    doc = nlp("It is, he, in there")
    assert features._depth_above(doc[0]) == 1
    assert features._depth_above(doc[1]) == 0
    assert features._depth_above(doc[2]) == 0
    assert features._depth_above(doc[3]) == 1
    assert features._depth_above(doc[4]) == 0
    assert features._depth_above(doc[5]) == 2
    assert features._depth_above(doc[6]) == 1


def test_depth_prop(nlp):
    doc = nlp("This is a sentence")
    assert features._depth_prop(doc[0]) == .5
    assert features._depth_prop(doc[1]) == 0
    assert features._depth_prop(doc[2]) == 1
    assert features._depth_prop(doc[3]) == .5
    # Punctuation is ignored
    doc = nlp("It is, he, in there")
    assert features._depth_prop(doc[0]) == .5
    assert features._depth_prop(doc[1]) == 0
    assert features._depth_prop(doc[2]) == 0
    assert features._depth_prop(doc[3]) == .5
    assert features._depth_prop(doc[4]) == 0
    assert features._depth_prop(doc[5]) == 1
    assert features._depth_prop(doc[6]) == .5


def test_depth_subtree_prop(nlp):
    doc = nlp("This is a sentence")
    assert features._depth_subtree_prop(doc[0]) == 1
    assert features._depth_subtree_prop(doc[1]) == 0
    assert features._depth_subtree_prop(doc[2]) == 1
    assert features._depth_subtree_prop(doc[3]) == .5
    # Punctuation is ignored
    doc = nlp("It is, he, in there")
    assert features._depth_subtree_prop(doc[0]) == 1
    assert features._depth_subtree_prop(doc[1]) == 0
    assert features._depth_subtree_prop(doc[2]) == 0
    assert features._depth_subtree_prop(doc[3]) == 1
    assert features._depth_subtree_prop(doc[4]) == 0
    assert features._depth_subtree_prop(doc[5]) == 1
    assert features._depth_subtree_prop(doc[6]) == .5


def test_features_feature(models):
    # Note the id here is necessary to make the Sentence instance hashable, but
    # that value must not exist elsewhere, or a cached value will be used. We
    # use negative values to avoid conflict with sentences from the database,
    # and these values must be distinct for all tests here.
    sentence = models.Sentence(id=-1, text="Some real words in a new sentence")
    # Test some values
    assert sentence.feature('aoa') == np.mean([4.95, 4.72, 5.84])
    assert sentence.feature('orthographic_density') == \
        np.mean(np.log([9, 15, 12, 26, 13, 16, 2]))
    assert sentence.feature('orthographic_density', stopwords='exclude') == \
        np.mean(np.log([15, 12, 16, 2]))
    assert_approx_equal(sentence.feature('relatedness'), 0.61913721635813002)
    assert_approx_equal(sentence.feature('relatedness', stopwords='exclude'),
                        0.68386334821245887)
    # Can't have stopwords='nan'
    with pytest.raises(AssertionError):
        sentence.feature('aoa', stopwords='nan')
    # Can't use categorical features
    with pytest.raises(AssertionError):
        sentence.feature('pos')
    # dep depth doesn't care about stopwords
    assert sentence.feature('dep_depth') == \
        sentence.feature('dep_depth', stopwords='exclude')


def test_features_features(models, nlp):
    # Note the id here is necessary to make the Sentence instance hashable, but
    # that value must not exist elsewhere, or a cached value will be used. We
    # use negative values to avoid conflict with sentences from the database,
    # and these values must be distinct for all tests here.
    sentence = models.Sentence(id=-2, text="Some real words in a new sentence")
    # Test some values
    assert nanequal(sentence.features('aoa'),
                    [np.nan, 4.95, np.nan, np.nan, np.nan, 4.72, 5.84])
    assert nanequal(sentence.features('orthographic_density'),
                    np.log([9, 15, 12, 26, 13, 16, 2]))
    assert nanequal(sentence.features('orthographic_density',
                                      stopwords='nan'),
                    np.log([np.nan, 15, 12, np.nan, np.nan, 16, 2]))
    assert nanequal(sentence.features('orthographic_density',
                                      stopwords='exclude'),
                    np.log([15, 12, 16, 2]))
    assert nanequal(sentence.features('orthographic_density', rel='mean'),
                    np.log([9, 15, 12, 26, 13, 16, 2])
                    - np.mean(np.log([9, 15, 12, 26, 13, 16, 2])))
    assert nanequal(sentence.features('orthographic_density', rel='mean',
                                      stopwords='nan'),
                    np.log([np.nan, 15, 12, np.nan, np.nan, 16, 2])
                    - np.nanmean(np.log([np.nan, 15, 12, np.nan,
                                         np.nan, 16, 2])))
    assert nanequal(sentence.features('orthographic_density', rel='mean',
                                      stopwords='exclude'),
                    np.log([15, 12, 16, 2]) - np.mean(np.log([15, 12, 16, 2])))
    assert nanequal(sentence.features('pos'),
                    [spacy.symbols.DET, spacy.symbols.ADJ, spacy.symbols.NOUN,
                     spacy.symbols.ADP, spacy.symbols.DET, spacy.symbols.ADJ,
                     spacy.symbols.NOUN])
    assert nanequal(sentence.features('pos', stopwords='nan'),
                    [np.nan, spacy.symbols.ADJ, spacy.symbols.NOUN, np.nan,
                     np.nan, spacy.symbols.ADJ, spacy.symbols.NOUN])
    assert nanequal(sentence.features('pos', stopwords='exclude'),
                    [spacy.symbols.ADJ, spacy.symbols.NOUN, spacy.symbols.ADJ,
                     spacy.symbols.NOUN])
    # Categorical variables can't be sentence-relative
    with pytest.raises(AssertionError):
        sentence.features('pos', rel='mean')
    # Sentence features can't be used
    with pytest.raises(AssertionError):
        sentence.features('relatedness')
    # Unknown values for arguments raise errors
    with pytest.raises(AssertionError):
        sentence.features('aoa', rel='xxx')
    with pytest.raises(AssertionError):
        sentence.features('aoa', stopwords='xxx')
    with pytest.raises(AssertionError):
        sentence.features('xxx')


def test_features_letters_count(models):
    assert models.Sentence._letters_count(
        (None, Namespace(lower_='dog'), None)) == 3
    assert models.Sentence._letters_count('dog') == 3
    assert models.Sentence._letters_count() == \
        features._get_zipf_frequency().keys()


def test_features_aoa(models):
    assert models.Sentence._aoa(
        (None, Namespace(lower_='time'), None)) == 5.16
    assert models.Sentence._aoa('time') == 5.16
    # 'NA' terms were not loaded.
    assert np.isnan(models.Sentence._aoa(
        (None, Namespace(lower_='wickiup'), None)))
    assert np.isnan(models.Sentence._aoa('wickiup'))
    assert models.Sentence._aoa() == features._get_aoa().keys()


def test_features_zipf_frequency(models):
    assert models.Sentence._zipf_frequency(
        (None, Namespace(lower_='labour'), None)) == 5.35
    assert models.Sentence._zipf_frequency('labour') == 5.35
    assert np.isnan(models.Sentence._aoa(
        (None, Namespace(lower_='xxxxxxx'), None)))
    assert np.isnan(models.Sentence._aoa('xxxxxxx'))
    assert models.Sentence._zipf_frequency() == \
        features._get_zipf_frequency().keys()


def test_features_orthographic_density(models):
    assert models.Sentence._orthographic_density(
        (None, Namespace(lemma_='dog'), None)) == 20
    assert models.Sentence._orthographic_density('dog') == 20
    assert np.isnan(models.Sentence._orthographic_density(
        (None, Namespace(lemma_='xxxxxxx'), None)))
    assert np.isnan(models.Sentence._orthographic_density('xxxxxxx'))
    assert models.Sentence._orthographic_density() == \
        features._get_clearpond()['orthographic'].keys()


def test_features_ngram_logprob(models, nlp):
    doc = nlp("and it is")
    tok = doc[2]

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._ngram_logprob(1, 'word', None)
    # Unknown models are rejected
    with pytest.raises(AssertionError):
        models.Sentence._ngram_logprob(4, 'word', (doc, tok, 2))
    with pytest.raises(AssertionError):
        models.Sentence._ngram_logprob(1, 'xxxx', (doc, tok, 2))
    # Test a few values
    assert_approx_equal(
        models.Sentence._ngram_logprob(1, 'word', (doc, tok, 2)),
        6.973013530610437)
    assert_approx_equal(
        models.Sentence._ngram_logprob(3, 'word', (doc, tok, 2)),
        10.610482907507068)


def test_features_n_gram_type(models, nlp):
    doc = nlp("and it is")
    tok = doc[2]

    # Test a few values
    assert_approx_equal(
        models.Sentence._1_gram_word((doc, tok, 2)),
        6.973013530610437)
    assert_approx_equal(
        models.Sentence._3_gram_word((doc, tok, 2)),
        10.610482907507068)
    assert_approx_equal(
        models.Sentence._3_gram_tag((doc, tok, 2)),
        5.443133027293688)
    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._1_gram_word()


def test_features_depth_under(models, nlp):
    doc = nlp("It is, he, in there")

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._depth_under(None)
    # Test a few values
    assert models.Sentence._depth_under((None, doc[0], None)) == 0
    assert models.Sentence._depth_under((None, doc[1], None)) == 2
    assert models.Sentence._depth_under((None, doc[2], None)) == 0
    assert models.Sentence._depth_under((None, doc[3], None)) == 0
    assert models.Sentence._depth_under((None, doc[4], None)) == 2
    assert models.Sentence._depth_under((None, doc[5], None)) == 0
    assert models.Sentence._depth_under((None, doc[6], None)) == 1


def test_features_depth_above(models, nlp):
    doc = nlp("It is, he, in there")

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._depth_above(None)
    # Test a few values
    assert models.Sentence._depth_above((None, doc[0], None)) == 1
    assert models.Sentence._depth_above((None, doc[1], None)) == 0
    assert models.Sentence._depth_above((None, doc[2], None)) == 0
    assert models.Sentence._depth_above((None, doc[3], None)) == 1
    assert models.Sentence._depth_above((None, doc[4], None)) == 0
    assert models.Sentence._depth_above((None, doc[5], None)) == 2
    assert models.Sentence._depth_above((None, doc[6], None)) == 1


def test_features_depth_prop(models, nlp):
    doc = nlp("It is, he, in there")

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._depth_prop(None)
    # Test a few values
    assert models.Sentence._depth_prop((None, doc[0], None)) == .5
    assert models.Sentence._depth_prop((None, doc[1], None)) == 0
    assert models.Sentence._depth_prop((None, doc[2], None)) == 0
    assert models.Sentence._depth_prop((None, doc[3], None)) == .5
    assert models.Sentence._depth_prop((None, doc[4], None)) == 0
    assert models.Sentence._depth_prop((None, doc[5], None)) == 1
    assert models.Sentence._depth_prop((None, doc[6], None)) == .5


def test_features_depth_subtree_prop(models, nlp):
    doc = nlp("It is, he, in there")

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._depth_subtree_prop(None)
    # Test a few values
    assert models.Sentence._depth_subtree_prop((None, doc[0], None)) == 1
    assert models.Sentence._depth_subtree_prop((None, doc[1], None)) == 0
    assert models.Sentence._depth_subtree_prop((None, doc[2], None)) == 0
    assert models.Sentence._depth_subtree_prop((None, doc[3], None)) == 1
    assert models.Sentence._depth_subtree_prop((None, doc[4], None)) == 0
    assert models.Sentence._depth_subtree_prop((None, doc[5], None)) == 1
    assert models.Sentence._depth_subtree_prop((None, doc[6], None)) == .5


def test_features_pos(models, nlp):
    doc = nlp("This is a sentence")

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._pos(None)
    # Test a few values
    assert models.Sentence._pos((None, doc[0], None)) == spacy.symbols.DET
    assert models.Sentence._pos((None, doc[1], None)) == spacy.symbols.VERB
    assert models.Sentence._pos((None, doc[2], None)) == spacy.symbols.DET
    assert models.Sentence._pos((None, doc[3], None)) == spacy.symbols.NOUN


def test_features_dep(models, nlp):
    doc = nlp("This is a sentence")

    # No target is not possible
    with pytest.raises(AssertionError):
        models.Sentence._dep(None)
    # Test a few values
    assert models.Sentence._dep((None, doc[0], None)) == spacy.symbols.nsubj
    # For some reason ROOT isn't is spacy.symbols
    assert models.Sentence._dep((None, doc[1], None)) == doc[1].dep
    assert models.Sentence._dep((None, doc[2], None)) == spacy.symbols.det
    assert models.Sentence._dep((None, doc[3], None)) == spacy.symbols.attr


def test_features_relatedness(models, nlp):
    doc = nlp("This is a sentence")
    assert models.Sentence._relatedness(tuple(doc[0:3])) == \
        np.mean([spatial.distance.cosine(doc[0].vector, doc[1].vector),
                 spatial.distance.cosine(doc[1].vector, doc[2].vector),
                 spatial.distance.cosine(doc[0].vector, doc[2].vector)])


def test_features_dep_depth(models, nlp):
    doc = nlp("This is a sentence")
    # Works with any sublist of tokens
    assert models.Sentence._dep_depth(doc) == 2
    assert models.Sentence._dep_depth(tuple(doc[1:3])) == 2
