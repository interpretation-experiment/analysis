import itertools

import spacy
from Bio import pairwise2

from .utils import memoized, color, clear_colors, TermColors


# TODO: test
def _token_orth_whitespace(token, next_token):
    """Gets a `token`'s `orth_` and following whitespace, depending on
    the `next_token` that will be printed after it.

    The space returned is either ' ' or '_' if `token` and `next_token`
    should not be separated.

    """

    if isinstance(token, spacy.tokens.Token):
        orth = token.orth_
        try:
            nbor = token.nbor()
        except IndexError:
            nbor = None
        space = (token.whitespace_
                 or ' ' * (nbor is None
                           or type(nbor) != type(next_token)  # spacy bug
                           or nbor != next_token)
                 or '_')
    else:
        orth = token
        space = ' '
    assert len(space) == 1

    return orth, space


# TODO: test
def format_alignment(alignment, width=80):
    """Pretty-print an alignment as returned by `align_lemmas`,
    `width` characters wide.

    Returns the string to print. Formatting is as follows:
    * red: a word that is deleted
    * green: a word that is inserted
    * blue: a word that is replaced comes in replacement
    * grayscale (on top of the above colors): for all stopwords
    * italics: a word that was not counted as a replacement by the alignment,
      but has a differnt `orth_` than its aligned word (usually the same
      lemma, but different inflexion)
    * an underscore between two tokens indicates they are not separated by
      whitespace in the original sentences

    """

    from .contents import is_stopword

    tokens1, tokens2, score, begin, end = alignment
    assert len(tokens1) == len(tokens2)
    out = ''
    line1 = ''
    line2 = ''

    for i in range(len(tokens1)):
        # Get current and succeeding tokens
        tok1 = tokens1[i]
        tok2 = tokens2[i]
        if i < len(tokens1) - 1:
            next_tok1 = tokens1[i+1]
            next_tok2 = tokens2[i+1]
        else:
            next_tok1 = None
            next_tok2 = None

        # Get tokens' orth and whitespace
        orth1, space1 = _token_orth_whitespace(tok1, next_tok1)
        len1 = len(orth1)
        orth2, space2 = _token_orth_whitespace(tok2, next_tok2)
        len2 = len(orth2)

        # Set colors
        if not isinstance(tok1, spacy.tokens.Token):
            # Insertion
            orth2 = color(orth2, TermColors.green)
        elif not isinstance(tok2, spacy.tokens.Token):
            # Deletion
            orth1 = color(orth1, TermColors.red)
        else:
            if orth1 != orth2:
                if tok1.lemma != tok2.lemma:
                    # Replacement
                    orth1 = color(orth1, TermColors.blue)
                    orth2 = color(orth2, TermColors.blue)
                else:
                    # Change in inflexion
                    orth1 = color(orth1, TermColors.italics)
                    orth2 = color(orth2, TermColors.italics)

            # Gray out stopwords
            if is_stopword(tok1):
                orth1 = color(orth1, TermColors.faint)
            if is_stopword(tok2):
                orth2 = color(orth2, TermColors.faint)

        # Pad length
        if len1 < len2:
            space1 += space1 * (len2 - len1)
        if len2 < len1:
            space2 += space2 * (len1 - len2)

        # Flush line if necessary
        if len(clear_colors(line1)) + len1 + len(space1) > width:
            out += line1 + '\n' + line2 + '\n\n'
            line1 = ''
            line2 = ''

        # Append what we got
        line1 += orth1 + space1
        line2 += orth2 + space2

    # Flush line when finishing
    out += line1 + '\n' + line2 + '\n\n'
    out += 'score={}'.format(score)

    return out


# TODO: test
@memoized
def align_lemmas(tokens1, tokens2):
    """Find optimal alignments between the lemmas of two lists of tokens.

    Alignments are computed between lists of lemmas (for now without a
    notion of semantic distance), and then converted back to lists of tokens.
    Returns the list of best-scoring alignments, each item being a tuple
    `(sequence1, sequence2, score, alignment_start, alignment_end)`.

    """

    # Create the hash-token map
    hashes_tokens = {}
    gap = '-'
    for tok in itertools.chain(tokens1, tokens2, [gap]):
        tok_hash = hash(tok)
        if tok_hash in hashes_tokens:
            assert tok == hashes_tokens[tok_hash]
        else:
            hashes_tokens[tok_hash] = tok

    # Get lists of hashes for our lists of tokens
    tokens1_hashes = [[hash(tok)] for tok in tokens1]
    tokens2_hashes = [[hash(tok)] for tok in tokens2]

    # Our current token match score function
    def hash_match_score(hash1, hash2):
        assert len(hash1) == 1
        assert len(hash2) == 1
        tok1 = hashes_tokens[hash1[0]]
        tok2 = hashes_tokens[hash2[0]]
        return 1.5 * (tok1.lemma == tok2.lemma) - .5

    # Align the hashes
    alignments_hashes = pairwise2.align.globalcs(
        tokens1_hashes, tokens2_hashes, hash_match_score,
        -.5, -.1, gap_char=[hash(gap)])

    # Convert the alignment back into tokens
    alignments = []
    for a in alignments_hashes:
        tokens_lists = ([], [])
        for i in [0, 1]:
            for hash_chunk in a[i]:
                # hash_chunk can be a list of hashes, or a hash itself
                if isinstance(hash_chunk, list):
                    tokens_lists[i].extend(
                        [hashes_tokens[h] for h in hash_chunk])
                else:
                    assert isinstance(hash_chunk, int)
                    tokens_lists[i].append(hashes_tokens[hash_chunk])
        alignments.append(tokens_lists + (a[2], a[3], a[4]))

    return alignments


# TODO: test
def equip_sentence_alignments(models):
    """Define alignments between sentences.

    Alignments defined:
    * `align_lemmas`: align on complete list of lemmas
    * `align_content_lemmas`: align on list of content lemmas

    Also add `Sentence.ALIGNMENT_TYPES` that lists available alignments.

    """

    @memoized
    def _align_lemmas(self, sentence):
        """Find optimal alignments between `self`'s and `sentence`'s lemmas,
        returning alignments of `spacy.tokens.Token` objects."""

        return align_lemmas(self.tokens, sentence.tokens)

    models.Sentence.align_lemmas = _align_lemmas

    @memoized
    def _align_content_lemmas(self, sentence):
        """Find optimal alignments between `self`'s and `sentence`'s lemmas,
        returning alignments of `spacy.tokens.Token` objects."""

        return align_lemmas(self.content_tokens, sentence.content_tokens)

    models.Sentence.align_content_lemmas = _align_content_lemmas
    models.Sentence.ALIGNMENT_TYPES = ['lemmas', 'content_lemmas']
