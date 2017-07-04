import itertools

import spacy
from Bio import pairwise2
import colors
from frozendict import frozendict

from .utils import memoized, mappings, token_eq
from . import settings


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
                 or ' ' * (nbor is None or not token_eq(nbor, next_token))
                 or '_')
    else:
        orth = token
        space = ' '
    assert len(space) == 1

    return orth, space


# TODO: test
def format_alignment(alignment, width=80, depth=0, format='ansi'):
    """Pretty-print an alignment as returned by `align_lemmas`,
    `width` characters wide, width a margin proportional to `depth`.

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

    known_formats = ('ansi', 'rich')
    if format not in known_formats:
        raise ValueError("Unknown format type ('{}'), must be one of {}"
                         .format(format, known_formats))

    margin = _margin(depth)

    seq1, seq2, score = alignment[:3]
    assert len(seq1) == len(seq2)
    out = {'score': score, 'text1': [], 'text2': []}
    line1 = []
    line2 = []

    for i in range(len(seq1)):
        # Get current and succeeding tokens
        tok1, tok2 = seq1[i], seq2[i]
        if i < len(seq1) - 1:
            next_tok1 = seq1[i+1]
            next_tok2 = seq2[i+1]
        else:
            next_tok1 = next_tok2 = None

        # Get tokens' orth and whitespace
        orth1, space1 = _token_orth_whitespace(tok1, next_tok1)
        orth2, space2 = _token_orth_whitespace(tok2, next_tok2)
        len1 = len(orth1)
        len2 = len(orth2)

        # Set colors
        style1 = style2 = ()
        if (token_eq(tok1, settings.ALIGNMENT_GAP_CHAR) and
                not token_eq(tok2, settings.ALIGNMENT_GAP_CHAR)):
            # Insertion
            style2 += (('fg', 'green'),)
        elif (not token_eq(tok1, settings.ALIGNMENT_GAP_CHAR) and
              token_eq(tok2, settings.ALIGNMENT_GAP_CHAR)):
            # Deletion
            style1 += (('fg', 'red'),)
        elif orth1 != orth2:
            if tok1.lemma != tok2.lemma:
                # Replacement
                style1 += (('fg', 'yellow'),)
                style2 += (('fg', 'yellow'),)
            else:
                # Change in inflexion
                style1 += (('style', 'italic'),)
                style2 += (('style', 'italic'),)

        # Gray out stopwords
        if isinstance(tok1, spacy.tokens.Token) and is_stopword(tok1):
            style1 += (('style', 'faint'),)
        if isinstance(tok2, spacy.tokens.Token) and is_stopword(tok2):
            style2 += (('style', 'faint'),)

        # Pad length
        if len1 < len2:
            space1 += space1 * (len2 - len1)
        if len2 < len1:
            space2 += space2 * (len1 - len2)

        # Flush line if necessary
        if sum(len(word1) for _, word1 in line1) + len1 + len(space1) > width:
            out['text1'].append(line1)
            out['text2'].append(line2)
            line1 = []
            line2 = []

        # Append what we got
        line1.extend([(style1, orth1), ((), space1)])
        line2.extend([(style2, orth2), ((), space2)])

    # Flush line when finishing
    out['text1'].append(line1)
    out['text2'].append(line2)

    # Return this representation if asked to
    if format == 'rich':
        return out

    # Otherwise, convert to ANSI encoding and make into a string
    ansiout = ''
    for line1, line2 in zip(out['text1'], out['text2']):
        ansiout += margin
        for style1, word1 in line1:
            ansiout += colors.color(word1, **dict(style1))
        ansiout += '\n' + margin
        for style2, word2 in line2:
            ansiout += colors.color(word2, **dict(style2))
        ansiout += '\n\n'
    ansiout += margin + 'score={}'.format(score)

    return ansiout


# TODO: test
def format_deep_alignments(alignments, width=80, depth=0):
    """TODO: docs."""
    margin = _margin(depth)
    subouts = []
    for i, alignment in enumerate(alignments):
        title = "Alignment {}".format(i)
        subout = margin + title + '\n'
        subout += margin + len(title) * '=' + '\n'
        subout += format_deep_alignment(alignment, width, depth)
        subouts.append(subout)

    return '\n\n'.join(subouts)


# TODO: test
def format_deep_alignment(alignment, width=80, depth=0):
    """TODO: docs."""
    seq1 = alignment['seq1']
    seq2 = alignment['seq2']
    shallow_score = alignment['shallow_score']
    subalignments = alignment['subalignments']

    if len(subalignments) == 0:
        out = format_alignment([seq1, seq2, shallow_score], width, depth)
    else:
        subouts = []
        margin = _margin(depth)
        for i in range(len(subalignments)):
            subout = ''
            title = 'Subalignment {}'.format(i)
            subout += margin + title + '\n'
            subout += margin + '-' * len(title) + '\n'
            subout += format_deep_alignment_single_subalignment(alignment, i,
                                                                width, depth)
            subouts.append(subout)
        out = '\n\n'.join(subouts)

    return out


# TODO: test
def _margin(depth):
    """TODO: docs."""
    return ' ' * depth * 3


# TODO: test
def _find_exchange_idx(i, exchanges):
    """TODO: docs."""
    in_exchanges = [(start1 <= i < stop1) or (start2 <= i < stop2)
                    for (start1, stop1), (start2, stop2) in exchanges]
    if sum(in_exchanges):
        return in_exchanges.index(True)
    else:
        return None


# TODO: test
def _in_exchange(i, exchanges):
    """TODO: docs."""
    return _find_exchange_idx(i, exchanges) is not None


# TODO: test
def _encode_exchange(i, orth1, space1, orth2, space2, exchanges,
                     gap_char=settings.ALIGNMENT_GAP_CHAR):
    """TODO: docs."""
    exchange_idx = _find_exchange_idx(i, exchanges)
    if exchange_idx is None:
        # i is not in an exchange
        return orth1, space1, orth2, space2

    # Set the boundary orth
    (start1, stop1), (start2, stop2) = exchanges[exchange_idx]
    pad = 'right'
    if i == start1 or i == start2:
        exchange_orth = '|E{}'.format(exchange_idx)
        space_orth = gap_char
        if i == stop1 - 1 or i == stop2 - 1:
            pad = 'center'
            exchange_orth += '|'
            space_orth = False  # Will be replaced with space1 or space2
    elif i == stop1 - 1 or i == stop2 - 1:
        pad = 'left'
        exchange_orth = '|'
        space_orth = False  # Will be replaced with space1 or space2
    else:
        # i is not at the border of one of the gaps
        exchange_orth = gap_char
        space_orth = gap_char

    # And pad accordingly
    max_length = max(map(len, [orth1, orth2, exchange_orth]))
    padding = gap_char * (max_length - len(exchange_orth))
    if pad == 'left':
        exchange_orth = padding + exchange_orth
    elif pad == 'center':
        exchange_orth = exchange_orth[:-1] + padding + exchange_orth[-1:]
    else:
        exchange_orth += padding

    # Then change either orth1 or orth2 to exchange_orth
    if orth1 == gap_char:
        return exchange_orth, space_orth or space1, orth2, space2
    elif orth2 == gap_char:
        return orth1, space1, exchange_orth, space_orth or space2
    else:
        raise ValueError("i (={}) detected in an exchange (={}), "
                         "but neither orth1 (={}) not orth2 (={}) "
                         "equal the gap character (={})"
                         .format(i, exchanges[exchange_idx],
                                 orth1, orth2, gap_char))


# TODO: test
# TODO: add support for rich format
def format_deep_alignment_single_subalignment(alignment, subalignemnt_idx,
                                              width=80, depth=0):
    """TODO: docs."""

    from .contents import is_stopword

    margin = _margin(depth)

    seq1 = alignment['seq1']
    seq2 = alignment['seq2']
    shallow_score = alignment['shallow_score']
    deep_score = alignment['deep_score']
    subalignment = alignment['subalignments'][subalignemnt_idx]
    exchanges = list(subalignment.keys())

    assert len(seq1) == len(seq2)
    out = ''
    line1 = line2 = margin

    for i in range(len(seq1)):
        # Get current and succeeding tokens
        tok1, tok2 = seq1[i], seq2[i]
        if i < len(seq1) - 1:
            next_tok1 = seq1[i+1]
            next_tok2 = seq2[i+1]
        else:
            next_tok1 = next_tok2 = None

        # Get tokens' orth and whitespace
        orth1, space1 = _token_orth_whitespace(tok1, next_tok1)
        orth2, space2 = _token_orth_whitespace(tok2, next_tok2)
        orth1, space1, orth2, space2 = \
            _encode_exchange(i, orth1, space1, orth2, space2, exchanges)
        len1 = len(orth1)
        len2 = len(orth2)

        # Set colors
        if _in_exchange(i, exchanges):
            # Exchange. Also set the space's color as it could
            # be a gap_char.
            orth1 = colors.color(orth1, fg='blue')
            space1 = colors.color(space1, fg='blue')
            orth2 = colors.color(orth2, fg='blue')
            space2 = colors.color(space2, fg='blue')
        elif not isinstance(tok1, spacy.tokens.Token):
            # Insertion
            orth2 = colors.color(orth2, fg='green')
        elif not isinstance(tok2, spacy.tokens.Token):
            # Deletion
            orth1 = colors.color(orth1, fg='red')
        elif orth1 != orth2:
            if tok1.lemma != tok2.lemma:
                # Replacement
                orth1 = colors.color(orth1, fg='yellow')
                orth2 = colors.color(orth2, fg='yellow')
            else:
                # Change in inflexion
                orth1 = colors.color(orth1, style='italic')
                orth2 = colors.color(orth2, style='italic')

        # Gray out stopwords
        if isinstance(tok1, spacy.tokens.Token) and is_stopword(tok1):
            orth1 = colors.color(orth1, style='faint')
        if isinstance(tok2, spacy.tokens.Token) and is_stopword(tok2):
            orth2 = colors.color(orth2, style='faint')

        # Pad length
        if len1 < len2:
            space1 += space1 * (len2 - len1)
        if len2 < len1:
            space2 += space2 * (len1 - len2)

        # Flush line if necessary
        if colors.ansilen(line1) + len1 + len(space1) > width:
            out += line1 + '\n' + line2 + '\n\n'
            line1 = line2 = margin

        # Append what we got
        line1 += orth1 + space1
        line2 += orth2 + space2

    # Flush line when finishing
    out += line1 + '\n' + line2 + '\n\n'
    out += margin + 'deep_score={} shallow_score={}'.format(deep_score,
                                                            shallow_score)

    # Add all the exchanges
    for i, (exchange, exchange_deep_alignments) in \
            enumerate(subalignment.items()):
        out += '\n\n'
        title = 'Exchange {}'.format(i)
        out += margin + title + '\n'
        out += margin + '^' * len(title) + '\n'
        out += format_deep_alignments(exchange_deep_alignments,
                                      width, depth + 1)

    return out


# TODO: test
def normalise_alignment(alignment, clean_inside=True,
                        gap_char=settings.ALIGNMENT_GAP_CHAR):
    """TODO: docs."""
    seq1, seq2, score = alignment
    seq1 = tuple(seq1)
    seq2 = tuple(seq2)

    # Make both sequences equally long, and remove double
    # gaps if asked to
    seq1, seq2 = zip(
        *[(token1, token2)
          for token1, token2 in itertools.zip_longest(
              seq1, seq2, fillvalue=gap_char)
          if (not clean_inside or
              not token_eq(token1, gap_char) or
              not token_eq(token2, gap_char))]
    )

    # Always strip trailing gaps
    while (token_eq(seq1[-1], gap_char) and
           token_eq(seq2[-1], gap_char)):
        seq1 = seq1[:-1]
        seq2 = seq2[:-1]

    # Reorder contiguous gaps if cleaning inside too
    if clean_inside:
        keep_reordering = True
        while keep_reordering:
            gaps1 = gaps(seq1)
            gaps2 = gaps(seq2)
            keep_reordering = False
            for (start1, stop1), (start2, stop2) \
                    in itertools.product(gaps1, gaps2):
                if stop1 == start2:
                    # Reorder these two gaps
                    seq1 = (seq1[:start1] + seq1[start2:stop2] +
                            seq1[start1:stop1] + seq1[stop2:])
                    seq2 = (seq2[:start1] + seq2[start2:stop2] +
                            seq2[start1:stop1] + seq2[stop2:])
                    # And restart in case the gaps to reorder have changed
                    keep_reordering = True
                    break

    return (seq1, seq2, score)


# TODO: test
@memoized
def align_lemmas(tokens1, tokens2, gap_char=settings.ALIGNMENT_GAP_CHAR,
                 parameters=frozendict(settings.ALIGNMENT_PARAMETERS)):
    """Find optimal alignments between the lemmas of two lists of tokens.

    Alignments are computed between lists of lemmas (for now without a
    notion of semantic distance), and then converted back to lists of tokens.
    Returns the list of best-scoring alignments, each item being a tuple
    `(sequence1, sequence2, score, alignment_start, alignment_end)`.

    """

    # Create the index-token map
    idx2token = []
    token2idx = {}
    for i, tok in enumerate(itertools.chain(tokens1, tokens2, [gap_char])):
        idx2token.append(tok)
        if tok in token2idx:
            # This is not a problem in fact, we can simply overwrite
            # as injectivity is not necessary for idx->token, but it's
            #  nice to know if it happens at all
            raise ValueError("Token already in token2idx map")
        else:
            token2idx[tok] = i

    # Get lists of indices for our lists of tokens
    tokens1_idx = [[token2idx[tok]] for tok in tokens1]
    tokens2_idx = [[token2idx[tok]] for tok in tokens2]

    # Score a (mis)match between two lemmas
    def match_score(idx1, idx2):
        assert len(idx1) == 1 and len(idx2) == 1
        tok1 = idx2token[idx1[0]]
        tok2 = idx2token[idx2[0]]

        if not tok1.has_vector or not tok2.has_vector:
            similarity = int(tok1.lemma == tok2.lemma)
        else:
            similarity = tok1.similarity(tok2)

        return (parameters['COMPARE_FACTOR'] * similarity +
                parameters['COMPARE_ORIGIN'])

    # Align the indices
    alignments_idx = pairwise2.align.globalcs(
        tokens1_idx, tokens2_idx, match_score,
        parameters['GAP_OPEN'], parameters['GAP_EXTEND'],
        gap_char=[token2idx[gap_char]])

    # Convert the alignment back into tokens
    alignments = []
    for a in alignments_idx:
        tokens_lists = ([], [])
        for i in [0, 1]:
            for idx_chunk in a[i]:
                # idx_chunk can be a list of indices, or index itself
                if isinstance(idx_chunk, list):
                    tokens_lists[i].extend(
                        [idx2token[idx] for idx in idx_chunk])
                else:
                    assert isinstance(idx_chunk, int)
                    tokens_lists[i].append(idx2token[idx_chunk])
        alignments.append(normalise_alignment(tokens_lists + (a[2],)))

    return list(set(alignments))


# TODO: test
def gaps(sequence, gap_char=settings.ALIGNMENT_GAP_CHAR):
    """Find contiguous chunks of `gap_char=settings.ALIGNMENT_GAP_CHAR` in
    `sequence`, returning a list of `(start, outer-end)` indices."""

    found_gaps = []
    gap_start = None

    for i, el in enumerate(sequence):
        if token_eq(el, gap_char):
            if gap_start is None:
                # Open a gap
                gap_start = i
        else:
            if gap_start is not None:
                # Close a gap
                found_gaps.append((gap_start, i))
                gap_start = None

    # Close an ending gap
    if gap_start is not None:
        found_gaps.append((gap_start, i + 1))

    return found_gaps


def log(depth, *strings):
    """TODO: docs."""

    # print(depth * '~' + (depth > 0) * ' '
    #       + ' '.join(map(repr, strings)))
    # print()
    pass


# TODO: test
def deep_align_lemmas(tokens1, tokens2, depth=0,
                      gap_char=settings.ALIGNMENT_GAP_CHAR,
                      parameters=frozendict(settings.ALIGNMENT_PARAMETERS)):
    """TODO: docs."""

    log(depth, 'deep-aligning', tokens1, tokens2)
    base_alignments = align_lemmas(tokens1, tokens2)
    # deep_alignments =
    #     [ { 'seq1': sequence
    #       , 'seq2': sequence
    #       , 'shallow_score': int
    #       , 'deep_score': int
    #       , 'subalignments':
    #         [ { (gap1, gap2): [ deep_alignment, ... ]
    #           , ... <-- other pairs of gaps for this subalignment
    #           }
    #         , ... <-- other combinations of pairs of deep-aligned gaps
    #         ]
    #       }
    #     , ... <-- other deep alignments with equal score
    #     ]
    deep_alignments = []

    len1 = len(tokens1)
    len2 = len(tokens2)

    for seq1, seq2, shallow_score in base_alignments:
        log(depth, 'new base alignment...')
        gaps1 = gaps(seq1)
        gaps2 = gaps(seq2)

        # Check we haven't reached the bottom
        if (len(gaps1) == 0 or len(gaps2) == 0 or
                sum([g[1] - g[0] == len2 for g in gaps1]) or
                sum([g[1] - g[0] == len1 for g in gaps2])):
            # Either no gaps, or one of the gaps is the size of the
            # corresponding original token list, so stop the recursion.
            # No subalignments, so deep_score == shallow_score.
            log(depth, '...already minimal, not recurring further')
            deep_alignments.append({
                'seq1': seq1,
                'seq2': seq2,
                'shallow_score': shallow_score,
                'deep_score': shallow_score,
                'subalignments': []
            })
            continue
        log(depth, '...not minimal, looking for gap matches')

        # Deep-align each pair of gaps
        exchanges_deep_alignments = {}
        for gap1, gap2 in itertools.product(gaps1, gaps2):
            # Get the lists of tokens that correspond to the gaps
            subseq1 = tuple(seq1[slice(*gap2)])
            subseq2 = tuple(seq2[slice(*gap1)])
            exchanges_deep_alignments[(gap1, gap2)] = \
                deep_align_lemmas(subseq1, subseq2, depth+1)

        # Score all the subalignments of gaps1 to gaps2
        subalignments_scores = {}
        max_exchanges = min(len(gaps1), len(gaps2))
        for n_exchanges in range(max_exchanges + 1):
            for mapping in mappings(gaps1, gaps2, n_exchanges):
                # Sum the deep_scores of the first deep_alignment of each
                # exchange (the first is okay as all the deep_alignments
                # for a given exchange have the same deep_score)
                mapping_score = sum(
                    exchanges_deep_alignments[exchange][0]['deep_score']
                    for exchange in mapping
                )
                # Add to that the cost of n_exchanges
                subalignments_scores[mapping] = (
                    n_exchanges * parameters['EXCHANGE']
                    + mapping_score
                )
        best_subalignment_score = max(subalignments_scores.values())

        # Store the best mappingss in our deep_alignment
        subalignments = []
        for mapping, subalignment_score in subalignments_scores.items():
            if subalignment_score == best_subalignment_score:
                subalignments.append(dict(
                    (exchange, exchanges_deep_alignments[exchange])
                    for exchange in mapping
                ))
        deep_alignments.append({
            'seq1': seq1,
            'seq2': seq2,
            'shallow_score': shallow_score,
            'deep_score': shallow_score + best_subalignment_score,
            'subalignments': subalignments
        })
        log(depth,
            '{} subalignments scoring {} for this base alignment'
            .format(len(subalignments), best_subalignment_score))

    # Return the best deep_alignments
    best_score = max(da['deep_score'] for da in deep_alignments)
    log(depth,
        '{} deep_alignments scoring {}'
        .format(len(deep_alignments), best_score))
    return [da for da in deep_alignments if da['deep_score'] == best_score]


# TODO: test
def equip_sentence_alignments(models):
    """Define alignments between sentences.

    Alignments defined:
    * `align_lemmas`: align on complete list of lemmas
    * `align_content_lemmas`: align on list of content lemmas
    * `align_deep_lemmas`: deep-align on complete list of lemmas
    * `align_deep_content_lemmas`: deep-align on list of content lemmas

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

    @memoized
    def _align_deep_lemmas(self, sentence):
        """Find optimal deep-alignments between `self`'s and `sentence`'s
        lemmas, returning deep-alignments of `spacy.tokens.Token` objects."""

        return deep_align_lemmas(self.tokens, sentence.tokens)

    models.Sentence.align_deep_lemmas = _align_deep_lemmas

    @memoized
    def _align_deep_content_lemmas(self, sentence):
        """Find optimal deep-alignments between `self`'s and `sentence`'s
        lemmas, returning deep-alignments of `spacy.tokens.Token` objects."""

        return deep_align_lemmas(self.content_tokens, sentence.content_tokens)

    models.Sentence.align_deep_content_lemmas = _align_deep_content_lemmas
    models.Sentence.ALIGNMENT_TYPES = ['lemmas', 'content_lemmas',
                                       'deep_lemmas', 'deep_content_lemmas']
