import os
import csv
import json
import curses
from curses import textpad
import functools
import random
import logging
import itertools

import numpy as np
import click

from analysis import setup as setup_spreadr
from analysis.transformations import format_alignment
from analysis.utils import memoized, token_eq
from analysis.settings import ALIGNMENT_GAP_CHAR


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MARGIN = 2
KEY_INSTRUCTIONS = \
    '''←→: move, s: save and open next sentence, esc: discard and quit
1/2: add/remove gap above, 9/0: add/remove gap below, c: clean double gaps'''


@click.command()
@click.option('--db', type=str, required=True, prompt='Database to connect to')
@click.option('--outfile', type=click.Path(file_okay=True, writable=True),
              help='Where to save the gold alignments')
def cli(db, outfile):
    """Align pairs of parent-child sentences for future parameter
    optimisation."""

    # Initialise output file
    fieldnames = ('sentence_id', 'sentence_coding', 'parent_coding')
    aligned_sentence_ids = []
    if os.path.exists(outfile):
        click.secho("Appending to '{}'. ".format(outfile) +
                    'Sentences already aligned in this file will not be '
                    'proposed again.', fg='yellow')
        with open(outfile, 'r') as out:
            reader = csv.DictReader(out, fieldnames=fieldnames)
            for row in reader:
                if np.all([fieldname == value
                           for fieldname, value in row.items()]):
                    # Skip the header
                    continue
                # Record all already aligned sentence ids
                aligned_sentence_ids.append(row['sentence_id'])
    else:
        # Just write the csv's header and move on
        click.secho("Creating '{}'".format(outfile), fg='yellow')
        with open(outfile, 'w') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

    # Connect to db
    click.echo('Setting up spreadr and connecting to MySQL database {}... '
               .format(db), nl=False)
    models = setup_spreadr(db)
    click.secho('OK', fg='green', bold=True)

    # Get the sentence ids we can work on
    click.echo('Scanning for sentences that are still to be aligned '
               '(this could take a few minutes)... ', nl=False)
    kept_sentence_ids = list(models.Sentence.objects.kept
                             .filter(parent__isnull=False)
                             .values_list('id', flat=True))
    alignable_sentence_ids = list(set(kept_sentence_ids)
                                  .difference(aligned_sentence_ids))
    random.shuffle(alignable_sentence_ids)
    click.secho('OK', fg='green', bold=True)

    # Open outfile for writing (with line buffering) and delegate to curses UI
    outmessages = []
    with open(outfile, 'a', buffering=1) as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        curses.wrapper(tui, writer, outfile, len(aligned_sentence_ids),
                       alignable_sentence_ids, outmessages)

    for msg in outmessages:
        click.secho(msg)


def tui(screen, writer, filename, n_aligned_file, alignable_sentence_ids,
        outmessages):
    from gists.models import Sentence

    # Hide cursor
    curses.curs_set(False)

    # Set up colors
    assert curses.has_colors()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

    # Prepare our logging and main window
    ((screen_height, screen_width),
     (header_height, header_width, header_y, header_x),
     (main_height, main_width, main_y, main_x),
     (footer_height, footer_width, footer_y, footer_x),
     (log_height, log_width, log_y, log_x)) = get_window_sizes_coords()
    header = curses.newwin(header_height, header_width, header_y, header_x)
    main = curses.newwin(main_height, main_width, main_y, main_x)
    main.keypad(True)
    footer = curses.newwin(footer_height, footer_width, footer_y, footer_x)
    log_window = curses.newwin(log_height, log_width, log_y, log_x)

    # Log window scrolling and formatting
    log_window.scrollok(True)
    log_window.idlok(True)

    # Add the log handler
    log_handler = CursesHandler(log_window)
    log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    # Prepare our tracking state
    sid = alignable_sentence_ids.pop()
    sentence = Sentence.objects.get(id=sid)
    alignment = equalise_alignment(
        (sentence.parent.tokens, sentence.tokens, 0), True)
    cursor = 0

    # Layout formatting
    textpad.rectangle(screen, header_y - 1, header_x - 1,
                      header_y + header_height, header_x + header_width)
    textpad.rectangle(screen, footer_y - 1, footer_x - 1,
                      footer_y + footer_height, footer_x + footer_width)
    textpad.rectangle(screen, log_y - 1, log_x - 1,
                      log_y + log_height, log_x + log_width)
    screen.refresh()
    n_aligned_session = 0
    message = ''

    # Main event loop
    while True:

        # Draw main window
        main.clear()
        draw_alignment(main, alignment, cursor)
        main.noutrefresh()

        # Draw header
        header.clear()
        header_text = ("#{} → #{} (tree #{})"
                       .format(sentence.parent.id, sentence.id,
                               sentence.tree.id))
        header_remaining_width = header_width - len(header_text) - 1
        header_text += (("{:>" + "{}".format(header_remaining_width) + "}")
                        .format('{} sentences aligned in {} '
                                '({} from this session), {} more alignable'
                                .format(n_aligned_file + n_aligned_session,
                                        filename,
                                        n_aligned_session,
                                        len(alignable_sentence_ids))))
        header.addstr(header_text)
        header.noutrefresh()

        # Draw footer
        footer.clear()
        footer.addstr(KEY_INSTRUCTIONS)
        footer.addstr('\n' + message, curses.color_pair(2) | curses.A_BOLD)
        message = ''
        footer.noutrefresh()

        # Refresh screen and wait for a key
        curses.doupdate()
        key = main.getch()

        if key == curses.KEY_RIGHT:
            # Move right
            if cursor < max(len(alignment[0]), len(alignment[1])) - 1:
                cursor += 1
        elif key == curses.KEY_LEFT:
            # Move left
            if cursor > 0:
                cursor -= 1
        elif key == ord('s'):
            logger.info('Save current alignment')
            int_alignment = int_encode_alignment(
                equalise_alignment(alignment, True))
            writer.writerow({
                'sentence_id': sentence.id,
                'parent_coding': json.dumps(int_alignment[0]),
                'sentence_coding': json.dumps(int_alignment[1]),
            })
            n_aligned_session += 1

            if len(alignable_sentence_ids) > 0:
                logger.info('Load next sentence')
                sid = alignable_sentence_ids.pop()
                sentence = Sentence.objects.get(id=sid)
                alignment = equalise_alignment(
                    (sentence.parent.tokens, sentence.tokens, 0), True)
                cursor = 0
                message = ('Sentence saved, next sentence (#{}) loaded.'
                           .format(sentence.id))
            else:
                # No sentences left to align
                outmessages.append('No more sentences to align. quitting.')
                break
        elif key == ord('c'):
            # Clean double gaps
            logger.info('Clean double gaps')
            alignment = equalise_alignment(alignment, True)
            message = 'Double gaps cleaned.'
        elif key == ord('1'):
            # Add gap above
            logger.info('+gap↑ at index %d', cursor)
            seq1, seq2, score = alignment
            seq1 = seq1[:cursor] + (ALIGNMENT_GAP_CHAR,) + seq1[cursor:]
            alignment = equalise_alignment((seq1, seq2, score), False)
        elif key == ord('2'):
            # Remove gap above
            seq1, seq2, score = alignment
            if token_eq(seq1[cursor], ALIGNMENT_GAP_CHAR):
                logger.info('-gap↑ at index %d', cursor)
                seq1 = seq1[:cursor] + seq1[cursor + 1:]
                alignment = equalise_alignment((seq1, seq2, score), False)
            else:
                logger.info('Ignore -gap↑ at index %d (not a gap)',
                            cursor)
        elif key == ord('9'):
            # Add gap below
            logger.info('+gap↓ at index %d', cursor)
            seq1, seq2, score = alignment
            seq2 = seq2[:cursor] + (ALIGNMENT_GAP_CHAR,) + seq2[cursor:]
            alignment = equalise_alignment((seq1, seq2, score), False)
        elif key == ord('0'):
            # Remove gap below
            seq1, seq2, score = alignment
            if token_eq(seq2[cursor], ALIGNMENT_GAP_CHAR):
                logger.info('-gap↓ at index %d', cursor)
                seq2 = seq2[:cursor] + seq2[cursor + 1:]
                alignment = equalise_alignment((seq1, seq2, score), False)
            else:
                logger.info('Ignore -gap↓ at index %d (not a gap)',
                            cursor)
        elif key == 27:  # Esc or Alt
            # If it was Alt then curses has already sent another key,
            # otherwise -1 is sent, meaning Escape was pressed.
            main.nodelay(True)
            nextkey = main.getch()
            if nextkey == -1:
                # Escape was pressed
                break
            # Return to delay
            main.nodelay(False)

    outmessages.append('{} new sentence alignments added to {}.'
                       .format(n_aligned_session, filename))


def get_window_sizes_coords():
    screen_width = curses.COLS
    screen_height = curses.LINES

    header_width = screen_width - 2 * MARGIN
    header_height = 1

    footer_width = screen_width - 2 * MARGIN
    footer_height = 3

    log_width = screen_width - 2 * MARGIN
    log_height = 20

    main_width = screen_width - 2 * MARGIN
    main_height = (screen_height
                   - header_height
                   - footer_height
                   - log_height
                   - 5 * MARGIN)

    header_y, header_x = MARGIN, MARGIN
    main_y, main_x = header_y + header_height + MARGIN, MARGIN
    footer_y, footer_x = main_y + main_height + MARGIN, MARGIN
    log_y, log_x = footer_y + footer_height + MARGIN, MARGIN

    return ((screen_height, screen_width),
            (header_height, header_width, header_y, header_x),
            (main_height, main_width, main_y, main_x),
            (footer_height, footer_width, footer_y, footer_x),
            (log_height, log_width, log_y, log_x))


def addrichlist(screen, i, j, richlist):
    cursor = j
    for style, span in richlist:
        screen.addstr(i, cursor, span, style_to_curses(style))
        cursor += len(span)


def int_encode_alignment(alignment):
    seq1, seq2, _ = alignment
    iseq1 = []
    iseq2 = []

    for seq, iseq in zip([seq1, seq2], [iseq1, iseq2]):
        i = 0
        for token in seq:
            if token_eq(token, ALIGNMENT_GAP_CHAR):
                iseq.append(-1)
            else:
                iseq.append(i)
                i += 1
        assert i == sum(not token_eq(token, ALIGNMENT_GAP_CHAR)
                        for token in seq)

    return (iseq1, iseq2)


def equalise_alignment(alignment, clean_inside):
    seq1, seq2, score = alignment
    seq1 = tuple(seq1)
    seq2 = tuple(seq2)

    seq1, seq2 = zip(
        *[(token1, token2)
          for token1, token2 in itertools.zip_longest(
              seq1, seq2, fillvalue=ALIGNMENT_GAP_CHAR)
          if (not clean_inside or
              not token_eq(token1, ALIGNMENT_GAP_CHAR) or
              not token_eq(token2, ALIGNMENT_GAP_CHAR))]
    )

    # Always strip trailing gaps
    while (token_eq(seq1[-1], ALIGNMENT_GAP_CHAR) and
           token_eq(seq2[-1], ALIGNMENT_GAP_CHAR)):
        seq1 = seq1[:-1]
        seq2 = seq2[:-1]

    return (seq1, seq2, score)


def draw_alignment(screen, alignment, cursor):
    formatted = format_alignment(alignment, width=screen.getmaxyx()[1],
                                 format='rich')
    tokens_passed = 0

    for i, (line1, line2) in enumerate(zip(formatted['text1'],
                                           formatted['text2'])):
        addrichlist(screen, i * 3, 0, line1)
        addrichlist(screen, i * 3 + 1, 0, line2)

        # Every second item in line1 is whitespace
        if tokens_passed <= cursor < tokens_passed + len(line1) / 2:
            cursor_line = cursor - tokens_passed
            cursor_abs = sum(len(line1[2 * j][1]) + len(line1[2 * j + 1][1])
                             for j in range(cursor_line))
            screen.addstr(i * 3 + 2, cursor_abs,
                          '^' * max(len(line1[cursor_line * 2][1]),
                                    len(line2[cursor_line * 2][1])))

        tokens_passed += int(len(line1) / 2)  # Skip all the spaces


@memoized
def style_to_curses(style):
    style_to_curses_map = {
        'fg': {
            'red': curses.color_pair(1),
            'green': curses.color_pair(2),
            'yellow': curses.color_pair(3),
            'blue': curses.color_pair(4),
            'magenta': curses.color_pair(5),
            'cyan': curses.color_pair(6),
            'white': curses.color_pair(7),
        },
        'style': {
            'faint': curses.A_DIM,
            'bold': curses.A_BOLD,
            # Python 3.7 adds support for curses A_ITALIC,
            # but until then we use A_UNDERLINE.
            # See https://bugs.python.org/issue30101
            'italic': curses.A_UNDERLINE,
        }
    }

    curses_list = [style_to_curses_map[prop][value] for prop, value in style]
    return functools.reduce(lambda a, b: a | b, curses_list, 0)


class CursesHandler(logging.Handler):

    def __init__(self, screen):
        logging.Handler.__init__(self)
        self.screen = screen

    def emit(self, record):
        try:
            msg = self.format(record)
            screen = self.screen
            screen.addstr('{}\n'.format(msg))
            screen.noutrefresh()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
