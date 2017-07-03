import os
import csv
import json
import curses
import functools

import numpy as np
import click

from analysis import setup as setup_spreadr
from analysis.transformations import format_alignment


@click.command()
@click.option('--db', type=str, required=True, prompt='Database to connect to')
@click.option('--outfile', type=click.Path(file_okay=True, writable=True),
              help='Where to save the gold alignments')
def cli(db, outfile):
    """Align pairs of parent-child sentences for future parameter
    optimisation."""

    # Initialise output file
    fieldnames = ('sentence_id', 'sentence_coding', 'parent_coding')
    if os.path.exists(outfile):
        click.secho("Appending to '{}'".format(outfile))
        with open(outfile, 'r') as out:
            reader = csv.DictReader(out, fieldnames=fieldnames)
            for row in reader:
                if np.all([fieldname == value
                           for fieldname, value in row.items()]):
                    # Skip the header
                    continue
                # TODO: read existing data, get already-coded sentences,
                # and inform the user
                sentence_id = row['sentence_id']
                sentence_coding = json.loads(row['sentence_coding'])
                parent_coding = json.loads(row['parent_coding'])
                print(sentence_id, sentence_coding, parent_coding)
    else:
        click.secho("Creating '{}'".format(outfile))
        with open(outfile, 'w') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

    # Connect to db
    click.echo('Setting up spreadr and connecting to MySQL database {}... '
               .format(db), nl=False)
    setup_spreadr(db)
    click.secho('OK', fg='green', bold=True)

    # Open outfile for writing and delegate to curses UI
    with open(outfile, 'a') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        curses.wrapper(tui, writer)
        # writer.writerow({'sentence_id': 1,
        #                  'sentence_coding': json.dumps([-1, 0, 1]),
        #                  'parent_coding': json.dumps([0, 1, -1])})


def tui(screen, writer):
    from gists.models import Sentence

    # Clear screen
    screen.clear()
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

    # Pront a test sentence
    s = Sentence.objects.get(id=1071)
    formatted_alignment = format_alignment(s.parent.align_lemmas(s)[0],
                                           format='rich')
    for i, (line1, line2) in enumerate(zip(formatted_alignment['text1'],
                                       formatted_alignment['text2'])):
        addrichlist(screen, i * 3, 0, line1)
        addrichlist(screen, i * 3 + 1, 0, line2)

    screen.refresh()
    screen.getkey()


def addrichlist(screen, i, j, richlist):
    cursor = j
    for style, span in richlist:
        screen.addstr(i, cursor, span, style_to_curses(style))
        cursor += len(span)


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
    print(style_to_curses_map)

    curses_list = [style_to_curses_map[prop][value] for prop, value in style]
    return functools.reduce(lambda a, b: a | b, curses_list, 0)
