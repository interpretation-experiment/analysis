import csv
import os

import click

from utils import setup_spreadr


@click.group()
@click.option('--db', type=str, required=True, prompt='Database to connect to')
@click.pass_obj
def cli(obj, db):
    """Data loading and conversion."""
    obj['DB'] = db


@cli.command()
@click.argument('coding', type=str)
@click.argument('filename', type=click.Path(file_okay=True, writable=True))
@click.pass_obj
def sentences_to_codable_csv(obj, coding, filename):
    """Load sentences into a csv file to code `coding`."""

    # Check if filename already exists
    if os.path.exists(filename):
        click.secho('{} already exists, not overwriting it.'.format(filename),
                    fg='red', bold=True)
        click.secho('Aborting.', fg='red')
        return

    # Set up connection to db
    db = obj['DB']
    click.echo('Connecting to MySQL database {}... '.format(db), nl=False)
    setup_spreadr(db)
    click.secho('OK', fg='green', bold=True)

    # Write sentences to csv
    from gists.models import Sentence
    click.echo('Writing {} sentences to {}...'.format(Sentence.objects.count(), filename),
               nl=False)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['id', 'bucket', 'is_root',
                                            coding, 'text', 'parent_text'])
        writer.writeheader()
        for sentence in Sentence.objects.all().order_by('bucket'):
            writer.writerow({'id': sentence.id,
                             'bucket': sentence.bucket,
                             'is_root': True if sentence.parent is None else False,
                             coding: None,
                             'text': sentence.text,
                             'parent_text': getattr(sentence.parent, 'text', None)})

    click.secho('Done', fg='green', bold=True)
    click.secho('''
You can now:
* import {filename} to a spreadsheet,
* {coding}-code it,
* re-export it to csv,
* and put it in 'codings/{db}/{coding}/name-of-coder.csv'
'''.format(filename=filename, db=db, coding=coding), fg='cyan', bold=True)
