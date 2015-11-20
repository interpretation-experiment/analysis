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
@click.argument('filename', type=click.Path(file_okay=True, writable=True))
@click.pass_obj
def sentences_to_spamcodable_csv(obj, filename):
    """Load sentences into an ods file to code spam."""

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
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'bucket', 'is_root', 'is_spam', 'text'])
        for sentence in Sentence.objects.all().order_by('bucket'):
            writer.writerow([sentence.id,
                             sentence.bucket,
                             True if sentence.parent is None else False,
                             None,
                             sentence.text])

    click.secho('Done', fg='green', bold=True)
    click.secho(('You can now import {} to a spreadsheet, spam-code it, '
                 're-export to csv and load it back in').format(filename),
                fg='cyan', bold=True)
