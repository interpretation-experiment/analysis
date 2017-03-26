import csv
import os
import pickle

import click
import spacy
from nltk.corpus import brown

from analysis.utils import setup_spreadr
from analysis.language_model import NgramModel
from analysis import settings


# TODO: make sure this still works


@click.group()
@click.option('--db', type=str, required=True, prompt='Database to connect to')
@click.pass_obj
def cli(obj, db):
    """Data loading and conversion."""
    obj['DB'] = db


@cli.command()
@click.argument('n', type=click.IntRange(1, None))
@click.argument('type', type=click.Choice(['word', 'tag']))
def language_model(n, type):
    """Load an ngram model into its pickle file.

    `n` defines the size of the ngrams. `type` must be one of ['word', 'tag'],
    and defines what property of the words the model is about.

    """

    if type == 'word':
        tags = False
    elif type == 'tag':
        tags = True
    else:
        raise ValueError('unknown model type: {}'.format(type))

    outfile = settings.MODEL_TEMPLATE.format(n=n, type=type)

    # Check if outfile already exists
    if os.path.exists(outfile):
        click.secho('{} already exists, not overwriting it.'.format(outfile),
                    fg='red', bold=True)
        click.secho('Aborting.', fg='red')
        return

    # Load spaCy model
    click.secho('Loading spaCy model')
    nlp = spacy.load('en')

    # Load brown training data
    click.secho('Loading Brown News training data')
    brown_training = []
    for sent in brown.tagged_sents(categories='news', tagset='universal'):
        sent_training = []
        for word, tag in sent:
            # Normalise punctuation so spaCy will recognise it
            word = word.replace('`', "'")
            doc = nlp(word)
            # Subdivide the word if spaCy recognises it
            if len(doc) > 1:
                for subword in doc:
                    if subword.is_punct or subword.is_space:
                        continue
                    if tags:
                        sent_training.append(subword.pos_.upper())
                    else:
                        sent_training.append(subword.orth_.lower())
            else:
                if doc[0].is_punct or doc[0].is_space:
                    continue
                if tags:
                    sent_training.append(tag.upper())
                else:
                    sent_training.append(word.lower())
        if len(sent_training) > 0:
            brown_training.append(sent_training)

    # Train
    click.secho('Training {n}-gram {type} model'.format(n=n, type=type))
    model = NgramModel(n, brown_training, estimator_args=(0.2,))

    # Save to pickle
    click.secho("Saving model to '{}'".format(outfile))
    try:
        os.mkdir(settings.MODELS_FOLDER)
    except FileExistsError:
        pass
    with open(outfile, 'wb') as f:
        pickle.dump(model, f)

    click.secho('Done', fg='green', bold=True)


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
    click.echo('Writing {} sentences to {}...'
               .format(Sentence.objects.count(), filename),
               nl=False)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['id', 'bucket', 'is_root',
                                            coding, 'text', 'parent_text'])
        writer.writeheader()
        for sentence in Sentence.objects.all().order_by('bucket'):
            writer.writerow({
                'id': sentence.id,
                'bucket': sentence.bucket,
                'is_root': True if sentence.parent is None else False,
                coding: None,
                'text': sentence.text,
                'parent_text': getattr(sentence.parent, 'text', None)
            })

    click.secho('Done', fg='green', bold=True)
    click.secho('''
You can now:
* import {filename} to a spreadsheet,
* {coding}-code it,
* re-export it to csv,
* and put it in 'codings/{db}/{coding}/name-of-coder.csv'
'''.format(filename=filename, db=db, coding=coding), fg='cyan', bold=True)
