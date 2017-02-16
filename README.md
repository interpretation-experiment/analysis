Analysis
========

Tools for analysis of the gistr data, and generation of null data.

Environment setup
-----------------

Using the Fish shell with virtualfish and Python 3.6+:

```
git submodule update --init  # to check out the spreadr submodule

vf new -p (which python3) interpretation-experiment.analysis

pip install -r spreadr/requirements.txt

# The following will override a few of the above's requirements
# (spreadr isn't always as up-to-date), but that's okay.
pip install -r requirements.txt

# Get necessary nltk and spaCy data
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords
python -m spacy.en.download all
```

Experiment data importing:

* Get a `.sql` file of the whole database. Back it up.
* Edit your `.sql` file to set the database name you want to create locally to hold the data (in what follows here we'll use `spreadr_XXX`). There's three replacements to make (a comment, the `CREATE DATABASE` stance, and the `USE` stance).
* Import the data in MySQL: `mysql -u root < db.sql`

User setup (in a mysql shell, i.e. `mysql -u root`):

* Create the analysis user: `CREATE USER 'spreadr_analysis'@'localhost';`
* Grant it all privileges: `GRANT ALL on spreadr_XXX.* TO 'spreadr_analysis'@'localhost';` (change the database name to your own)

Finally, migrate the database: in the `spreadr` folder, `DJANGO_SETTINGS_MODULE=spreadr.settings_analysis DB_NAME=spreadr_XXX python manage.py migrate` (change the database name to your own).

Usage
-----

Use `python analysis.py --help` to explore what's possible.

Right now there are two parts in the code:

* Stuff in `analysis.py` and `commands/`, which is used to preprocess, code, or generate data
* Stuff in `utils.py` and `linguistics.py`, which serve as utilities for the notebooks in all the `notebooks_*/` folders.

These correspond to two steps you should follow:

* Code the data with any needed codings (mostly spam, but could be any sentence-level feature); use `python analysis.py load --db DB_NAME sentences_to_codable_csv CODING OUTFILE.csv` and follow the instructions.
* Open Jupyter and run any of the notebooks you want. They use the spreadr models, but augment them with utilities and any sentence-codings you provided from the previous step, thanks to `utils.py` and `linguistics.py`.

Note that, for now, the notebooks in a `notebooks_XXX/` folder are written for the dependencies and the API spreadr exposed when the corresponding data was collected. That may change in the future, but right now your best bet to get those notebooks working is to look up the version indicated in `notebooks_XXX/spreadr-version`, and check out that version (as a git tag) in `spreadr/`. Then re-install the requirements and re-run the data importing before running the notebooks.
