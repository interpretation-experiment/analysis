Analysis
========

Tools for analysis of the gistr data, and generation of null data.

Environment setup
-----------------

Using the Fish shell with virtualfish and Python 3.6+:

```
# Setup base environment
vf new -p (which python3) interpretation-experiment.analysis
pip install -r requirements.txt

# Get necessary nltk and spaCy data
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords brown cmudict
python -m spacy download en_core_web_md

# Check ou all the spreadr submodules and set up their environments
git submodule update --init
./setup_envs.sh
```

Experiment data importing:

* Get a `.sql` file of the whole database. Back it up.
* Edit your `.sql` file to set the database name you want to create locally to hold the data (in what follows here we'll use `spreadr_XXX`). There's three replacements to make (a comment, the `CREATE DATABASE` stance, and the `USE` stance).
* Import the data in MySQL: `mysql -u root < db.sql`

User setup (in a mysql shell, i.e. `mysql -u root`):

* Create the analysis user: `CREATE USER 'spreadr_analysis'@'localhost';`
* Grant it all privileges: `GRANT ALL on spreadr_XXX.* TO 'spreadr_analysis'@'localhost';` (change the database name to your own)

Usage
-----

Use `python -m analysis.cli --help` to explore what's possible.

Right now there are two parts in the code:

* Commands (in `analysis/commands/`) accessible through the cli, which are used to preprocess, code, or generate data
* Library stuff in `analysis/`, which serve as utilities for the notebooks in all the `notebooks/*` folders.

These correspond to three steps you should follow:

* Generate the language models needed by the notebooks if they're not already in `data/models/`, or if outdated:

```bash
for n in 1 2 3; do
  for type in word tag; do
    python -m analysis.cli load --db nothing language_model $n $type
  done
done
```

* Code the data with any needed codings (mostly spam, but could be any sentence-level feature); use `python -m analysis.cli load --db DB_NAME sentences_to_codable_csv CODING OUTFILE.csv` and follow the instructions.
* Open Jupyter and run any of the notebooks you want. They use the spreadr models, but augment them with utilities and any sentence-codings you provided from the previous step, thanks to stuff in `analysis/`.

TODO: fix `exp_1` notebooks and tests. It at least needs to upgrade the django version so that it supports python 3.6, and an update to the `settings_analysis` (so it doesn't read `MY_CNF`).

TODO: use a stub database for tests, instead of the real databases.
