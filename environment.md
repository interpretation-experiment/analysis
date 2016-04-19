# Environment setup

This is temporary (does not scale, is not easy to reproduce), and needs some automation. They're just notes to be able to recreate the process.

* Create a python3.5 virtualenv ("interpretation-experiment.analysis")
* `pip install numpy`
* `pip install scipy nltk`
* `pip install click`
* `pip install jsonschema terminado` to get ipython to work
* `pip install sklearn`
* `pip install seaborn, statsmodels`
* in python, `nltk.download(['wordnet', 'punkt', 'stopwords'])`
* `vf addpath ~/.virtualens/spreadr/lib/python3.5/site-packages` to be able to use the spreadr environment without duplicating it. This works only if the `spreadr` environment corresponds to the commit checked out in the submodule of this repository.

Data importing

* Edit the `.sql` file to set the right database name you want (in the example here we'll use `spreadr_exp_1`)
* `mysql -u root < db.sql`

User setup (in a mysql shell, i.e. `mysql -u root`)

* Create the analysis user: `CREATE USER 'spreadr_analysis'@'localhost';`
* Grant it all privileges: `GRANT ALL on spreadr_exp_1.* TO 'spreadr_analysis'@'localhost';` (change the database name to your own)

Finally, migrate the database: in the `spreadr` folder, `env DJANGO_SETTINGS_MODULE=spreadr.settings_analysis DB_NAME=spreadr_exp_1 python manage.py migrate` (change the database name to your own).
