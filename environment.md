# Environment setup

This is temporary (does not scale, is not easy to reproduce), and needs some automation. They're just notes to be able to recreate the process.

* Create a python3 virtualenv ("interpretation-experiment.analysis") with `--system-sites-packages` (so's not to redownload numpy, ntlk, and all that straight away)
* `pip install click`
* `vf addpath ~/.virtualens/spreadr/lib/python3.4/site-packages` to be able to use spreadr classes
* `pip install jsonschema terminado` to get ipython to work
* `pip install sklearn`
* `pip install seaborn`
* in python, `nltk.download()` and download wordnet,punkt,stopwords
