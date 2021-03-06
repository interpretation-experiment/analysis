import numpy as np

from .utils import memoized, load_codings


# TODO: test
def _branch_sentences(sentence, with_root=False, with_leaf=True, cutoff=True):
    """TODO: docs."""

    from gists.models import GistsConfiguration
    config = GistsConfiguration.get_solo()

    # Walk back up to the head of the branch
    head = sentence.head

    if with_root:
        yield head.parent

    current = head
    while (current.children.kept.count() > 0
            and (not cutoff or current.depth < config.target_branch_depth)):
        if current.children.kept.count() > 1:
            raise ValueError("More than one (kept) children for sentence {}"
                             .format(current.id))
        yield current
        current = current.children.kept.first()

    if with_leaf:
        # Yield the final leaf
        yield current


# TODO: test
def equip_sentence_branch_sentences(models):
    """TODO: docs."""
    models.Sentence.branch_sentences = _branch_sentences


def equip_sentence_shaping(models):
    """Import and compute shaping codings on sentences.

    Codings imported:
    * `spam` (with `spam_detail`, and `nonspam` on `SentenceManager`
      and `QuerySet`)

    Computed codings:
    * `doublepost` (with `nondoublepost` on `SentenceManager` and `QuerySet`)
    * `rogue` (with `nonrogue` on `SentenceManager` and `QuerySet`)

    Also update Sentence.LOADED_CODINGS which lists loaded codings, and `kept`
    and `with_dropped()` on `SentenceManager` and `QuerySet` which filters on
    sentences kept after deciding on all shaping codings.

    """

    from django.db.models.manager import Manager
    from django.db.models.query import QuerySet

    # Get database name
    from django.conf import settings as django_settings
    db = django_settings.DATABASES['default']['NAME']

    # Load spam codings
    spam_codings = load_codings(
        db, 'spam',
        lambda c: c.lower() == 'true' or c.lower() == 'yes' or c == '1')

    # Add them as properties to Sentence
    models.Sentence._spam_codings = spam_codings
    models.Sentence.spam_detail = property(memoized(
        lambda self: self._spam_codings.get(self.id, [(False, None)])
    ))
    models.Sentence._self_spam = property(memoized(
        lambda self: np.mean([spam for (spam, _) in self.spam_detail]) > 0.5
    ))
    models.Sentence.spam = property(memoized(
        lambda self: (self._self_spam or
                      (self.parent is not None and self.parent.spam))
    ))

    # Give easy access to non-spam sentences
    def get_nonspam(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if not s.spam]
        return self.filter(pk__in=ids)

    Manager.nonspam = property(get_nonspam)
    QuerySet.nonspam = property(get_nonspam)

    # Create doublepost codings
    def get_self_doublepost(self):
        if self.parent is None:
            # Root sentences are not doubleposts
            return False

        if self.profile.user.is_staff:
            # Staff-made sentences are not self_doublepost (though they
            # can descend from a self_doublepost)
            return False

        # So we're from a non-staff profile. Are we a self_doublepost?
        others = self.tree.sentences\
            .filter(profile=self.profile)\
            .exclude(id=self.id)\
            .order_by('id')
        # If there is another sentence by this same (non-staff) profile,
        # with smaller id, we're self_doublepost.
        return (others.count() > 0 and others.first().id < self.id)

    models.Sentence._self_doublepost = property(memoized(get_self_doublepost))
    models.Sentence.doublepost = property(memoized(
        lambda self: (self._self_doublepost or
                      (self.parent is not None and self.parent.doublepost))
    ))

    # Give easy access to non-doublepost sentences
    def get_nondoublepost(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if not s.doublepost]
        return self.filter(pk__in=ids)

    Manager.nondoublepost = property(get_nondoublepost)
    QuerySet.nondoublepost = property(get_nondoublepost)

    # Create rogue codings
    def get_self_rogue(self):
        if self.parent is None:
            # Root sentences are not rogue
            return False

        if self.children.nonspam.nondoublepost.count() > 0:
            # We're not a leaf, so not self_rogue (though we could have
            # rogue children)
            return False

        # We're a potential leaf, do we compete with other leaves in this
        # branch?
        head = self.head
        depth = self.depth
        competitors = \
            [s for s in self.tree.sentences.nonspam.nondoublepost.all()
             if (s.children.nonspam.nondoublepost.count() == 0
                 and s.head == head)]

        betters = [s for s in competitors if s.depth > depth]
        if len(betters) > 0:
            # Another leaf (or even several) in this branch is strictly
            # deeper
            return True

        equals = [s for s in competitors if s.depth == depth]
        if len(equals) > 0:
            # Another leaf (or even several) in this branch is as deep as
            # we are. Take the sentence with smallest id as non-self_rogue.
            equals = sorted(equals, key=lambda s: s.id)
            return equals[0] != self

        # No better competitor, so we're not self_rogue
        return False

    models.Sentence._self_rogue = property(memoized(get_self_rogue))
    models.Sentence.rogue = property(memoized(
        lambda self: (self._self_rogue or
                      (self.children.nonspam.nondoublepost.count() > 0 and
                       np.all([c.rogue for c in
                               self.children.nonspam.nondoublepost.all()])))
    ))

    # Give easy access to non-rogue sentences
    def get_nonrogue(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if not s.rogue]
        return self.filter(pk__in=ids)

    Manager.nonrogue = property(get_nonrogue)
    QuerySet.nonrogue = property(get_nonrogue)

    models.Sentence.kept = property(memoized(
        lambda self: not self.spam and not self.doublepost and not self.rogue
    ))

    # Easily access sentences we want to keep
    def get_kept(self):
        if self.model != models.Sentence:
            raise ValueError('Only available on Sentence')

        ids = [s.id for s in self.all() if s.kept]
        return self.filter(pk__in=ids)

    Manager.kept = property(get_kept)
    QuerySet.kept = property(get_kept)
    Manager.with_dropped = lambda self, with_dropped: \
        self if with_dropped else self.kept
    QuerySet.with_dropped = lambda self, with_dropped: \
        self if with_dropped else self.kept

    LOADED_CODINGS = getattr(models.Sentence, 'LOADED_CODINGS', set())
    LOADED_CODINGS.update(['spam', 'doublepost', 'rogue'])
    models.Sentence.LOADED_CODINGS = LOADED_CODINGS


def equip_model_managers_with_bucket_type(models):
    """Add `training`, `experiment`, `game` to `Sentence` and `Tree` model
    managers and querysets.

    These let you quickly narrow down a queryset to the named bucket.

    """

    from django.db.models.manager import Manager
    from django.db.models.query import QuerySet

    def filter_bucket(self, bucket_name):
        if self.model == models.Sentence:
            return self.filter(bucket__exact=bucket_name)
        elif self.model == models.Tree:
            return self.filter(root__bucket__exact=bucket_name)
        else:
            raise ValueError('Only available on Sentence and Tree')

    # This will work for Sentence and Tree
    Manager.training = property(lambda self: filter_bucket(self, 'training'))
    QuerySet.training = property(lambda self: filter_bucket(self, 'training'))
    Manager.experiment = property(
        lambda self: filter_bucket(self, 'experiment'))
    QuerySet.experiment = property(
        lambda self: filter_bucket(self, 'experiment'))
    Manager.game = property(lambda self: filter_bucket(self, 'game'))
    QuerySet.game = property(lambda self: filter_bucket(self, 'game'))


def equip_sentence_with_head_depth(models):
    """Define `head` and `depth` on `Sentence`s."""

    @memoized
    def get_head(self):
        """Get the head of the branch this sentence is in,
        bailing if the sentence is root."""

        if self.parent is None:
            raise ValueError('Already at root')
        if self.parent.parent is None:
            return self
        return self.parent.head

    models.Sentence.head = property(get_head)

    @memoized
    def get_depth(self):
        """Get the depth of the sentence in its branch."""

        if self.parent is None:
            return 0
        return 1 + self.parent.depth

    models.Sentence.depth = property(get_depth)
