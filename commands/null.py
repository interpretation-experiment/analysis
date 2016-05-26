import click


@click.group()
def cli():
    """Null data generation."""
    pass


@cli.command()
@click.argument('filename', type=click.Path(exists=True))
def example(filename):
    """An example subcommand using a `filename` argument."""
    click.echo('Example command with a file argument')
