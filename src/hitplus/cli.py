import click


@click.group()
@click.version_option(package_name="hitplus")
def cli() -> None:
    """Hit+ ML Framework — pitch-by-pitch hitter outcome models."""


@cli.command()
@click.option("--model", required=True, help="Submodel name (e.g., swing_decision)")
@click.option(
    "--full", is_flag=True, help="Use full temporal split instead of dev mode"
)
@click.option("--force", is_flag=True, help="Force re-run of all steps")
@click.option(
    "--until", default=None, help="Run pipeline up to and including this step"
)
def run(model: str, full: bool, force: bool, until: str | None) -> None:
    """Run the full pipeline for a submodel."""
    click.echo(f"Pipeline run for '{model}' not yet implemented.")


@cli.command()
@click.option("--model", required=True, help="Submodel name")
@click.option("--full", is_flag=True, help="Use full dataset")
def extract(model: str, full: bool) -> None:
    """Extract raw data from the database."""
    click.echo(f"Extract for '{model}' not yet implemented.")


@cli.command()
@click.option("--model", required=True, help="Submodel name")
@click.option("--full", is_flag=True, help="Use full dataset")
def train(model: str, full: bool) -> None:
    """Train models for a submodel."""
    click.echo(f"Train for '{model}' not yet implemented.")


@cli.command()
@click.option("--model", required=True, help="Submodel name")
@click.option("--full", is_flag=True, help="Use full dataset")
@click.option("--plot", is_flag=True, help="Generate validation plots")
def validate(model: str, full: bool, plot: bool) -> None:
    """Run validation for a submodel."""
    click.echo(f"Validate for '{model}' not yet implemented.")


@cli.command()
@click.option("--model", required=True, help="Submodel name")
@click.option("--baseline", required=True, help="Baseline model version")
@click.option("--candidate", required=True, help="Candidate model version")
def compare(model: str, baseline: str, candidate: str) -> None:
    """Compare two model versions."""
    click.echo(f"Compare for '{model}' not yet implemented.")


@cli.command()
@click.option("--model", required=True, help="Submodel name")
def inspect(model: str) -> None:
    """Inspect artifacts for a submodel."""
    click.echo(f"Inspect for '{model}' not yet implemented.")


@cli.command()
@click.option("--model", required=True, help="Submodel name")
@click.option(
    "--type",
    "plot_type",
    required=True,
    type=click.Choice(["calibration", "importance", "roc", "zone_heatmap"]),
    help="Plot type to generate",
)
def viz(model: str, plot_type: str) -> None:
    """Generate visualizations for a submodel."""
    click.echo(f"Viz '{plot_type}' for '{model}' not yet implemented.")
