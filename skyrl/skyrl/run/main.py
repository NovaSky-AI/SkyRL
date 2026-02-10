import typer

import skyrl
from skyrl.run.train import train

app = typer.Typer()

app.command(help="Train a model")(train)


@app.command()
def version():
    typer.echo(f"skyrl v{skyrl.__version__}")


if __name__ == "__main__":
    app()
