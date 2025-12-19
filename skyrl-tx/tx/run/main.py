import typer

import tx
from tx.run.train import train

app = typer.Typer()

app.command(help="Train a model")(train)


@app.command()
def version():
    typer.echo(f"tx v{tx.__version__}")


if __name__ == "__main__":
    app()
