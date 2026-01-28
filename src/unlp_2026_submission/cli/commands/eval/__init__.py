import typer

from .accuracy import app as accuracy_app
from .mteb import app as mteb_app
from .faithfulness import app as faithfulness_app

app = typer.Typer()

app.add_typer(accuracy_app)
app.add_typer(mteb_app)
app.add_typer(faithfulness_app)