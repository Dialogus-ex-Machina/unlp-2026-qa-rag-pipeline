import typer

from unlp_2026_submission.cli.eval import app as eval_app
from unlp_2026_submission.cli.invoke import app as invoke_app
from unlp_2026_submission.cli.kb import app as kb_app

app = typer.Typer()

app.add_typer(eval_app, name="evaluate")
app.add_typer(invoke_app)
app.add_typer(kb_app, name="kb", help="Knowledge base management commands")

if __name__ == "__main__":
    app()