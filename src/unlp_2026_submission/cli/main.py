import typer

from unlp_2026_submission.cli.commands.eval import app as eval_app
from unlp_2026_submission.cli.commands.invoke import app as invoke_app
from unlp_2026_submission.cli.commands.kb import app as kb_app

app = typer.Typer(no_args_is_help=True)

app.add_typer(eval_app, name="evaluate")
app.add_typer(invoke_app)
app.add_typer(kb_app, name="kb", help="Knowledge base management commands")

if __name__ == "__main__":
    app()