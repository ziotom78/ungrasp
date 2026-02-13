import nox

nox.options.default_venv_backend = "uv"


# Define the versions you want to test against
@nox.session(python=["3.11", "3.14"])
def tests(session):
    """Run the test suite."""
    session.run("uv", "sync", "--python", session.python, external=True)
    session.run("uv", "run", "pytest", external=True)
