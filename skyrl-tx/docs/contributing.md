# Contributing

We happily accept contributions!

## Development setup

You can run the unit tests with

```bash
uv run --extra dev pytest -s tests
```

You can build and view the documentation by running the following in the project directory:

```bash
uv run --extra dev mkdocs serve
```

The commands in [the CI workflow](https://github.com/tx-project/tx/tree/main/.github/workflows)
are the best source of useful commands to run during development.