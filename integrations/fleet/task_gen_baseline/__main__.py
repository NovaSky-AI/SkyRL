"""Module entrypoint for the task-generation baseline CLI."""

from integrations.fleet.task_gen_baseline.cli import generate_cli

if __name__ == "__main__":
    parser = generate_cli()
    args = parser.parse_args()
    args.func(args)
