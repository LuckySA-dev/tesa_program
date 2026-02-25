# Contributing to TESA Defence AI

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/<your-org>/tesa-defence-ai.git
cd tesa-defence-ai
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
```

## Code Conventions

- **Imports**: Use `from hw import sprint, get_device` for hardware-safe output and device selection.
- **Print**: Never use bare `print()` with emoji characters — use `sprint()` to avoid `UnicodeEncodeError` on Windows.
- **Config**: All tuneable parameters go in `src/config.py` with env var overrides.
- **Source**: All core modules live in `src/`. Dashboard and CLI live at project root.

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Make your changes with clear, descriptive commit messages.
3. Run the smoke test: `python test_pipeline_short.py`
4. Run API test: `python src/test_api_real.py`
5. Run format check: `python cli.py compliance`
6. Open a PR against `main` with a summary of changes.

## Commit Messages

Use conventional commit prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `refactor:` — code restructure
- `test:` — add/update tests
- `chore:` — maintenance (deps, CI, etc.)

## Reporting Issues

Open an issue with:

1. Steps to reproduce
2. Expected vs actual behavior
3. Python version, OS, GPU info (`python cli.py info`)
