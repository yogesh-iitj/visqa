# Contributing to ViSQA

Thank you for your interest in contributing! Here's how to get started.

## Setup for Development

```bash
git clone https://github.com/yourname/visqa.git
cd visqa
pip install -e ".[dev,all]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use Black and isort:
```bash
black visqa/ scripts/
isort visqa/ scripts/
```

## Pull Request Process

1. Fork the repo and create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Run `pytest tests/` and ensure all pass
4. Open a PR with a clear description of what you changed and why

## Areas Welcome for Contribution

- New grounding model integrations (e.g. GLIP, OWLv2)
- New dataset loaders
- Speed optimizations
- Bug fixes
- Documentation improvements
