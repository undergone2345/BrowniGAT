# Contributing To BrowniGAT

Thanks for contributing to BrowniGAT.

The project is evolving from a target-prioritization repository into a broader biomedical AI infrastructure stack covering:

- multimodal biomedical graph ingestion
- foundation-model workspace and trainer orchestration
- engine-style scheduling and recovery control planes
- intervention design and synthetic biology planning

## What We Welcome

- bug reports with reproducible examples
- benchmark additions and stronger baselines
- new modality importers and schema validators
- trainer, scheduler, and checkpoint lifecycle improvements
- synthetic biology and intervention-design extensions
- documentation, tutorials, and result interpretation guides

## Development Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the test suite:

```bash
python -m unittest discover -s tests
```

## Suggested Contribution Flow

1. Open an issue describing the bug, gap, or proposed feature.
2. Create a focused branch.
3. Keep changes scoped to one major improvement when possible.
4. Add or update tests whenever behavior changes.
5. Update README, configs, or docs if the user-facing workflow changes.
6. Open a pull request with motivation, implementation notes, and validation details.

## Coding Expectations

- Prefer clear, modular utilities over monolithic scripts.
- Keep configuration-driven behavior in config files when possible.
- Preserve compatibility with the lightweight toy datasets and tests.
- Document new engine artifacts, scheduler files, or output schemas.

## Validation Checklist

Before opening a PR, try to include:

- `python -m unittest discover -s tests`
- a short note describing which configs or commands you ran
- output or screenshots if your change affects reporting or plots

## Project Direction

Good contributions usually strengthen one of these public project goals:

- make multimodal biological data easier to ingest and normalize
- make foundation training workflows more reproducible and engine-like
- make biomedical intervention design outputs more interpretable and actionable
- make the repository easier for outside researchers to reuse and extend
