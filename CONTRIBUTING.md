# Contributing

All contributor documentation lives in **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** —
architecture map, the add-a-provider walkthrough (the repo's only extension point), the two
test tiers, CI/workflows, and the ground rules enforced in review.

Quick setup:

```sh
git clone <your fork>
cd agentic-search-arena
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/arena -q     # must be 100% green before you start
```
