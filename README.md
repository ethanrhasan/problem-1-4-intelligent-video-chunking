# Problem 1.4: Intelligent Video Chunking — solution

Linear-cost, CPU-only video chunker: one low-res decode pass, cheap genre-aware boundary signals, a per-video profile, and dynamic-programming boundary selection with information-density-aware chunk lengths. Outputs scene / section / whole-video chunks with keyframes and indexing hints.

- Design, choices and results: [`docs/README.md`](docs/README.md)
- Code: [`src/`](src/) (entry point `src/main.py`), tests in [`tests/`](tests/)
- Exact commands: [`run.yaml`](run.yaml)

---

# Problem 1.4: Intelligent Video Chunking

## Submission instructions

1. Install the **AIntropy Grader Bot** GitHub App (https://github.com/apps/aintropy-grader-bot) and grant it access to your fork, if you haven't already — we can't evaluate your submission without this.
2. Put your code under `src/`.
3. Put your write-up under `docs/` — markdown, slides, a notebook, or another format of your choosing are all fine, as long as it's committed in this repo.
4. Fill in `run.yaml` with the **exact** commands needed to set up and run your code. We run these commands exactly as written to evaluate your submission — this is the one part of the submission we read literally, not loosely: if `run.yaml` doesn't work, we can't see your code run.
5. Push to your own fork's default branch any time during your test's time limit. No pull request is needed or expected.

Once you start the test you have until your test's time limit to complete the assignment — see your invitation email for the exact duration. We ignore any commits pushed to your repo after that.

Questions? careers@aintropy.ai
