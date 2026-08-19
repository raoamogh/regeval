# AI-Assisted Development Policy

## Disclosure

Portions of regeval — scaffolding, boilerplate, initial drafts of modules, and code review during development — were produced with AI pair-programming assistance (Claude, by Anthropic). This document explains how, and what that does and doesn't mean about the project.

## What AI assistance was used for

- **Scaffolding**: project structure, `pyproject.toml`, CI configuration.
- **First-draft implementations**: initial versions of the Provider, Scorer, Suite, Runner, Diff, and Report modules, based on a specified design (e.g. "implement a Scorer interface with exact-match, semantic-similarity, and LLM-as-judge implementations, following the same abstract-interface pattern as flightlock's Backend").
- **Test generation**: initial test cases, then reviewed and run against real behavior — including real concurrent execution tests, real fake-transport HTTP tests (no live network calls in the automated suite), and real manual runs against a local Ollama instance to verify end-to-end behavior before shipping.
- **Debugging**: AI assistance was used to diagnose real test failures during development, including a missing default parameter value that silently broke 7 tests, and a test double (`_FakeProvider`) that didn't account for the LLM-judge scorer's second, dynamically-built prompt.
- **Documentation**: README structure, docstrings, this policy.

## What was NOT delegated

- **Understanding**: every design decision — why the runner uses a bounded thread pool, why the diff engine flags score decay even on still-passing cases, why the judge scorer parses a strict `SCORE:`/`REASONING:` format instead of relying on JSON mode — is understood by the maintainer, not just pasted in.
- **Verification**: every test in this repo genuinely runs and passes, including manual end-to-end verification against a real local model (Ollama) for all three scorers, with real output inspected and sanity-checked (see the "Real example output" section of the [README](README.md), which is actual captured output, not a mockup).
- **Architecture decisions**: the Provider/Scorer abstraction pattern, the choice to route semantic-similarity through the provider's own `/embeddings` endpoint rather than pulling in a heavy ML dependency, the score-drop-threshold design in the diff engine — all deliberate, explainable choices.

## Guidance for contributors using AI tools

Using AI tools to help write a pull request is fine and doesn't need to be disclosed line-by-line — but the same standard applied to the maintainer's own use applies to contributors:

- **You must understand your own PR.** If asked "why did you do it this way" or "what does this do if X happens," you should be able to answer without re-generating the explanation.
- **You must actually run the tests, not just generate them** — including against a real provider (Ollama is free and requires no signup) if your change touches provider or scorer behavior, not just the fake-transport unit tests.
- **Don't submit AI output you haven't read.** Smaller, understood changes are preferred over large, unverified ones.
- **Be honest if asked.** If a maintainer asks whether AI was used for a specific part of a PR, answer directly.
- **You're responsible for correctness either way.** "The AI wrote it" is not a defense for a bug or a security issue (e.g. see the suite-YAML API key handling note in [SECURITY.md](SECURITY.md)).

## Why this disclosure exists

Transparency about tooling is a standard expectation in software development, not an admission of weakness. The measure of this codebase is whether it works, is tested, and is understood by its author — not how every line of it first got typed.

## Questions

Open an issue or see [SUPPORT.md](SUPPORT.md).