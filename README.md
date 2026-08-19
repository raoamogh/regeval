<p align="center">
  <img src="https://raw.githubusercontent.com/raoamogh/regeval/main/assets/banner.svg" alt="regeval banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/raoamogh/regeval/actions/workflows/ci.yml"><img src="https://github.com/raoamogh/regeval/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://pypi.org/project/regeval/"><img src="https://img.shields.io/pypi/v/regeval.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python versions">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Code style: ruff">
  <img src="https://img.shields.io/badge/coverage-98%25-brightgreen" alt="Test coverage">
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/raoamogh/regeval?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/last-commit/raoamogh/regeval" alt="Last commit">
  <a href="https://github.com/raoamogh/regeval/issues"><img src="https://img.shields.io/github/issues/raoamogh/regeval" alt="Open issues"></a>
  <a href="CODE_OF_CONDUCT.md"><img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"></a>
</p>

<h1 align="center">regeval</h1>

<p align="center"><b>pytest, but for LLM prompts.</b></p>

Catch LLM output regressions before they ship. Write down what "correct" looks like once, then automatically check it every time a prompt, model, or pipeline changes — locally, or as a comment on every pull request.

## Table of contents

- [The problem](#the-problem)
- [Quickstart (zero API key needed)](#quickstart-zero-api-key-needed)
- [Real example output](#real-example-output)
- [How it works](#how-it-works)
- [Three scorers, three use cases](#three-scorers-three-use-cases)
- [GitHub Action](#github-action)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [Support](#support)
- [License](#license)

## The problem

You tweak a system prompt to handle a new case. It works for that case. Did it silently break something else? Right now, almost nobody actually checks — they ship it and find out from a confused customer or a Slack message. regeval turns "did I break anything" from a guess into an actual, automated answer.

## Quickstart (zero API key needed)

regeval works against any OpenAI-compatible API — including [Ollama](https://ollama.com) running locally, so you can try the entire tool for free, with no signup:

```bash
pip install regeval
ollama pull llama3

regeval init
regeval run regeval.suite.yaml
```

```
     Example suite — 2/2 passed
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Case    ┃ Status ┃ Score ┃  Time ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ math    │  PASS  │  1.00 │ 0.35s │
│ capital │  PASS  │  1.00 │ 0.60s │
└─────────┴────────┴───────┴───────┘
All cases passed.
```

## Real example output

Not a mockup — actual output from `examples/semantic_similarity_suite.yaml` run against a local `llama3` model. Notice the model's phrasing doesn't match the expected text word-for-word, and the scorer correctly recognizes it's still right:

```
expected: "Gravity is the invisible force that pulls things toward
           each other, like what keeps your feet on the ground."

actual:   "Gravity is a magic force that pulls everything towards each
           other, like a big hug from the Earth, and it's what keeps
           you from floating off into space when you're playing outside!"

score: 0.879   →   PASS (threshold 0.75)
```

`exact_match` would have failed this outright. `semantic_similarity` correctly recognized the meaning matches even though not a single sentence is shared.

For genuinely nuanced cases, `llm_judge` gives you reasoning, not just a score:

```
case: explain_tradeoff
score: 0.80
reasoning: "The actual response accurately captures the main tradeoff
            between hash tables and sorted arrays, but does not exactly
            match the expected result in terms of the specific lookup
            times mentioned."
```

## How it works

1. Write a YAML suite: prompts, expected results, which scorer to use.
2. `regeval run suite.yaml` calls your provider concurrently for every case (real thread pool, bounded, not sequential), scores each result, and shows a live progress bar.
3. Save the result as your baseline: `--output baseline.json`.
4. Change your prompt. Run again: `--output current.json`.
5. `regeval diff baseline.json current.json` — flags anything that regressed, including cases that are still technically passing but whose score quietly dropped.
6. Wire the included GitHub Action into your CI so this happens automatically on every pull request, with the report posted as a comment.

## Three scorers, three use cases

| Scorer | Best for | How it works |
|---|---|---|
| `exact_match` | Answers that should be identical every time (support email, fixed facts) | String equality, optional case/whitespace normalization |
| `semantic_similarity` | Answers where wording varies but meaning shouldn't (policy explanations) | Embedding cosine similarity via your provider's `/embeddings` endpoint |
| `llm_judge` | Nuanced quality judgments (tone, completeness, correctness of reasoning) | A second model call grades the output 0-10 with a one-sentence reason |

## GitHub Action

Included in this repo at `.github/actions/regeval-action` — runs your suite, diffs against a stored baseline, and posts the result as a PR comment automatically:

```yaml
- uses: raoamogh/regeval/.github/actions/regeval-action@main
  with:
    suite: evals/support_bot.suite.yaml
    baseline: evals/baseline.json
    api-key: ${{ secrets.OPENAI_API_KEY }}
```

Full example: [`examples/github-workflow-example.yml`](examples/github-workflow-example.yml).

## Installation

```bash
pip install regeval                # core
pip install regeval[caching]       # + flightlock integration, dedupes identical prompts across reruns
```

## Roadmap

- [x] Provider interface (OpenAI-compatible: OpenAI, Groq, OpenRouter, Ollama)
- [x] Three scorers: exact match, semantic similarity, LLM-as-judge
- [x] Suite YAML loading with actionable validation errors
- [x] Concurrent execution engine
- [x] Regression diff with score-drop-threshold detection
- [x] Rich terminal reporting
- [x] CLI (`init`, `run`, `diff`)
- [x] GitHub Action for automated PR comments
- [ ] flightlock-backed call caching (in progress)
- [ ] Publish to PyPI
- [ ] Async provider support

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## Support

See [SUPPORT.md](SUPPORT.md).

## AI-assisted development

See [AI_POLICY.md](AI_POLICY.md) for how AI pair-programming was used in building this project, and what's expected of contributors.

## License

MIT — see [LICENSE](LICENSE).