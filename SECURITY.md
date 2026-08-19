# Security Policy

## Supported Versions

regeval is in early alpha (0.x). Security fixes will be released as patch versions on the latest 0.x line. There is no long-term support for older versions at this stage.

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public GitHub issue.**

Instead, report it privately via [GitHub's private vulnerability reporting](https://github.com/raoamogh/regeval/security/advisories/new), or email amoghagrao@gmail.com directly.

Please include a description of the vulnerability, its potential impact, and steps to reproduce.

You can expect an initial response within a few days.

## Known considerations

- **Suite YAML files can contain API keys.** If you hardcode an `api_key` directly in a suite file rather than referencing an environment variable, and commit that file, you will leak the key. Prefer environment variable substitution or a secrets manager, especially in CI (see the included GitHub Action, which passes the key via a workflow secret, never a committed file).
- **regeval sends your prompts and expected results to whatever `base_url` you configure.** There is no allowlist or validation on this URL — regeval will happily send data to any endpoint, including a malicious one if a suite file has been tampered with. Only run suite files you trust, especially in CI where suite files might come from external pull requests.
- **`llm_judge` sends both your actual and expected text to a second model call.** If your expected/actual content is sensitive, be aware it's being sent to whichever provider you've configured for judging, not just the provider generating the original answer (though in practice these are usually the same provider).