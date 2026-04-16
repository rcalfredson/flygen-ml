# Local Fixture Configuration

This directory stores lightweight fixture metadata for trusted raw `.data` and `.trx` pairs without checking those large raw files into git.

## Current Approach

- raw fixture files live outside the repo
- tests read local fixture paths from `local_fixture_registry.json`
- tests skip gracefully if the configured files are not present on the current machine

## Current Trusted Pair

The initial local fixture registry is configured around this pair:

- `/media/Synology4/Yang Chen/2024-11-10/c42__2024-11-10__13-49-12.data`
- `/media/Synology4/Yang Chen/2024-11-10/c42__2024-11-10__13-49-12.trx`

This is treated as an arbitrarily selected starting pair for semantic validation.

## How To Use

1. Mount or otherwise make the configured paths available locally.
2. Run the test suite.
3. As trusted semantic expectations become known, fill in the optional expected fields in `local_fixture_registry.json`.

## Notes

- Do not commit large raw fixture files here.
- Keep expected semantic fields narrow and fixture-backed.
- Prefer a few trusted pairs over many weakly understood examples.
