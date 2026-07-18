# Contributing

## Commit and pull request titles

Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
format:

```text
<type>(optional-scope)!: <description>
```

Allowed types are `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`,
`refactor`, `revert`, `style`, and `test`. Use `!` for a breaking change and
describe it in the commit or pull request body.

Examples:

```text
feat(mmu): add transparent translation support
fix: handle signed division overflow
refactor!: remove the legacy bus interface
```

CI validates every non-merge commit introduced by a pull request or push.
Squash-merge pull requests using the pull request title as the commit title, so
release-please evaluates the title checked by CI. The `Validate PR title` check
also fails when a title does not follow this format.

## Local checks

Run the same checks as CI before opening a pull request:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
cargo package --locked
```

## Releases

After CI succeeds on `master`, release-please creates or updates a release pull
request from the conventional commits since the previous release. Merging that
pull request causes CI to run again, then release-please creates the Git tag and
GitHub release. The tagged crate is verified and published to crates.io using
the `CARGO_REGISTRY_TOKEN` GitHub Actions secret.

Do not edit `CHANGELOG.md`, package versions, or
`.release-please-manifest.json` for a normal release; release-please owns those
changes.
