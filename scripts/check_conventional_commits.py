#!/usr/bin/env python3
"""Validate Conventional Commit headers in a Git revision range."""

from __future__ import annotations

import re
import subprocess
import sys


HEADER = re.compile(
    r"^(?:build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(?:\([A-Za-z0-9$._/*, -]+\))?!?: \S.*$"
)
ZERO_SHA = "0" * 40


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def commits_in_range(base: str, head: str) -> list[str]:
    if not base or base == ZERO_SHA:
        return [head]

    output = git("rev-list", "--reverse", "--no-merges", f"{base}..{head}")
    return output.splitlines() if output else []


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <base-sha> <head-sha>", file=sys.stderr)
        return 2

    base, head = sys.argv[1:]
    invalid: list[tuple[str, str]] = []

    for commit in commits_in_range(base, head):
        subject = git("show", "-s", "--format=%s", commit)
        if not HEADER.fullmatch(subject):
            invalid.append((commit, subject))

    if invalid:
        print("Commit messages must use Conventional Commit format:", file=sys.stderr)
        print("  <type>(optional-scope)!: <description>", file=sys.stderr)
        print("Invalid commits:", file=sys.stderr)
        for commit, subject in invalid:
            print(f"  {commit[:12]} {subject}", file=sys.stderr)
        return 1

    print("All introduced non-merge commits use Conventional Commit format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
