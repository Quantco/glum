# Contributing

We welcome contributions of any kind!

- New features
- Feature requests
- Bug reports
- Documentation
- Tests
- Questions

## Pull request process

- Before opening a non-trivial PR, please first discuss the change you wish to
  make via issue, Slack, email or any other method with the owners of this
  repository. This is meant to prevent spending time on a feature that will not
  be merged.
- Please make sure that a new feature comes with adequate tests. If these
  require data, please check if any of our existing test data sets fits the
  bill.
- Please make sure that all functions come with proper docstrings. If you do
  extensive work on docstrings, please check if the Sphinx documentation renders
  them correctly. The CI system builds it on every commit and pushes the
  rendered HTMLs to
  `https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/quantcore.glm/{YOUR_COMMIT}/index.html`
- Please make sure you have our pre-commit hooks installed.
- If you fix a bug, please consider first contributing a test that _fails_
  because of the bug and then adding the fix as a separate commit, so that the
  CI system picks it up.
- Please add an entry to the change log and increment the version number
  according to the type of change. We use semantic versioning. Update the major
  if you break the public API. Update the minor if you add new functionality.
  Update the patch if you fixed a bug. All changes that have not been released
  are collected under the date `UNRELEASED`.

## Releases

- We make package releases infrequently, but usually any time a new non-trivial
  feature is contributed or a bug is fixed. To make a release, just open a PR
  that updates the change log with the current date. Once that PR is approved
  and merged, you can create a new release on
  [GitHub](https://github.com/Quantco/quantcore.glm/releases/new). Use the
  version from the change log as tag and copy the change log entry into the
  release description. New releases on GitHub are automatically deployed to the
  QuantCo conda channel.
