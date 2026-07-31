# Changelog Template

Use when drafting release notes manually.

## [Unreleased]

## [X.Y.Z] — YYYY-MM-DD

### Added
- New feature or capability.

### Changed
- Existing behavior modification.

### Deprecated
- Feature scheduled for removal.

### Removed
- Deleted feature or file.

### Fixed
- Bug fix description.

### Security
- Security-related change.

---

## Categories

| Category | Use When |
|----------|----------|
| **Added** | New features, files, endpoints, dependencies |
| **Changed** | Modifications to existing behavior, config, defaults |
| **Deprecated** | Marking something for future removal |
| **Removed** | Deleting features, files, endpoints |
| **Fixed** | Bug fixes, regression fixes |
| **Security** | CVE patches, auth changes, dependency bumps |

## Auto-Generation

Generate from git commits since last tag:

```bash
bash scripts/update_changelog.sh v0.3.0
```

This parses commit messages and categorizes by prefix:
- `feat:`, `add:`, `new:` → Added
- `refactor:`, `update:`, `change:`, `dep:` → Changed
- `fix:`, `bug:`, `hotfix:` → Fixed
- `remove:`, `delete:`, `drop:` → Removed

## Before Release Checklist

- [ ] All features documented in CHANGELOG.md
- [ ] Version bumped in `pyproject.toml`
- [ ] Version injected in `app/desktop.py` title
- [ ] Tag created: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Tag pushed: `git push origin vX.Y.Z`
- [ ] GitHub Actions builds complete
- [ ] Release notes reviewed on GitHub
- [ ] Draft released (or published)
