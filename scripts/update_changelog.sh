#!/usr/bin/env bash
# Auto-generate CHANGELOG entry from git commits since last tag.
# Usage: bash scripts/update_changelog.sh [version_tag]
# Appends to CHANGELOG.md under ## [Unreleased]

set -euo pipefail

VERSION="${1:-$(date +%Y-%m-%d)}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANGELOG="$PROJECT_DIR/CHANGELOG.md"

# Get commits since last tag
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

if [ -z "$LAST_TAG" ]; then
    COMMITS=$(git log --pretty=format:"- %s" --no-merges)
else
    COMMITS=$(git log "${LAST_TAG}..HEAD" --pretty=format:"- %s" --no-merges)
fi

if [ -z "$COMMITS" ]; then
    echo "No new commits since $LAST_TAG. Nothing to add."
    exit 0
fi

# Categorize commits
categorize() {
    local prefix="$1"
    local label="$2"
    local matches
    matches=$(echo "$COMMITS" | grep -iE "^\- .*${prefix}:" || true)
    if [ -n "$matches" ]; then
        echo "### ${label}"
        echo "$matches" | sed -E "s/^\- .*${prefix}:?\s*/- /"
        echo ""
    fi
}

ADDED=$(categorize "feat\|add\|new" "Added")
CHANGED=$(categorize "refactor\|update\|change\|dep" "Changed")
FIXED=$(categorize "fix\|bug\|hotfix" "Fixed")
REMOVED=$(categorize "remove\|delete\|drop" "Removed")

ENTRY="## [${VERSION}] — $(date +%Y-%m-%d)

${ADDED}${CHANGED}${FIXED}${REMOVED}"

# Check if Unreleased section exists, prepend after it
if grep -q "^## \[Unreleased\]" "$CHANGELOG"; then
    # Insert after the first blank line after [Unreleased]
    awk -v entry="$ENTRY" '
        /^## \[Unreleased\]/ { print; found=1; next }
        found && /^$/ { print; print ""; print entry; found=0; next }
        { print }
    ' "$CHANGELOG" > "$CHANGELOG.tmp" && mv "$CHANGELOG.tmp" "$CHANGELOG"
    echo "✅ Added entry under [Unreleased] in CHANGELOG.md"
else
    # Prepend to file
    {
        echo "# Changelog"
        echo ""
        echo "## [Unreleased]"
        echo ""
        echo "$ENTRY"
        echo ""
        tail -n +3 "$CHANGELOG"
    } > "$CHANGELOG.tmp" && mv "$CHANGELOG.tmp" "$CHANGELOG"
    echo "✅ Created CHANGELOG.md with new entry"
fi

echo ""
echo "Preview:"
echo "---"
echo "$ENTRY"
echo "---"
echo ""
echo "Review and edit CHANGELOG.md, then commit."
