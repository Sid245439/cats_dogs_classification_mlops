#!/bin/bash
#
# Quick CI/CD Demo - Use this when recording the "code change to deployment" flow
# 1. Make a small change
# 2. Commit & push
# 3. Then show GitHub Actions in browser (https://github.com/Sid245439/cats_dogs_classification_mlops/actions)
#

cd "$(dirname "$0")/.."

echo "=========================================="
echo "CI/CD Demo - Trigger pipeline with a change"
echo "=========================================="

# Add a timestamp to README (simulates "code change")
echo "" >> README.md
echo "<!-- Demo: $(date) -->" >> README.md

echo ""
echo "Staged changes:"
git add README.md
git status

echo ""
echo "Commit and push to trigger CI/CD..."
git commit -m "Demo: trigger CI/CD pipeline - $(date +%Y-%m-%d)"
git push origin main

echo ""
echo "=========================================="
echo "Push complete! Pipeline triggered."
echo ""
echo "Next: Open in browser:"
echo "  https://github.com/Sid245439/cats_dogs_classification_mlops/actions"
echo ""
echo "Show: test → train → build → deploy jobs"
echo "=========================================="
