#!/usr/bin/env bash
#
# deploy_docs.sh - Build and deploy Sphinx docs to GitHub Pages
#
# Usage:
#   ./scripts/deploy_docs.sh              # Deploy current branch
#   ./scripts/deploy_docs.sh main         # Deploy main branch to root
#   ./scripts/deploy_docs.sh sphinx_docs  # Deploy to /sphinx_docs/ subdirectory
#

set -e  # Exit on error

# Configuration
DOCS_DIR="docs"
BUILD_DIR="docs/build/html"
GH_PAGES_BRANCH="gh-pages"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current branch name
if [ -n "$1" ]; then
    CURRENT_BRANCH="$1"
else
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

# Determine deployment directory
if [ "$CURRENT_BRANCH" = "main" ]; then
    DEPLOY_DIR="."
    echo -e "${GREEN}Deploying to root (main branch)${NC}"
else
    DEPLOY_DIR="$CURRENT_BRANCH"
    echo -e "${GREEN}Deploying to /$DEPLOY_DIR/ subdirectory${NC}"
fi

# Verify we're in repo root
cd "$REPO_ROOT"

# Check for uncommitted changes (warning only)
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    echo -e "${YELLOW}The deployed docs will reflect the current working directory${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean previous build
echo -e "${GREEN}Cleaning previous build...${NC}"
cd "$DOCS_DIR"
make clean

# Build documentation
echo -e "${GREEN}Building documentation...${NC}"
make html

# Return to repo root
cd "$REPO_ROOT"

# Check if build succeeded
if [ ! -f "$BUILD_DIR/index.html" ]; then
    echo -e "${RED}Error: Build failed - index.html not found${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful: $(du -sh $BUILD_DIR | cut -f1)${NC}"

# Add .nojekyll file (required for GitHub Pages to serve _static directories)
touch "$BUILD_DIR/.nojekyll"

# Stash current changes if any
STASH_NAME="docs-deploy-$(date +%s)"
git stash push -m "$STASH_NAME" || true

# Check if gh-pages branch exists
echo -e "${GREEN}Preparing gh-pages branch...${NC}"
if git show-ref --verify --quiet refs/heads/$GH_PAGES_BRANCH; then
    echo "gh-pages branch exists, checking out..."
    git checkout $GH_PAGES_BRANCH
else
    echo "Creating new gh-pages branch..."
    git checkout --orphan $GH_PAGES_BRANCH
    git rm -rf .
    echo "# GitHub Pages for OMatG Documentation" > README.md
    git add README.md
    git commit -m "Initialize gh-pages branch"
fi

# Pull latest gh-pages from remote (if exists)
git pull origin $GH_PAGES_BRANCH --ff-only 2>/dev/null || echo "No remote gh-pages branch yet"

# Create deployment directory if needed
if [ "$DEPLOY_DIR" != "." ]; then
    mkdir -p "$DEPLOY_DIR"
    # Remove old content in this subdirectory
    rm -rf "$DEPLOY_DIR"/*
else
    # For main branch, remove all HTML content but preserve subdirectories
    find . -maxdepth 1 -type f -name "*.html" -delete
    find . -maxdepth 1 -type f -name "*.js" -delete
    rm -rf _static _sources _modules api user_guide getting_started development
fi

# Copy built HTML to deployment location
echo -e "${GREEN}Copying built HTML to gh-pages branch...${NC}"
if [ "$DEPLOY_DIR" = "." ]; then
    cp -r "$REPO_ROOT/$BUILD_DIR"/* .
else
    cp -r "$REPO_ROOT/$BUILD_DIR"/* "$DEPLOY_DIR/"
fi

# Add all changes
git add -A

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}No changes to deploy${NC}"
    git checkout "$CURRENT_BRANCH"
    # Restore stashed changes if any
    if git stash list | grep -q "$STASH_NAME"; then
        git stash pop
    fi
    exit 0
fi

# Get the commit hash before switching branches (for commit message)
ORIGINAL_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Commit changes
COMMIT_MSG="Deploy docs from $CURRENT_BRANCH branch

Built on $(date)
Source commit: $ORIGINAL_COMMIT"

git commit -m "$COMMIT_MSG"

# Show what will be pushed
echo -e "${GREEN}Changes ready to push:${NC}"
git log --oneline -1

# Ask for confirmation before pushing
echo -e "${YELLOW}Ready to push to origin/$GH_PAGES_BRANCH${NC}"
read -p "Push to remote? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin $GH_PAGES_BRANCH
    echo -e "${GREEN}Deployment successful!${NC}"

    # Show deployment URL
    REPO_URL=$(git config --get remote.origin.url)
    REPO_NAME=$(basename -s .git "$REPO_URL")
    REPO_USER=$(echo "$REPO_URL" | sed 's/.*github.com[:/]\(.*\)\/.*/\1/')

    if [ "$DEPLOY_DIR" = "." ]; then
        DOCS_URL="https://$REPO_USER.github.io/$REPO_NAME/"
    else
        DOCS_URL="https://$REPO_USER.github.io/$REPO_NAME/$DEPLOY_DIR/"
    fi

    echo -e "${GREEN}Documentation will be available at:${NC}"
    echo -e "${GREEN}$DOCS_URL${NC}"
    echo -e "${YELLOW}(may take a few minutes to deploy)${NC}"
else
    echo -e "${YELLOW}Push cancelled${NC}"
fi

# Return to original branch
git checkout "$CURRENT_BRANCH"

# Restore stashed changes if any
if git stash list | grep -q "$STASH_NAME"; then
    git stash pop
fi

echo -e "${GREEN}Done!${NC}"
