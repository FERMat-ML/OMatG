# Documentation Deployment Guide

This guide explains how to build and deploy the OMatG Sphinx documentation to GitHub Pages.

## Overview

The documentation uses a **pre-build deployment strategy**:

1. **Build locally** (where torch is installed)
2. **Push to gh-pages branch** using helper script
3. **GitHub Action deploys** automatically from gh-pages

## Prerequisites

- Python environment with torch and dependencies installed
- Access to push to the repository

## Quick Start

### Deploy Main Documentation (to root)

```bash
# From repository root
./scripts/deploy_docs.sh main
```

This deploys to: `https://fermat-ml.github.io/OMatG/`

### Deploy Feature Branch Documentation

```bash
# Switch to your feature branch
git checkout sphinx_docs

# Deploy (creates /sphinx_docs/ subdirectory)
./scripts/deploy_docs.sh
```

This deploys to: `https://fermat-ml.github.io/OMatG/sphinx_docs/`

## Detailed Workflow

### 1. Build Documentation Locally

The build process requires torch and ML dependencies to run autodoc:

```bash
cd docs
make clean html
```

Verify the build:
```bash
# Check size
du -sh build/html/

# Preview locally
python -m http.server --directory build/html 8000
# Visit http://localhost:8000
```

### 2. Deploy with Helper Script

The `scripts/deploy_docs.sh` script automates the gh-pages deployment:

```bash
# Deploy current branch
./scripts/deploy_docs.sh

# Deploy specific branch
./scripts/deploy_docs.sh main
./scripts/deploy_docs.sh sphinx_docs
```

**What the script does:**
1. Builds documentation with `make clean html`
2. Checks out/creates gh-pages branch
3. Copies built HTML to appropriate directory:
   - `main` branch → root (`/`)
   - Other branches → subdirectory (`/branch-name/`)
4. Commits and pushes to gh-pages
5. Returns you to your original branch

### 3. GitHub Action Deploys Automatically

When you push to gh-pages, the GitHub Action (`.github/workflows/deploy-docs.yml`) automatically:
1. Checks out the gh-pages branch
2. Uploads it to GitHub Pages
3. Deploys (takes 1-3 minutes)

Monitor deployment: https://github.com/FERMat-ML/OMatG/actions

## Branch-Specific Deployments

### Main Branch (Production)

```bash
git checkout main
./scripts/deploy_docs.sh main
```

→ Deploys to `https://fermat-ml.github.io/OMatG/`

### Feature Branches (Preview)

```bash
git checkout sphinx_docs
./scripts/deploy_docs.sh sphinx_docs
```

→ Deploys to `https://fermat-ml.github.io/OMatG/sphinx_docs/`

**Important:** Feature branch deployments are preserved until manually removed. Multiple feature branches can coexist:

```
https://fermat-ml.github.io/OMatG/               # main
https://fermat-ml.github.io/OMatG/sphinx_docs/
https://fermat-ml.github.io/OMatG/new-feature/
```

## Testing on Personal Fork

Before deploying to the main repository, test on your fork:

### 1. Fork the Repository

Create a fork: https://github.com/FERMat-ML/OMatG/fork

### 2. Clone Your Fork

```bash
git clone git@github.com:YOUR_USERNAME/OMatG.git
cd OMatG
```

### 3. Enable GitHub Pages on Fork

1. Go to fork settings: `https://github.com/YOUR_USERNAME/OMatG/settings/pages`
2. Under "Source", select:
   - **Branch:** `gh-pages`
   - **Folder:** `/ (root)`
3. Click **Save**

### 4. Deploy to Your Fork

```bash
# Build and deploy
./scripts/deploy_docs.sh main
```

Wait 2-3 minutes, then visit: `https://YOUR_USERNAME.github.io/OMatG/`

### 5. Test Feature Branch Deployment

```bash
git checkout -b test-branch
# Make some doc changes
./scripts/deploy_docs.sh test-branch
```

Visit: `https://YOUR_USERNAME.github.io/OMatG/test-branch/`

### 6. Verify Everything Works

- [ ] Main docs appear at root URL
- [ ] Feature branch docs appear at subdirectory URL
- [ ] Both versions coexist without conflicts
- [ ] All links work (check navigation)
- [ ] Static files load (CSS, JS, images)
- [ ] Search works

## Removing Old Deployments

To remove a feature branch deployment from gh-pages:

```bash
# Checkout gh-pages
git checkout gh-pages

# Remove the subdirectory
rm -rf sphinx_docs/

# Commit and push
git add -A
git commit -m "Remove sphinx_docs deployment"
git push origin gh-pages

# Return to your working branch
git checkout main
```

## Troubleshooting

### Build Fails: Import Errors

**Problem:** `make html` fails with import errors

**Solution:** Ensure you're in an environment with torch installed:
```bash
python -c "import torch; print(torch.__version__)"
pip install -e .
```

### Deployment URL Shows 404

**Problem:** Docs URL returns 404

**Causes:**
1. GitHub Pages not enabled (check Settings → Pages)
2. Deployment still in progress (wait 2-3 minutes)
3. Wrong branch selected in Pages settings (should be `gh-pages`)

**Solution:**
1. Verify settings: `https://github.com/USER/REPO/settings/pages`
2. Check deployment status: `https://github.com/USER/REPO/actions`

### CSS/JS Not Loading (Broken Styling)

**Problem:** Page loads but has no styling

**Cause:** Missing `.nojekyll` file

**Solution:** The script adds this automatically, but verify:
```bash
git checkout gh-pages
ls -la .nojekyll  # Should exist
```

If missing:
```bash
touch .nojekyll
git add .nojekyll
git commit -m "Add .nojekyll for GitHub Pages"
git push origin gh-pages
```

### Feature Branch Overwrites Main Docs

**Problem:** Deploying a feature branch replaces root documentation

**Cause:** Deployed with `./scripts/deploy_docs.sh main` while on feature branch

**Solution:** Always deploy with correct branch name:
```bash
# Check current branch
git branch --show-current

# Deploy with correct target
./scripts/deploy_docs.sh $(git branch --show-current)
```

### Script Fails: "cannot push to protected branch"

**Problem:** Push to gh-pages fails with permission error

**Cause:** Branch protection rules or insufficient permissions

**Solution:**
1. Check repository settings: Settings → Branches
2. Ensure gh-pages is not protected (or add yourself to allowed pushers)
3. Verify you have write access to the repository

## Manual Deployment (Alternative)

If the script doesn't work, you can deploy manually:

```bash
# 1. Build docs
cd docs
make clean html
cd ..

# 2. Install ghp-import (if needed)
pip install ghp-import

# 3. Deploy to gh-pages
ghp-import -n -p -f -m "Deploy docs" -b gh-pages docs/build/html

# -n: include .nojekyll
# -p: push to remote
# -f: force push
# -m: commit message
# -b: target branch
```

**Note:** This deploys to root only (doesn't support subdirectories).

## Best Practices

1. **Always test locally first:** Run `make html` and preview before deploying
2. **Use your fork for testing:** Test workflow changes on your fork before mainline
3. **Deploy main branch last:** Deploy feature branches first to test subdirectory structure
4. **Clean up old branches:** Remove merged feature branch deployments from gh-pages
5. **Monitor Actions tab:** Check deployment succeeded before sharing links

## Architecture Reference

```
Workflow:
  1. Developer: edit docs/source/*.rst
  2. Developer: ./scripts/deploy_docs.sh
     ├─ make clean html (builds docs locally)
     ├─ git checkout gh-pages
     ├─ cp build/html/* → gh-pages/
     └─ git push origin gh-pages
  3. GitHub Action: triggered by gh-pages push
     └─ Deploy gh-pages → GitHub Pages
  4. Result: https://username.github.io/repo/

Directory Structure on gh-pages:
  /                    # Main docs (from main branch)
  ├── index.html
  ├── _static/
  ├── api/
  └── ...
  /sphinx_docs/        # Feature branch docs
  ├── index.html
  └── ...
  /another-branch/     # Another feature branch
```

## Related Files

- `scripts/deploy_docs.sh` - Build and deployment script
- `.github/workflows/deploy-docs.yml` - GitHub Action for deployment
- `docs/Makefile` - Sphinx build configuration
- `docs/source/conf.py` - Sphinx documentation configuration

## Support

For issues with deployment:
1. Check Actions tab: https://github.com/FERMat-ML/OMatG/actions
2. Review workflow file: `.github/workflows/deploy-docs.yml`
3. Check gh-pages branch: https://github.com/FERMat-ML/OMatG/tree/gh-pages
4. Verify Pages settings: https://github.com/FERMat-ML/OMatG/settings/pages
