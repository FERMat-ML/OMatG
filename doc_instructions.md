# Sphinx Documentation Setup Guide

This guide provides detailed instructions for setting up and customizing Sphinx documentation using the PyData Sphinx Theme, based on the configuration in `docs/source/conf.py`.

## Table of Contents
1. [Installation](#installation)
2. [Configuration File Overview](#configuration-file-overview)
3. [Understanding RST Syntax](#understanding-rst-syntax)
4. [Document Structure and Navigation](#document-structure-and-navigation)
5. [PyData Theme Customization](#pydata-theme-customization)
6. [Auto-generating API Documentation](#auto-generating-api-documentation)
7. [Creating Custom Pages](#creating-custom-pages)
8. [Adding Static Assets](#adding-static-assets)
9. [Custom Templates](#custom-templates)
10. [Building Documentation](#building-documentation)
11. [Hosting on GitHub Pages](#hosting-on-github-pages)
12. [Advanced Customization](#advanced-customization)

---

## Installation

Install Sphinx and required extensions:

```bash
pip install sphinx pydata-sphinx-theme sphinx-copybutton
```

Key packages:
- `sphinx`: Core documentation generator
- `pydata-sphinx-theme`: Modern, responsive theme
- `sphinx-copybutton`: Adds copy button to code blocks

---

## Configuration File Overview

The `docs/source/conf.py` file controls all Sphinx behavior. Key sections:

### Project Information
```python
project = "YourProjectName"
copyright = "2025, Your Name"
author = "Your Name"
release = "1.0.0"
```

### Extensions
```python
extensions = [
    "sphinx.ext.autodoc",        # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",       # Support Google/NumPy style docstrings
    "sphinx.ext.viewcode",       # Add links to source code
    "sphinx.ext.intersphinx",    # Link to other project docs
    "sphinx.ext.autosummary",    # Generate summary tables
    "sphinx_copybutton",         # Copy button for code blocks
]
```

### Path Configuration
```python
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))  # Points to project root
```
This ensures Sphinx can import your Python modules.

---

## Understanding RST Syntax

ReStructuredText (RST) is Sphinx's markup language. Essential syntax:

### Headings
```rst
##################
Top-level Heading (with overline)
##################

Section Heading
===============

Subsection Heading
------------------

Subsubsection Heading
~~~~~~~~~~~~~~~~~~~~~

Paragraph Heading
^^^^^^^^^^^^^^^^^
```

**Important**: Underline (and overline) must be at least as long as the text.

### Text Formatting
```rst
*italic text*
**bold text**
``inline code``
```

### Links
```rst
External link: `Link Text <https://example.com>`__
Internal reference: :ref:`label-name`
Document reference: :doc:`other_document`
```

### Lists
```rst
Unordered list:

- Item 1
- Item 2

  - Nested item

Ordered list:

1. First item
2. Second item
```

### Code Blocks
```rst
.. code-block:: python

   def example():
       return "Hello"
```

### Admonitions
```rst
.. note::
   This is a note.

.. warning::
   This is a warning.

.. tip::
   This is a tip.
```

---

## Document Structure and Navigation

### The Table of Contents Tree (toctree)

The `toctree` directive creates navigation structure:

#### Basic toctree
```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
```

#### toctree Options
- `:maxdepth: N` - How many heading levels to show (1-4)
- `:caption: Text` - Section heading in sidebar
- `:hidden:` - Hide from page but include in navigation
- `:numbered:` - Number sections
- `:titlesonly:` - Show only document titles, not subsections

#### Creating Hierarchical Navigation

**index.rst** (main page):
```rst
Welcome to MyProject
====================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/quickstart
   guide/tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
```

**guide/installation.rst**:
```rst
Installation
============

Content here...
```

### Cross-referencing

```rst
Link to section in same file:
`Section Name`_

Link to section in different file:
:ref:`custom-label`

Then define the label before the section:
.. _custom-label:

Section Heading
===============
```

---

## PyData Theme Customization

The PyData Sphinx Theme offers extensive customization options.

### Basic Theme Configuration

Add to `conf.py`:

```python
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "My Project",
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/username/repo",
            "icon": "fab fa-github-square",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/myproject/",
            "icon": "fas fa-box",
        },
    ],
}
```

### Navigation Bar Customization

```python
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],

    # Persistent right sidebar
    "navbar_persistent": ["search-button"],

    # Footer content
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
```

### Adding Navigation Links

```python
html_theme_options = {
    "external_links": [
        {"name": "Documentation", "url": "https://docs.example.com"},
        {"name": "Community", "url": "https://community.example.com"},
    ],
}
```

### Sidebar Configuration

```python
html_theme_options = {
    "show_toc_level": 2,  # Depth of TOC in right sidebar
    "navigation_depth": 4,  # Depth in left sidebar
    "collapse_navigation": False,  # Keep navigation expanded
    "show_prev_next": True,  # Previous/Next buttons
}
```

### Search Configuration

```python
html_theme_options = {
    "search_bar_text": "Search the docs...",
    "search_bar_position": "navbar",  # or "sidebar"
}
```

### Announcement Banners

```python
html_theme_options = {
    "announcement": "🎉 Version 2.0 released! <a href='/changelog'>See what's new</a>",
}
```

### Custom CSS

Add to `conf.py`:
```python
html_static_path = ["_static"]
html_css_files = ["custom.css"]
```

Create `docs/source/_static/custom.css`:
```css
/* Customize colors */
:root {
    --pst-color-primary: #4051b5;
    --pst-color-secondary: #6c757d;
}

/* Custom heading styles */
h1 {
    color: var(--pst-color-primary);
    border-bottom: 2px solid var(--pst-color-primary);
}

/* Customize admonitions */
.admonition.tip {
    border-left: 4px solid #28a745;
}
```

### Buttons and Cards

In RST files:

```rst
.. button-link:: https://example.com
   :color: primary
   :outline:

   Click Me

.. grid:: 2

   .. grid-item-card:: Feature 1
      :link: feature1
      :link-type: doc

      Description of feature 1

   .. grid-item-card:: Feature 2
      :link: feature2
      :link-type: doc

      Description of feature 2
```

Requires:
```bash
pip install sphinx-design
```

Add to `conf.py`:
```python
extensions = [
    # ... other extensions
    "sphinx_design",
]
```

---

## Auto-generating API Documentation

### Using sphinx-apidoc

Generate RST files for all Python modules:

```bash
cd docs
sphinx-apidoc -f -o source/ ../your_package/
```

Options:
- `-f`: Force overwrite existing files
- `-o source/`: Output directory
- `../your_package/`: Package to document

### Autodoc Directives

Manual module documentation in RST:

```rst
Module Name
===========

.. automodule:: mypackage.mymodule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

Options:
- `:members:` - Document all members
- `:members: func1, func2` - Document specific members
- `:undoc-members:` - Include members without docstrings
- `:show-inheritance:` - Show base classes
- `:private-members:` - Include private members (_name)
- `:special-members:` - Include special methods (__init__, etc.)

### Document Specific Items

```rst
Classes
-------

.. autoclass:: mypackage.MyClass
   :members:
   :inherited-members:
   :member-order: bysource

Functions
---------

.. autofunction:: mypackage.my_function

Exceptions
----------

.. autoexception:: mypackage.MyException
```

### Autosummary for Overview Tables

```rst
API Reference
=============

.. autosummary::
   :toctree: generated
   :recursive:

   mypackage.module1
   mypackage.module2
```

This generates clickable tables with links to detailed docs.

Add to `conf.py`:
```python
autosummary_generate = True
```

### Napoleon - Google/NumPy Docstrings

Napoleon allows you to write docstrings in Google or NumPy format instead of RST.

**Google Style**:
```python
def example_function(param1, param2):
    """Short description.

    Longer description explaining the function.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        >>> example_function(1, "test")
        True
    """
```

**NumPy Style**:
```python
def example_function(param1, param2):
    """
    Short description.

    Longer description.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.
    """
```

Configure in `conf.py`:
```python
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
```

---

## Creating Custom Pages

### Tutorial/Guide Pages

Create `docs/source/tutorials/getting_started.rst`:

```rst
Getting Started
===============

This tutorial will help you get started with MyProject.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install myproject

Quick Example
-------------

Here's a simple example:

.. code-block:: python

   from myproject import MyClass

   # Create instance
   obj = MyClass()
   obj.do_something()

Expected output:

.. code-block:: text

   Success!

What's Next?
------------

Continue to:

- :doc:`advanced_tutorial`
- :doc:`../api/index`
```

### Adding to Navigation

In `index.rst`:
```rst
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/getting_started
   tutorials/advanced_tutorial
```

### Multi-column Layouts

```rst
.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Get MyProject installed

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      5-minute introduction

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete API documentation
```

### Tabs

```rst
.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         print("Hello")

   .. tab-item:: JavaScript

      .. code-block:: javascript

         console.log("Hello");
```

Requires `sphinx-design` extension.

---

## Adding Static Assets

### Images

Place images in `docs/source/_static/images/`.

In RST:
```rst
.. image:: /_static/images/diagram.png
   :alt: Architecture diagram
   :width: 600px
   :align: center

Or with caption:

.. figure:: /_static/images/screenshot.png
   :alt: Screenshot
   :width: 80%

   Figure 1: Application screenshot
```

### Custom JavaScript

Create `docs/source/_static/custom.js`:
```javascript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Custom JS loaded');
    // Your custom code
});
```

Add to `conf.py`:
```python
html_js_files = ["custom.js"]
```

### Favicon

```python
html_favicon = "_static/favicon.ico"
```

### Logo

```python
html_logo = "_static/logo.png"

# Or in theme options for light/dark versions:
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
}
```

---

## Custom Templates

Templates control page layout and structure.

### Override Default Templates

PyData theme templates are in `pydata_sphinx_theme/layout.html`, etc.

Create custom template `docs/source/_templates/page.html`:

```html
{% extends "!page.html" %}

{% block body %}
<div class="custom-banner">
  <p>Custom content above page</p>
</div>
{{ super() }}
{% endblock %}
```

### Custom Sidebar

`docs/source/_templates/sidebar/custom-sidebar.html`:
```html
<div class="sidebar-custom-section">
  <h3>Quick Links</h3>
  <ul>
    <li><a href="{{ pathto('quickstart') }}">Quickstart</a></li>
    <li><a href="{{ pathto('faq') }}">FAQ</a></li>
  </ul>
</div>
```

Reference in `conf.py`:
```python
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "custom-sidebar"],
}
```

### Footer Template

`docs/source/_templates/footer.html`:
```html
<footer class="bd-footer">
  <div class="bd-footer__inner container">
    <div class="footer-items">
      <div class="footer-item">
        <p>© 2025 Your Project. All rights reserved.</p>
      </div>
    </div>
  </div>
</footer>
```

---

## Building Documentation

### Build Commands

From `docs/` directory:

```bash
# Build HTML
make html

# Clean previous builds
make clean

# Full rebuild
make clean html

# On Windows:
make.bat html
```

### View Documentation

Open `docs/build/html/index.html` in browser.

### Live Rebuild (Auto-rebuild on changes)

Install:
```bash
pip install sphinx-autobuild
```

Run:
```bash
sphinx-autobuild docs/source docs/build/html
```

Opens browser at `http://127.0.0.1:8000` with auto-reload.

### Build Other Formats

```bash
make latex      # LaTeX/PDF
make epub       # EPUB e-book
make man        # Man pages
make texinfo    # Texinfo files
```

---

## Hosting on GitHub Pages

GitHub Pages provides free hosting for static websites, making it perfect for Sphinx documentation. Your docs will be accessible at `https://username.github.io/projectname/` or `https://projectname.github.io/` for organization pages.

### Option 1: Automated Deployment with GitHub Actions (Recommended)

This is the easiest and most maintainable approach. GitHub Actions automatically builds and deploys your docs whenever you push changes.

#### Step 1: Create GitHub Actions Workflow

Create `.github/workflows/docs.yml` in your repository:

```yaml
name: Build and Deploy Sphinx Docs

on:
  push:
    branches:
      - main  # or master, depending on your default branch
  pull_request:
    branches:
      - main

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install sphinx pydata-sphinx-theme sphinx-copybutton
          # Install your project dependencies if needed for autodoc
          pip install -e .

      - name: Build documentation
        run: |
          cd docs
          make clean html

      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/build/html
          cname: docs.yourdomain.com  # Optional: for custom domain
```

#### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages** (in the left sidebar)
3. Under **Source**, select:
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
4. Click **Save**

#### Step 3: Commit and Push

```bash
git add .github/workflows/docs.yml
git commit -m "Add GitHub Actions workflow for documentation"
git push
```

The workflow will automatically:
- Build your Sphinx documentation
- Create/update the `gh-pages` branch
- Deploy to GitHub Pages

#### Step 4: Access Your Documentation

After the action completes (check the Actions tab on GitHub), your docs will be available at:
- `https://username.github.io/repository-name/`

### Option 2: Manual Deployment with ghp-import

For manual control over deployments:

#### Install ghp-import

```bash
pip install ghp-import
```

#### Build and Deploy

```bash
# Build documentation
cd docs
make clean html

# Deploy to gh-pages branch
ghp-import -n -p -f build/html
```

Options:
- `-n`: Include a `.nojekyll` file (required for Sphinx)
- `-p`: Push to remote repository
- `-f`: Force push (overwrite existing gh-pages branch)

### Option 3: Deploy from a Subdirectory

If you want to keep built docs in your main branch (not recommended for large projects):

#### Build to docs/ Directory

Modify `conf.py` or build to a specific location:

```bash
sphinx-build -b html docs/source docs
```

#### Configure GitHub Pages

1. Settings → Pages
2. Source: **main branch** → **/docs folder**
3. Save

**Note**: You'll need to commit the built HTML files to your repository, which can bloat your repo size.

### Configuration for GitHub Pages

#### Add .nojekyll File

GitHub Pages uses Jekyll by default, which ignores directories starting with `_`. Sphinx uses `_static` and `_templates`, so you need to disable Jekyll.

The GitHub Actions workflow and ghp-import automatically add this file. If deploying manually, add an empty `.nojekyll` file:

```bash
touch docs/build/html/.nojekyll
```

#### Configure Base URL

If your docs are at `username.github.io/project/` (not root), update `conf.py`:

```python
# For project pages (username.github.io/projectname/)
html_baseurl = 'https://username.github.io/projectname/'

# For organization pages (projectname.github.io/)
html_baseurl = 'https://projectname.github.io/'
```

This ensures all links work correctly.

#### Update Links in Theme

```python
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/username/projectname",
            "icon": "fab fa-github-square",
        },
    ],
}
```

### Custom Domain Setup

To use a custom domain like `docs.yourdomain.com`:

#### Step 1: Configure DNS

Add a CNAME record in your domain registrar:

```
docs.yourdomain.com  →  username.github.io
```

#### Step 2: Add CNAME File

Add to GitHub Actions workflow (already shown above):

```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./docs/build/html
    cname: docs.yourdomain.com
```

Or manually create `docs/build/html/CNAME`:

```
docs.yourdomain.com
```

#### Step 3: Configure in GitHub

1. Settings → Pages
2. Under **Custom domain**, enter: `docs.yourdomain.com`
3. Check **Enforce HTTPS** (recommended)

### Multi-version Documentation

Host multiple documentation versions:

#### Directory Structure

```
gh-pages branch:
├── index.html (redirect to latest)
├── latest/
│   └── (current docs)
├── v2.0/
│   └── (v2.0 docs)
└── v1.0/
    └── (v1.0 docs)
```

#### GitHub Actions for Multi-version

```yaml
name: Build and Deploy Sphinx Docs

on:
  push:
    branches:
      - main
    tags:
      - 'v*'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for all tags

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install sphinx pydata-sphinx-theme sphinx-copybutton
          pip install -e .

      - name: Determine version
        id: version
        run: |
          if [[ $GITHUB_REF == refs/tags/* ]]; then
            VERSION=${GITHUB_REF#refs/tags/}
          else
            VERSION="latest"
          fi
          echo "VERSION=$VERSION" >> $GITHUB_OUTPUT

      - name: Build documentation
        run: |
          cd docs
          make clean html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/build/html
          destination_dir: ${{ steps.version.outputs.VERSION }}
          keep_files: true
```

#### Version Switcher

Configure in `conf.py`:

```python
html_theme_options = {
    "switcher": {
        "json_url": "https://yourusername.github.io/yourproject/_static/switcher.json",
        "version_match": version,
    },
}
```

Create `docs/source/_static/switcher.json`:

```json
[
    {
        "name": "latest",
        "version": "latest",
        "url": "https://yourusername.github.io/yourproject/latest/"
    },
    {
        "name": "v2.0 (stable)",
        "version": "v2.0",
        "url": "https://yourusername.github.io/yourproject/v2.0/"
    },
    {
        "name": "v1.0",
        "version": "v1.0",
        "url": "https://yourusername.github.io/yourproject/v1.0/"
    }
]
```

### Troubleshooting GitHub Pages

#### Issue: 404 Page Not Found

**Causes**:
1. GitHub Pages not enabled
2. Wrong branch/folder selected
3. Deployment not complete

**Solutions**:
- Check Settings → Pages for correct configuration
- Wait a few minutes after pushing (initial deployment can take 5-10 minutes)
- Check Actions tab for deployment status

#### Issue: Broken Links and Missing CSS

**Cause**: Missing `.nojekyll` file

**Solution**: Ensure `.nojekyll` exists in the root of your gh-pages branch. GitHub Actions and ghp-import add this automatically.

Verify:
```bash
git checkout gh-pages
ls -la .nojekyll  # Should exist
```

#### Issue: Images/Static Files Not Loading

**Cause**: Incorrect paths or baseurl

**Solution**: Use absolute paths from root in RST:

```rst
.. image:: /_static/images/logo.png
```

And set `html_baseurl` in `conf.py`:

```python
html_baseurl = 'https://username.github.io/projectname/'
```

#### Issue: Search Not Working

**Cause**: Search requires JavaScript and proper CORS headers

**Solution**: GitHub Pages serves with correct headers by default. Ensure `search.html` and `searchindex.js` are present in built docs.

Check:
```bash
ls docs/build/html/search.html
ls docs/build/html/searchindex.js
```

#### Issue: Workflow Fails with Permission Error

**Error**: `refusing to allow a GitHub App to create or update workflow`

**Solution**: Update repository settings:

1. Settings → Actions → General
2. Under **Workflow permissions**, select **Read and write permissions**
3. Check **Allow GitHub Actions to create and approve pull requests**
4. Save

#### Issue: Changes Not Appearing

**Causes**:
1. Browser cache
2. GitHub Pages CDN cache
3. Build/deployment failed

**Solutions**:
- Hard refresh browser (Ctrl+F5 / Cmd+Shift+R)
- Check Actions tab for errors
- Wait 5-10 minutes for CDN propagation
- Clear browser cache

#### Issue: Import Errors During Build

**Cause**: Missing dependencies in GitHub Actions

**Solution**: Update workflow to install all dependencies:

```yaml
- name: Install dependencies
  run: |
    pip install -e .  # Install your package
    pip install -r docs/requirements.txt  # Or specific sphinx packages
```

Create `docs/requirements.txt`:

```
sphinx>=7.0.0
pydata-sphinx-theme>=0.14.0
sphinx-copybutton
sphinx-design
# Add other extensions
```

### Best Practices

1. **Use GitHub Actions**: Automate builds to avoid manual errors

2. **Pin Dependency Versions**: Use specific versions in requirements.txt for reproducible builds

3. **Test Locally First**: Always run `make clean html` locally before pushing

4. **Monitor Actions**: Check the Actions tab after pushing to ensure successful deployment

5. **Use PR Previews**: Build docs on pull requests to preview changes before merging

6. **Version Your Docs**: For major projects, maintain docs for multiple versions

7. **Custom Domain with HTTPS**: Enhance professionalism and security

8. **Add Status Badge**: Show build status in README:

```markdown
[![Documentation](https://github.com/username/repo/actions/workflows/docs.yml/badge.svg)](https://username.github.io/repo/)
```

### Alternative: Read the Docs

If GitHub Pages doesn't meet your needs, consider [Read the Docs](https://readthedocs.org/):

**Advantages**:
- Automatic version management
- PR previews
- Advanced analytics
- Custom themes and extensions

**Setup**:
1. Sign up at readthedocs.org
2. Connect your GitHub repository
3. Configure `.readthedocs.yaml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .

sphinx:
  configuration: docs/source/conf.py
```

---

## Advanced Customization

### Intersphinx - Link to Other Docs

Link to Python, NumPy, etc. documentation:

```python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
```

In RST:
```rst
See :py:func:`python:sorted` for details.
```

### Version Dropdown

Show multiple documentation versions:

```python
html_theme_options = {
    "switcher": {
        "json_url": "https://myproject.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": version,
    },
}
```

Create `_static/switcher.json`:
```json
[
    {"version": "latest", "url": "https://myproject.readthedocs.io/en/latest/"},
    {"version": "v2.0", "url": "https://myproject.readthedocs.io/en/v2.0/"},
    {"version": "v1.0", "url": "https://myproject.readthedocs.io/en/v1.0/"}
]
```

### Custom Directives

Create custom RST directive in `conf.py`:

```python
from docutils import nodes
from docutils.parsers.rst import Directive

class HighlightBox(Directive):
    has_content = True

    def run(self):
        text = '\n'.join(self.content)
        node = nodes.container(text)
        node['classes'].append('highlight-box')
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

def setup(app):
    app.add_directive("highlight", HighlightBox)
    app.add_css_file("custom.css")
```

Use in RST:
```rst
.. highlight::

   This content will be in a highlighted box!
```

Style in `_static/custom.css`:
```css
.highlight-box {
    border: 2px solid #ff6b6b;
    padding: 1rem;
    margin: 1rem 0;
    background-color: #ffe0e0;
}
```

### Custom Roles

Inline custom formatting:

```python
from docutils import nodes

def custom_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(rawtext, text, classes=['custom-role'])
    return [node], []

def setup(app):
    app.add_role('custom', custom_role)
```

Use:
```rst
This is :custom:`specially formatted` text.
```

### Code Syntax Highlighting

Configure Pygments:

```python
pygments_style = "sphinx"  # Light mode
pygments_dark_style = "monokai"  # Dark mode
```

Specify language in code blocks:
```rst
.. code-block:: python
   :linenos:
   :emphasize-lines: 2,3

   def example():
       important_line = 1
       another_important = 2
       return important_line + another_important
```

### Download Links

Make code blocks downloadable:

```rst
.. literalinclude:: ../../examples/example.py
   :language: python
   :linenos:
   :download:
```

### Conditional Content

Show content only in HTML:

```rst
.. only:: html

   This appears only in HTML output.

   .. figure:: /_static/interactive_plot.html
      :align: center
```

### Math Support

Add extension:
```python
extensions = [
    # ...
    "sphinx.ext.mathjax",
]
```

Write equations:
```rst
Inline: :math:`E = mc^2`

Display:

.. math::

   \int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
```

### Include External Files

```rst
.. include:: ../CHANGELOG.md
   :parser: markdown
```

Requires:
```bash
pip install myst-parser
```

Add to `conf.py`:
```python
extensions = ["myst_parser"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
```

---

## Complete Configuration Example

Here's a fully customized `conf.py`:

```python
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# Project information
project = "MyProject"
copyright = "2025, Your Name"
author = "Your Name"
release = "2.0.0"
version = "2.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "logo": {
        "text": "MyProject Documentation",
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/username/myproject",
            "icon": "fab fa-github-square",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/myproject/",
            "icon": "fas fa-box",
        },
    ],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_prev_next": True,
    "search_bar_text": "Search documentation...",
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary
autosummary_generate = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Syntax highlighting
pygments_style = "sphinx"
pygments_dark_style = "monokai"
```

---

## Tips and Best Practices

1. **Start Simple**: Begin with basic configuration and add features as needed.

2. **Use autosummary**: For large APIs, autosummary tables provide better navigation than long pages.

3. **Organize Content**: Use clear hierarchy with packages → modules → classes → functions.

4. **Cross-reference Everything**: Link related functions, classes, and concepts.

5. **Include Examples**: Code examples in docstrings and tutorials are invaluable.

6. **Test Builds Regularly**: Run `make html` frequently to catch formatting errors.

7. **Version Your Docs**: Use git to track changes to RST files.

8. **Responsive Design**: PyData theme is mobile-friendly by default.

9. **Search Optimization**: Use clear headings and keywords for better search results.

10. **Deploy Continuously**: Use Read the Docs, GitHub Pages, or similar for automatic builds.

---

## Common Issues and Solutions

### Import Errors During Build

**Problem**: `WARNING: autodoc: failed to import module 'mypackage'`

**Solution**: Check `sys.path.insert()` in `conf.py` points to correct directory:
```python
sys.path.insert(0, os.path.abspath("../../"))
```

### Underline Too Short

**Problem**: `WARNING: Title underline too short`

**Solution**: Make underline exactly same length or longer than title:
```rst
My Title
========
```

### toctree Not Found

**Problem**: `WARNING: toctree contains reference to nonexistent document`

**Solution**: Verify file exists and path is correct (relative to current file, no `.rst` extension):
```rst
.. toctree::

   tutorials/quickstart  # Not quickstart.rst
```

### Theme Not Found

**Problem**: `sphinx.errors.ThemeError: no theme named 'pydata_sphinx_theme'`

**Solution**: Install theme:
```bash
pip install pydata-sphinx-theme
```

### CSS/JS Not Loading

**Problem**: Custom CSS/JS files not applied

**Solution**:
1. Check files are in `_static/` directory
2. Run `make clean html` to clear cache
3. Verify paths in `conf.py`:
```python
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # Not "_static/custom.css"
```

---

## Resources

### Sphinx and Theme Documentation
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [PyData Sphinx Theme Docs](https://pydata-sphinx-theme.readthedocs.io/)
- [RST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Sphinx Design](https://sphinx-design.readthedocs.io/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

### GitHub Pages and Deployment
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages)
- [ghp-import Tool](https://github.com/c-w/ghp-import)
- [Read the Docs](https://readthedocs.org/)

---

This guide covers the essentials and advanced features for creating professional, customizable Sphinx documentation with the PyData theme. Experiment with different options to find what works best for your project!
