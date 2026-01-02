============
Contributing
============

We welcome contributions to OMatG!

Ways to Contribute
===================

* Report bugs and issues
* Suggest new features
* Improve documentation
* Submit code contributions
* Share example workflows
* Help answer questions

Reporting Issues
================

When reporting bugs, please include:

1. **Description**: Clear description of the problem
2. **Steps to reproduce**: Minimal example that reproduces the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:

   * Python version
   * PyTorch version
   * OMatG version
   * Operating system

Example:

.. code-block:: markdown

   **Description**: Training crashes with NaN loss

   **Steps to reproduce**:
   ```bash
   omg fit --config=config.yaml
   ```

   **Expected**: Training completes successfully

   **Actual**: NaN loss after epoch 10

   **Environment**:
   - Python 3.11
   - PyTorch 2.8.0
   - OMatG 1.2.0
   - Ubuntu 22.04

Code Contributions
==================

Setting Up Development Environment
-----------------------------------

1. Fork the repository on GitHub

2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/OMatG.git
      cd OMatG

3. Add upstream remote:

   .. code-block:: bash

      git remote add upstream https://github.com/FerMat-ML/OMatG.git

4. Create virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/macOS
      # venv\Scripts\activate  # Windows

5. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

Making Changes
--------------

1. **Sync with upstream**:

   .. code-block:: bash

      git checkout main
      git fetch upstream
      git merge upstream/main

2. **Create feature branch**:

   .. code-block:: bash

      git checkout -b feature/my-feature

3. **Make changes**: Edit code following style guide

4. **Add tests**: Cover new functionality

5. **Run tests**:

   .. code-block:: bash

      pytest tests/

6. **Format code**:

   .. code-block:: bash

      black omg/ tests/
      isort omg/ tests/

7. **Commit changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description"

8. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/my-feature

9. **Create pull request** on GitHub

Pull Request Guidelines
------------------------

* **Title**: Clear, descriptive title
* **Description**: Explain what and why
* **Tests**: Include tests for new features
* **Documentation**: Update relevant docs
* **Style**: Follow code style guidelines
* **Atomic**: One feature/fix per PR

PR Template:

.. code-block:: markdown

   ## Description
   Brief description of changes.

   ## Motivation
   Why is this change needed?

   ## Changes
   - List of changes
   - Another change

   ## Tests
   - [ ] Added tests for new functionality
   - [ ] All tests pass

   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated user guide (if applicable)

   ## Checklist
   - [ ] Code follows style guide
   - [ ] Tests pass
   - [ ] Documentation updated

Code Style Guidelines
=====================

Python Style
------------

Follow PEP 8 with these specifics:

* **Line length**: 88 characters (Black default)
* **Imports**: Sorted with isort
* **Quotes**: Double quotes for strings
* **Type hints**: For public APIs

Formatting Tools
----------------

.. code-block:: bash

   # Format code
   black omg/ tests/

   # Sort imports
   isort omg/ tests/

   # Check style (optional)
   flake8 omg/ tests/

Naming Conventions
------------------

* **Classes**: ``PascalCase``
* **Functions/methods**: ``snake_case``
* **Constants**: ``UPPER_SNAKE_CASE``
* **Private**: Prefix with ``_``

.. code-block:: python

   class MyClass:
       """Class docstring."""

       CONSTANT = 42

       def __init__(self):
           self._private_var = None

       def public_method(self, param: int) -> str:
           """Method docstring."""
           return str(param)

       def _private_method(self):
           """Private helper."""
           pass

Docstring Style
---------------

Use Google or NumPy style:

**Google Style**:

.. code-block:: python

   def function(param1: int, param2: str) -> bool:
       """Short description.

       Longer description.

       Args:
           param1: Description of param1.
           param2: Description of param2.

       Returns:
           Description of return value.

       Raises:
           ValueError: When param1 is negative.

       Example:
           >>> function(1, "test")
           True
       """
       pass

**NumPy Style**:

.. code-block:: python

   def function(param1, param2):
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

       Examples
       --------
       >>> function(1, "test")
       True
       """
       pass

Testing Guidelines
==================

Test Structure
--------------

Place tests in ``tests/`` mirroring package structure:

.. code-block:: text

   tests/
   ├── test_si.py
   ├── test_sampler.py
   ├── test_datamodule.py
   ├── test_model.py
   └── test_analysis.py

Writing Tests
-------------

.. code-block:: python

   import pytest
   import torch
   from omg.sampler import IndependentSampler

   def test_sampler_creation():
       """Test sampler can be created."""
       sampler = IndependentSampler(...)
       assert sampler is not None

   def test_sampler_output_shape():
       """Test sampler produces correct shapes."""
       sampler = IndependentSampler(...)
       data = ...
       output = sampler.sample(data)
       assert output.shape == expected_shape

   def test_sampler_invalid_input():
       """Test sampler raises on invalid input."""
       sampler = IndependentSampler(...)
       with pytest.raises(ValueError):
           sampler.sample(invalid_data)

Fixtures
--------

Use pytest fixtures for common setups:

.. code-block:: python

   import pytest

   @pytest.fixture
   def dummy_structure():
       """Create dummy structure for testing."""
       return torch.randn(10, 3)

   def test_with_fixture(dummy_structure):
       """Test using fixture."""
       assert dummy_structure.shape == (10, 3)

Running Tests
-------------

.. code-block:: bash

   # All tests
   pytest

   # Specific file
   pytest tests/test_si.py

   # Specific test
   pytest tests/test_si.py::test_linear_interpolant

   # With coverage
   pytest --cov=omg --cov-report=html

   # Verbose
   pytest -v

Documentation Contributions
============================

Documentation is as important as code!

Types of Documentation
----------------------

* **User guides**: Conceptual explanations (``docs/source/user_guide/``)
* **API docs**: Auto-generated from docstrings
* **Tutorials**: Step-by-step examples (``examples/``)
* **README**: Project overview

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make clean html
   open build/html/index.html

Adding Documentation
--------------------

1. **User guide page**:

   Create ``docs/source/user_guide/new_topic.rst``

2. **Add to toctree** in ``docs/source/user_guide/index.rst``

3. **Build and check**:

   .. code-block:: bash

      make clean html

Example Structure
-----------------

.. code-block:: rst

   ==========
   Topic Name
   ==========

   Brief introduction.

   Overview
   ========

   Detailed explanation.

   Example
   =======

   .. code-block:: python

      # Code example
      from omg import something

   Best Practices
   ==============

   * Tip 1
   * Tip 2

   See Also
   ========

   * :doc:`related_topic`
   * :doc:`../api/module`

Review Process
==============

All contributions go through review:

1. **Automated checks**: CI runs tests and style checks
2. **Maintainer review**: Code quality, design, tests
3. **Discussion**: Feedback and requested changes
4. **Approval**: Maintainer approves
5. **Merge**: Contribution merged to main

Be patient - reviews may take a few days.

Code of Conduct
===============

* Be respectful and inclusive
* Welcome newcomers
* Accept constructive criticism
* Focus on what's best for the community

Recognition
===========

Contributors are recognized in:

* GitHub contributors page
* AUTHORS file
* Release notes

Thank You
=========

Thank you for contributing to OMatG!

Your contributions help make materials discovery accessible to everyone.
