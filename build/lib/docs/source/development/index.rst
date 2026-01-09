===========
Development
===========

Contributing to OMatG development.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Contributing Guide
        :link: contributing
        :link-type: doc

        How to contribute code, documentation, and bug reports

Development Setup
=================

Clone Repository
----------------

.. code-block:: bash

   git clone https://github.com/FerMat-ML/OMatG.git
   cd OMatG

Install in Development Mode
----------------------------

.. code-block:: bash

   pip install -e .

This installs OMatG in editable mode so code changes are immediately available.

Install Development Dependencies
---------------------------------

.. code-block:: bash

   pip install -e ".[dev]"

This includes testing, linting, and documentation tools.

Development Workflow
====================

1. **Create a branch**:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. **Make changes**: Edit code following the style guide

3. **Run tests**:

   .. code-block:: bash

      pytest tests/

4. **Format code**:

   .. code-block:: bash

      black omg/
      isort omg/

5. **Commit changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add new feature"

6. **Push and create PR**:

   .. code-block:: bash

      git push origin feature/my-new-feature

Project Structure
=================

.. code-block:: text

   OMatG/
   ├── omg/                    # Main package
   │   ├── si/                 # Stochastic interpolants
   │   ├── sampler/            # Base distributions
   │   ├── datamodule/         # Data loading
   │   ├── model/              # Neural networks
   │   ├── analysis/           # Validation and metrics
   │   ├── omg_lightning.py    # Lightning module
   │   ├── omg_trainer.py      # Trainer with metrics
   │   └── main.py             # CLI entry point
   ├── tests/                  # Test suite
   ├── docs/                   # Documentation
   ├── examples/               # Example scripts
   └── pyproject.toml          # Package configuration

Code Style
==========

OMatG follows:

* **PEP 8**: Python style guide
* **Black**: Code formatter
* **isort**: Import sorter
* **Type hints**: For public APIs

Format code:

.. code-block:: bash

   black omg/ tests/
   isort omg/ tests/

Testing
=======

Run Tests
---------

.. code-block:: bash

   # All tests
   pytest

   # Specific module
   pytest tests/test_si.py

   # With coverage
   pytest --cov=omg --cov-report=html

Write Tests
-----------

Place tests in ``tests/`` directory:

.. code-block:: python

   # tests/test_mymodule.py
   import pytest
   from omg.mymodule import MyClass

   def test_myclass():
       obj = MyClass()
       assert obj.method() == expected_result

   def test_myclass_error():
       with pytest.raises(ValueError):
           MyClass().invalid_method()

Documentation
=============

Build Documentation
-------------------

.. code-block:: bash

   cd docs
   make clean html

View documentation:

.. code-block:: bash

   open build/html/index.html  # macOS
   xdg-open build/html/index.html  # Linux

Live Rebuild
------------

.. code-block:: bash

   pip install sphinx-autobuild
   sphinx-autobuild docs/source docs/build/html

Opens at http://127.0.0.1:8000 with auto-reload.

Write Documentation
-------------------

* User guides: ``docs/source/user_guide/``
* API docs: Use docstrings with Google/NumPy style
* Examples: ``examples/``

Docstring Example:

.. code-block:: python

   def my_function(param1: int, param2: str) -> bool:
       """Short description.

       Longer description explaining the function.

       Args:
           param1: Description of param1.
           param2: Description of param2.

       Returns:
           Description of return value.

       Raises:
           ValueError: When param1 is negative.

       Example:
           >>> my_function(1, "test")
           True
       """
       pass

Release Process
===============

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create git tag:

   .. code-block:: bash

      git tag -a v1.3.0 -m "Release v1.3.0"
      git push origin v1.3.0

4. Build and upload to PyPI:

   .. code-block:: bash

      python -m build
      python -m twine upload dist/*

Getting Help
============

* GitHub Issues: https://github.com/FerMat-ML/OMatG/issues
* Discussions: https://github.com/FerMat-ML/OMatG/discussions
* Email: See AUTHORS file

.. toctree::
   :maxdepth: 2
   :hidden:

   contributing
