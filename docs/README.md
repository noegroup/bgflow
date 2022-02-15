# Compiling bgflow's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that Sphinx and corresponding packages are installed.


```bash
pip install sphinx sphinx_rtd_theme sphinx_gallerie sphinx_nbexamples sphinxcontrib-katex sphinxcontrib-bibtex
```

Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `docs/_build/html/` directory and can be viewed by opening `index.html`.

The documentation rst-files can be found in `docs` and `docs/api`. To include a class/function in the documentation it
needs to be referenced in a corresponding `__init__.py`. That way the documentation is directly where the code is. For
an example see `nn/flow/__init__.py`. If the `__init__.py` does not have any docstrings so far, it most likely needs to
be referenced in a rst-file. The rst-files need to look like the ones in `docs/api` and must also be included
in `docs/index.rst`.

Notebooks can be also part of the documentation as well. To include them they need to be in `examples/nb_examples`
and named `example_{name}.ipynb`. Especially md comments in separate cells work well.


A configuration file for [Read The Docs](https://readthedocs.org/) (readthedocs.yaml) is included in the top level of
the repository. To use Read the Docs to host your documentation, go to https://readthedocs.org/
and connect this repository. You may need to change your default branch to `main` under Advanced Settings for the
project.

If you would like to use Read The Docs with `autodoc` (included automatically)
and your package has dependencies, you will need to include those dependencies in your documentation yaml
file (`docs/requirements.yaml`).

