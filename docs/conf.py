# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyzentropy'
copyright = '2026, Nigel Lee En Hew'
author = 'Nigel Lee En Hew'
release = '0.1.0'
__version__ = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.linkcode',
              'sphinx.ext.duration',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx_autodoc_typehints',
              'sphinx_github_changelog',
              'sphinx_rtd_size',
              'sphinx.ext.mathjax',
              'myst_nb'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        import inspect
        import os, sys
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    try:
        rel_path, line_start, line_end = find_source()
        # __file__ is imported from pyzentropy
        filename = f"pyzentropy/{rel_path}#L{line_start}-L{line_end}"
    except Exception:
        # no need to be relative to core here as module includes full path.
        filename = info["module"].replace(".", "/") + ".py"

    tag = "v" + str(__version__)
    return f"https://github.com/PhasesResearchLab/pyzentropy/blob/{tag}/{filename}"