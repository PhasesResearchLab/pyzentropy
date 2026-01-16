# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'pyzentropy'
copyright = '2025, Hew, Nigel'
author = 'Hew, Nigel'
language = 'en'

# -- General configuration

extensions = [
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'myst_nb',
    'sphinx_github_changelog',
    'sphinx_rtd_size',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]
autosummary_generate = True
autodoc_member_order = 'bysource'

# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/PhasesResearchLab/pyzentropy/blob/main/%s.py" % filename

# -- Options for EPUB output
epub_show_urls = 'footnote'
