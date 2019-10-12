.. tempo_eval documentation master file, created by
   sphinx-quickstart on Fri May 10 13:42:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tempo_eval
==========

tempo_eval is three things in one:

1. A repository for musical tempo annotations and estimates
2. A simple evaluation framework
3. A Markdown site generator

It documents strengths and weaknesses of classic as well
as modern tempo estimation approaches on multiple versions of
commonly used datasets using a variety of metrics.

For evaluation results, please see `here<https://tempoeval.github.io/tempo_eval/report/index.html`_.


Repository
----------

In its `GitHub repository <https://github.com/tempoeval/tempo_eval>`_,
tempo_eval contains both published reference annotations and estimates
produced by various tempo estimation systems as `JAMS
<https://github.com/marl/jams>`_.

If you are the author of an estimation system or creator of a new
dataset, please `contribute <contribute.html>`_ your estimates and annotations.


Evaluation Framework
--------------------

tempo_eval compares tempo estimates to reference annotations using
several metrics.


Site Generator
--------------

Evaluation results are not only created as dull CSV files, but also
visualized as SVG and PDF diagrams embedded in a comprehensive,
publishable, Markdown-formatted report.



Getting Started
---------------

.. toctree::
   :maxdepth: 1

   install
   usage


More
----

.. toctree::
   :maxdepth: 1

   contribute


Reference
---------

.. toctree::
   :maxdepth: 1

   evaluation
   parser
   changes

* :ref:`genindex`
