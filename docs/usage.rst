Usage
=====

After `installation <install.html>`_ you may call tempo_eval from
the command line using the ``tempo_eval`` command. Calling it without
arguments will generate a `Markdown
<https://en.wikipedia.org/wiki/Markdown>`_-formatted evaluation
report in the *current* directory *for all* built-in datasets and
annotations.

.. code-block:: console

    $ tempo_eval


Help
----

Most likely that is not what you want. To learn about command line
options, you may pass the parameter ``--help``.

.. code-block:: console

    $ tempo_eval --help


Output Directory
----------------

You can change the output directory using the ``--dir`` option:

.. code-block:: console

    $ tempo_eval --dir MY_OUTPUT_DIR


Specify the Corpus
------------------

To specify the datasets for which you want to create evaluation
reports, you can pass one or more corpus names using the
``--corpus`` parameter:

.. code-block:: console

    $ tempo_eval --corpus [CORPUS0 [CORPUS1 [...]]]

Possible corpus names are for example ``gtzan`` or ``ballroom`` and
correspond to the folder names of the built-in annotations.


Output Format
-------------

tempo_eval is designed to create Markdown reports which can easily be
hosted on `GitHub Pages <https://pages.github.com>`_. While Markdown
is very readable even without having been rendered,
sometimes it is preferable to have an HTML version. This can be
generated simply by specifying ``html`` as format:

.. code-block:: console

    $ tempo_eval --format html

Alternatively, Markdown can be viewed in Chrome with the help of
an extension (e.g. with `Markdown Preview Plus
<https://chrome.google.com/webstore/detail/markdown-preview-plus/febilkbfcbhebfnokafefeacimjdckgl>`_).


Report Size
-----------

By default, tempo_eval generates a very comprehensive report. Not
all report elements are always needed (or wanted). To create shorter versions,
pass the ``--size`` parameter:

.. code-block:: console

    $ tempo_eval --size S

Valid values are ``S``, ``M``, and ``L`` (the default).
``XL`` is reserved for experimental use.


Custom Datasets
---------------

In order to generate reports for your custom annotations and estimates
you may specify either one on the command line using the ``--estimates``
and ``--references`` parameters:

.. code-block:: console

    $ tempo_eval --references FOLDER_WITH_REFERENCE_JAMS \
                 --estimates FOLDER_WITH_ESTIMATE_JAMS

Both the given reference and estimates folders are walked recursively
and any found JAMS are read. To make comparison with built-in reference
annotations easy, make sure to always specify an
``annotation_metadata.version`` and ``corpus`` (see also `here
<https://jams.readthedocs.io/en/stable/jams_structure.html#annotation-metadata>`_).
In essence, you want to match the used corpus names of the built-in reference
annotations, which are lowercase and contain underscores ``_`` instead of
spaces.
