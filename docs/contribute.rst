Contribute
==========

If you have implemented a new tempo estimation approach or
created a new reference dataset, please contribute your data.


References
----------

If you have created a revised set of annotations for an existing dataset
or a completely new dataset, please create corresponding `JAMS
<https://github.com/marl/jams>`_ and a pull request.

Additional annotations for existing datasets should be added to the
existing ``.jams`` files. Please make sure to include both the data and
the code that adds your data to the jams.

Example
~~~~~~~

To add new *reference annotations* to the dataset *ballroom*...

1. Create a new folder under ``annotations/references/ballroom/``
   named after the publication that features your new annotations,
   e.g., ``smith2018``. If your annotations were not released in
   a scientific publication, come up with some other reasonable name.
2. Place your original annotations (that don't have to be `JAMS
   <https://github.com/marl/jams>`_) into the new folder.
3. Add code to ``tempo_eval/parser/ballroom.py`` that creates ``.jams``
   files containing *all* known annotations for the dataset. These
   ``.jams`` must be placed into the folder
   ``annotations/references/ballroom/jams/``

Should you want to add a *new* dataset, please follow the recipe above
with the exception that you should create a new Python file named
``your_dataset_name.py``, which reads your original annotations and
creates ``.jams`` files in the appropriate folder.

Please read about how to annotate your
`JAMS <https://github.com/marl/jams>`_ in `Annotation Metadata`_.


Estimates
---------

An essential characteristic of tempo_eval is that it does not only
contain reference datasets, but also estimates.
Contributing your estimates allows other researchers to compare to your
work without having to re-run all your experiments.
At the same time, you can compare your results with a wide variety
of existing approaches.


Example
~~~~~~~

To add new *estimates* for the dataset *ballroom*...

1. Create a new folder under ``annotations/estimates/ballroom/``
   named after the publication that features your new approach,
   e.g., ``smith2018``. If your estimates were not released in
   a scientific publication, come up with some other reasonable name
   that allows others to recognize your system, e.g., a product name.
2. Oftentimes one system can be run with different parameters or
   one publication features multiple approaches. In your ``smith2018``
   folder, create subfolders for each of these approaches or parameter
   sets. The subfolder's name could simply be a version number
   ``version_3_2_1`` or just ``default``. It's highly recommended to
   include information in the JAMS as well.
3. Place your original annotations (that don't have to be `JAMS
   <https://github.com/marl/jams>`_) into the new subfolder.
4. If your annotations aren't yet in the `JAMS
   <https://github.com/marl/jams>`_ format, convert them.
   tempo_eval provides a script called `convert2jams`_, which makes it
   easy to convert popular formats like CSV.
   Before you do this, please read about how to properly annotate your
   `JAMS <https://github.com/marl/jams>`_ in `Annotation Metadata`_.



Annotation Metadata
-------------------

Please ensure that your `JAMS <https://github.com/marl/jams>`_
contain suitable `annotation_metadata
<https://jams.readthedocs.io/en/stable/jams_structure.html#annotation-metadata>`_.
The bare minimum for tempo_eval is a ``corpus`` name and a ``version`` that is
unique per corpus.
The corpus name should match the ``annotations/references/``-folder name,
if possible, e.g., ``ballroom``.

Because the ``annotator`` object is of type `Sandbox
<https://jams.readthedocs.io/en/stable/generated/jams.Sandbox.html>`_, it
supports any kind of ``dict``. tempo_eval honors two special keys: ``bibtex``
and ``ref_url``. Both keys aim at giving credit to your work.
While ``ref_url`` lets you specify a simple URL to your own website,
``bibtex`` lets you specify a minimal bibtex entry. Should you choose to
add such an entry, you might also want to consider adding it to
``tempo_eval/references.bib``.

.. code-block:: javascript
    :linenos:
    :caption: Example for ``ref_url`` and ``bibtex`` annotator keys.
    :emphasize-lines: 8,9

    [...]
    "annotation_metadata": {
        "curator": {
            "name": "Simon Dixon",
            "email": "s.e.dixon@qmul.ac.uk"
        },
        "annotator": {
            "bibtex": "@article{Gouyon2006,\nAuthor = \"Gouyon, Fabien and Klapuri, Anssi P. and Dixon, Simon and Alonso, Miguel and Tzanetakis, George and Uhle, Christian and Cano, Pedro\",\nJournal = \"IEEE Transactions on Audio, Speech, and Language Processing\",\nNumber = \"5\",\nPages = \"1832--1844\",\nTitle = \"An experimental comparison of audio tempo induction algorithms\",\nVolume = \"14\",\nYear = \"2006\"\n}\n",
            "ref_url": "http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html"
        },
        "version": "1.0",
        "corpus": "ballroom",
        "annotation_tools": "",
        "annotation_rules": "",
        "validation": "",
        "data_source": "BallroomDancers.com, checked by human"
    }
    [...]


convert2jams
------------

convert2jams is a simple command line script that converts your CSV, TSV,
JSON or plain text tempo annotations into `JAMS <https://github.com/marl/jams>`_.

For detailed usage information please run:

.. code-block:: console

    $ convert2jams --help

Here's an example that turns text files into jams.

.. code-block:: console

    $ convert2jams -d out_dir -i text_file_dir -a audio_file_dir \
        -c corpus_name -v annotation_version

For this to work, each text file located in ``text_file_dir`` must contain only
tempo values for a single audio file. These tempo values can either be single
values or a `MIREX-style triplet
<https://www.music-ir.org/mirex/wiki/2006:Audio_Tempo_Extraction>`_.
The text file names must correspond to the audio file names. For example, the
annotation for the audio file ``my_track.wav`` should be in a text file called
``my_track.txt``. All audio files should be under ``audio_file_dir``.

Note that each jam file should at least be annotated with a corpus name and a
version.
More parameters, like bibtex, curator, etc. can be easily specified via the
command line.

Alternatively, if you already have a jam file that contains a suitable
`annotation_metadata
<https://jams.readthedocs.io/en/stable/jams_structure.html#annotation-metadata>`_-block,
you can use it as template using the ``--template`` parameter:

.. code-block:: console

    $ convert2jams -d out_dir -i text_file_dir -a audio_file_dir \
        --template template.jams


