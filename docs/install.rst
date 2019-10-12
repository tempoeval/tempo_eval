Installation Instructions
=========================

Source
------

If you've downloaded the archive manually from the `releases
<https://github.com/tempoeval/tempo_eval/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf tempo_eval-VERSION.tar.gz
    cd tempo_eval-VERSION/
    python setup.py install

If you intend to develop tempo_eval or make changes to the source code, you can
install with `pip install -e` to link to your actively developed source tree::

    tar xzf tempo_eval-VERSION.tar.gz
    cd tempo_eval-VERSION/
    pip install -e .

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/tempoeval/tempo_eval
