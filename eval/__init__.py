# eval/__init__.py
# Package marker — makes eval/ importable as a Python module.
#
# WHY AN EMPTY __init__.py?
#
# Python requires a file called __init__.py in a directory before it
# treats that directory as a package. Without it, you can't do:
#   from eval.metrics import compute_metrics
# or run:
#   python -m eval.run
#
# It can be completely empty — its presence alone is enough.
# (Python 3.3+ supports "namespace packages" without __init__.py,
# but explicit is better than implicit, and some tools still require it.)
