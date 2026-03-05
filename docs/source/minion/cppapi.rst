API (C++)
================

The **C++** API provides access to the core optimization algorithms and benchmark functions implemented in Minion.

.. note::
   The full Doxygen index may include anonymous namespaces from implementation files, which can break Breathe parsing.
   This page documents key public C++ classes.

Core classes
------------

.. doxygenclass:: minion::Minimizer
   :project: Minion
   :members:

.. doxygenclass:: minion::MinimizerBase
   :project: Minion
   :members:

.. doxygenclass:: minion::Options
   :project: Minion
   :members:

.. doxygenclass:: minion::DefaultSettings
   :project: Minion
   :members:

Utilities
---------
Core math utility templates are currently omitted from this page because Breathe's parser can be strict with some template declarations from Doxygen XML.
