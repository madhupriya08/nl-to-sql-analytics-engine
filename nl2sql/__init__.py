"""Agentic natural-language-to-SQL analytics engine.

The interesting engineering here is not the SQL generation -- an LLM does
that in one call. It is the safety layer that sits between the model's
output and the database's ``execute()``, and the seam that lets that layer
be tested without an API key.
"""

__version__ = "0.1.0"
