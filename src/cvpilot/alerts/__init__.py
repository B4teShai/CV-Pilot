"""Application layer: client-side alert rules (ADR-0009).

``config`` validates the small JSON configuration, ``engine`` evaluates the rules on the metrics
a run logs and on its lifecycle events, in the training process, and hands each alert to a
delivery function. Channels (``cvpilot.notify``) are reached through the ``channels`` registry
by the code that starts the run (``cvpilot.runs``); nothing here imports one.
"""
