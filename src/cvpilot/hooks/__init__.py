"""Infrastructure: framework hooks (Lightning, and the YOLO framework), one module each.

A hook turns a training framework's callbacks into ``MetricSink`` calls. Rules (tests in
``tests/unit/test_hooks_*.py``):

* Callbacks never raise into the host: every body runs under a ``Guard`` that catches
  ``Exception``, counts it, logs the first one per hook as ``hook_failed`` (error type only) and
  stops calling a hook that failed ``MAX_CONSECUTIVE_FAILURES`` times in a row (``hook_disabled``).
  ``KeyboardInterrupt`` and ``SystemExit`` are never swallowed.
* Callbacks never block the host beyond one buffered batch write: they make no network call (the
  forwarders have their own threads) and ``sink.log`` writes at most one Parquet batch of
  ``FLUSH_POINTS`` points on the calling thread.
* Framework objects are read through narrow adapters that touch only the attributes named in
  them, convert values defensively and bound what they take (``MAX_POINTS_PER_CALL``).
* A framework package is imported only inside the function that needs it, and no module outside
  ``hooks/`` mentions one. The adapter of a copyleft framework says so in ``license_notice``.
"""
