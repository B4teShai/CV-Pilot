"""Infrastructure: alert channels (``webhook``, ``slack``, ``telegram``), one module each.

A channel is built like a forwarder (``allow_insecure``, ``allowed_hosts``), reads where it sends
and its credential once from the environment, and delivers over https through ``_http`` only:
public addresses only, no redirects, a total deadline per call. The fan-out that calls them, with
the queue, stall and isolation rules, is ``cvpilot.runs.forwarding`` (ADR-0009).
"""
