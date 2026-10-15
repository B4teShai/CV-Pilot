"""Infrastructure: tracker forwarders (``mlflow``, ``wandb``, ``trackio``), one module each.

Each adapter imports its SDK inside ``open`` only, reads no credential (the SDK reads its own
environment variables), refuses plain-http endpoints and endpoints with credentials unless
``CVPILOT_ENV=dev`` allows http, and never turns TLS verification off. The fan-out that calls
them, with the queue, timeout and isolation rules, lives in ``cvpilot.runs.forwarding``.
"""
