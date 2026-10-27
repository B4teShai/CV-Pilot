# CV-Pilot architecture

CV-Pilot is a Python library and command line that sits next to your training code. It records each
run in a local folder, scores models, compares them with a baseline and saves checkpoints you can
trust after a crash. For what it does and how to use it, start with the [tutorial](../tutorial.md).
What works today is on the [status page](../status.md). Terms are in the
[glossary](../guide/concepts.md#glossary).

![CV-Pilot records your training, compares the new model with your best one and tells CI pass or fail](../diagrams/png/overview.png)

## The pages

| Page | Question it answers |
|---|---|
| [01 Overview](01-overview.md) | Who uses it, what is in and out of scope, what must hold. |
| [02 Components](02-components.md) | Which packages exist, what each owns and what may import what. |
| [03 Data model](03-data-model.md) | Which records are stored, how they relate and where they live on disk. |
| [04 Flows](04-flows.md) | How logging, gating and checkpointing run end to end, and how each fails. |
| [05 Runtime](05-runtime.md) | Which threads run, where data goes and which settings exist. |
| [06 Cross-cutting](06-cross-cutting.md) | Logging, errors, security, the CI trust model and licensing. |
| [ADRs](../adr/0001-scope-and-license.md) | Why it is built this way. |

The diagrams are generated from `docs/diagrams/*.diagram.mjs`. Each PNG embeds its scene, so you
can open it in [excalidraw.com](https://excalidraw.com) and edit it. A change that adds a package, a
public interface, a stored record or an outside dependency updates the matching page and diagram in
the same pull request ([workflow](../engineering/workflow.md)).
