from mayo.cli import meta
from mayo import task
from mayo import override

__all__ = [task, override]
locals().update(meta())
