from mayo.cli import meta
from mayo import task
from mayo import override
from mayo import objects

__all__ = [task, override, objects]
locals().update(meta())
