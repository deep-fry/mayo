from mayo.log import log
from mayo.config import Config
from mayo.net import Net
from mayo.train import Train
from mayo.eval import Evaluate
from mayo.cli import meta
from mayo import surgery

__all__ = [log, Config, Net, Train, Evaluate]
locals().update(meta())

surgery._register_surgery_objects(surgery)
