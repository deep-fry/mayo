#!/usr/bin/env python3
import os
from importlib.util import spec_from_file_location, module_from_spec


root = os.path.dirname(__file__)
if root == '.':
    root = ''
path = os.path.join(root, 'mayo', 'cli.py')
spec = spec_from_file_location('cli', path)
cli = module_from_spec(spec)
spec.loader.exec_module(cli)
cli.CLI().main()
