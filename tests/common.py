import unittest


class TestCase(unittest.TestCase):
    def assertEqual(self, *args, msg=None):
        for first, second in zip(args, args[1:]):
            super().assertEqual(first, second, msg)

    def assertObjectEqual(self, x, y):
        self.assertEqual(x.__class__, y.__class__)
        self.assertEqual(
            getattr(x, '__dict__', None), getattr(y, '__dict__', None))
