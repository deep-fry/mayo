import collections


def unique(items):
    found = set()
    keep = []
    for item in items:
        if item in found:
            continue
        found.add(item)
        keep.append(item)
    return keep


def flatten(items, skip_none=False):
    for i in items:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        elif i is not None:
            yield i


def ensure_list(item_or_list):
    import tensorflow as tf
    if isinstance(item_or_list, (str, collections.Mapping, tf.Tensor)):
        return [item_or_list]
    if isinstance(item_or_list, list):
        return item_or_list
    raise TypeError('Unrecognized type.')


def recursive_apply(obj, apply_funcs, skip_func=None):
    """
    Recursively apply functions to a nested map+list data structure.

    obj:
        The object to be applied with functions recursively.
    apply_funcs:
        A mapping where the key is a type and the value is a function to apply
        to an object of the type.  The function accepts a signature (obj) where
        `obj` is the object.
    skip_func:
        A function to determine whether to skip the current object or not, and
        disallow recursion into that object.  It accepts a signature (obj) as
        described above.
    """
    from mayo.parse import _DotDict
    if skip_func:
        skip_obj = skip_func(obj)
        if skip_obj is not None:
            return skip_obj
    if isinstance(obj, _DotDict):
        obj = obj._mapping
    if isinstance(obj, collections.Mapping):
        for k, v in obj.items():
            obj[k] = recursive_apply(v, apply_funcs, skip_func)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        obj = obj.__class__(
            recursive_apply(v, apply_funcs, skip_func) for v in obj)
    for cls, func in apply_funcs.items():
        if isinstance(obj, cls):
            return func(obj)
    return obj
