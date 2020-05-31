# coding: utf-8

# values in _zglobal only valid at running time, no backup !!
_zglobal = {}


def global_get(key):
    if key not in _zglobal:
        return None
    return _zglobal[key]


def global_update(key, value):
    _zglobal[key] = value
