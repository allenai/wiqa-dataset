from typing import List


def add_key_to_map_arr(key, value, map_):
    if key not in map_:
        map_[key] = []
    map_[key].append(value)


def getElem(arr: List, elem_idx:int, defaultValue):
    if not arr or len(arr) <= elem_idx:
        return defaultValue
    return arr[elem_idx]