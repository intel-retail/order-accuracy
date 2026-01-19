from collections import deque
from threading import Lock

MAX_RESULTS = 3

_results = deque(maxlen=MAX_RESULTS)
_lock = Lock()

def add_result(result: dict):
    with _lock:
        _results.appendleft(result)

def get_results():
    with _lock:
        return list(_results)
