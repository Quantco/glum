import time


def runtime(f, *args, **kwargs):
    start = time.time()
    out = f(*args, **kwargs)
    end = time.time()
    return end - start, out
