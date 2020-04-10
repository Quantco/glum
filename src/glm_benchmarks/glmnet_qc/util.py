from typing import Callable, Tuple

import numpy as np

default_links = {"gaussian": "identity", "poisson": "log", "bernoulli": "logit"}


def get_link_and_inverse(link_name) -> Tuple[Callable, Callable]:
    if link_name == "identity":

        def identity(x):
            return x

        return identity, identity

    if link_name == "log":
        return np.log, np.exp
    if link_name == "logit":

        def inv_link(x):
            return 1 / (1 + np.exp(x))

        def link(x):
            return np.log((1 - x) / x)

        return link, inv_link
    raise NotImplementedError(f"{link_name} is not a supported link function")
