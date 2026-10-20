"""Minimal stand-in for the pytest features used by the R8 contract tests.

The shared push-T virtual environment does not ship pytest and installing into it
would touch an environment other sessions rely on.  This shim provides exactly
raises(), approx() and mark.skipif(); tests are executed by run_contract_tests.py.
"""


class _Raises:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("expected %s but nothing was raised" % self.expected)
        return issubclass(exc_type, self.expected)


def raises(expected):
    return _Raises(expected)


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = float(expected)
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        other = float(other)
        return abs(other - self.expected) <= max(self.abs, self.rel * abs(self.expected))

    def __repr__(self):
        return "approx(%r, rel=%r)" % (self.expected, self.rel)


def approx(expected, rel=1e-6, abs=1e-12):
    return _Approx(expected, rel=rel, abs=abs)


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def decorator(fn):
            fn.__skip__ = bool(condition)
            fn.__skip_reason__ = reason
            return fn

        return decorator

    @staticmethod
    def parametrize(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


mark = _Mark()
