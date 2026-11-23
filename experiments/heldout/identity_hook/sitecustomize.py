"""Fresh-identity runs only: shift every TAILOR composite identity index by a fixed offset.

The frozen driver names the identity of cover i as img_{i:05d} through eval_matrix.OursComposite.secret_for,
both when it embeds and in every decode (in process and in the cross-environment decode_xenv step). This
module is imported at interpreter start when its directory leads PYTHONPATH. It reads identity.json next to
itself and, when eval_matrix is imported, wraps OursComposite.secret_for so that cover i is embedded and
verified under identity img_{offset+i}, an identity that calibration never used. Embedding and decoding go
through the same method, so they stay consistent. No frozen file changes. Without identity.json, or in a
process that never imports eval_matrix (the attack environments), it does nothing.

Kept compatible with the oldest interpreter of the chain (the attack environments).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONF = os.path.join(_HERE, 'identity.json')


def _offset():
    try:
        with open(_CONF) as fh:
            return int(json.load(fh)['offset'])
    except (IOError, OSError):
        return None


def _patch(module, off):
    cls = getattr(module, 'OursComposite', None)
    if cls is None or getattr(cls, '_tailor_identity_offset', None) is not None:
        return
    original = cls.secret_for

    def secret_for(self, idx, _original=original, _off=off):
        return _original(self, int(idx) + _off)

    cls.secret_for = secret_for
    cls._tailor_identity_offset = off
    sys.stderr.write('IDENTITY_OFFSET %d installed on %s\n' % (off, getattr(module, '__file__', '?')))
    sys.stderr.flush()


_OFF = _offset()
if _OFF is not None:
    import importlib.abc
    import importlib.machinery

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            _patch(module, _OFF)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != 'eval_matrix':
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
