"""Maintain shifted legacy raw-BA priors without importing the old ID decoder."""
import copy
MARKER = '_unified_legacy_prior_installed'

def install(module=None):
    if module is None:
        import surrogate_model as module
    cls = module.Surrogate
    if getattr(cls, MARKER, False):
        return
    original = cls.with_live
    def with_live(self, offsets):
        result = original(self, offsets)
        for key, offset in (offsets or {}).items():
            if len(key) == 4 and key[0] == 'fe':
                _, stage, fragment, attack = key
            else:
                fragment, attack = key
                stage = None
            pk = self.perimage_key(fragment, attack, stage)
            record = self._perimage.get(pk)
            if not record or 'raw_views' not in record or abs(offset) < 1e-12:
                continue
            updated = copy.deepcopy(record)
            for cell in updated['raw_views']:
                for name in ('ba', 'cas_ba'):
                    cell[name][fragment] = [None if value is None else min(1., max(0., value + offset))
                                            for value in cell[name][fragment]]
            updated['ba'] = [list(cell['ba'][fragment]) for cell in updated['raw_views']]
            updated['ver'] = [[False] * len(row) for row in updated['ba']]
            result._perimage[pk] = updated
        return result
    cls.with_live = with_live
    setattr(cls, MARKER, True)
