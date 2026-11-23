"""Run registry: a combined table may only draw on runs with the same frozen evaluation identity.

A run's tag names everything that must be equal for two runs to sit in one table: the
verification protocol, the request cohort, the decoder (code plus weights), the image role, the
attack parameterization, the key, the latency basis and the margin policy. The field names are
the result-schema field names, so a record carries its run's tag verbatim. combine() refuses any
difference. The margin policy is the single field a table may vary on purpose (the screening
margin comparison), and only when the caller declares it.
"""
from . import schema

TAG_FIELDS = ('protocol_version', 'cohort_hash', 'decoder_hash', 'image_role', 'attack_param_id',
              'margin_policy', 'latency_basis', 'key_seed')
MAY_DIFFER = ('margin_policy',)


class RegistryError(ValueError):
    """Runs or records that must not be combined."""


def _validated(tag):
    missing = [f for f in TAG_FIELDS if not isinstance(tag.get(f), str) or not tag.get(f)]
    if missing:
        raise RegistryError(f'run tag lacks {missing}')
    return {f: tag[f] for f in TAG_FIELDS}


class Registry:
    def __init__(self):
        self.runs = {}

    def register(self, run_id, tag):
        tag = _validated(tag)
        if run_id in self.runs and self.runs[run_id] != tag:
            raise RegistryError(f'run {run_id!r} is already registered with another identity')
        self.runs[run_id] = tag
        return tag

    def _tag(self, run_id):
        if run_id not in self.runs:
            raise RegistryError(f'run {run_id!r} is not registered')
        return self.runs[run_id]

    def combine(self, run_ids, allow_differ=()):
        run_ids = list(run_ids)
        if not run_ids:
            raise RegistryError('nothing to combine')
        forbidden = sorted(set(allow_differ) - set(MAY_DIFFER))
        if forbidden:
            raise RegistryError(f'{forbidden} may never differ inside one table')
        tags = [self._tag(r) for r in run_ids]
        combined = {}
        for field in TAG_FIELDS:
            values = sorted({t[field] for t in tags})
            if len(values) > 1 and field not in allow_differ:
                raise RegistryError(f'{field} differs across {run_ids}: {values}')
            combined[field] = values[0] if len(values) == 1 else tuple(values)
        return combined

    def check_records(self, run_id, records):
        """Every record must be schema-valid, belong to this run and carry the run's tag."""
        tag = self._tag(run_id)
        for i, record in enumerate(records):
            try:
                schema.validate_record(record)
            except schema.SchemaError as err:
                raise RegistryError(f'record {i} of run {run_id!r} is not schema-valid: {err}') from err
            if record['run_id'] != run_id:
                raise RegistryError(f"record {i} belongs to run {record['run_id']!r}, not {run_id!r}")
            wrong = [f for f in TAG_FIELDS if record[f] != tag[f]]
            if wrong:
                raise RegistryError(f'record {i} of run {run_id!r} disagrees with the run tag on {wrong}')
        return len(records)
