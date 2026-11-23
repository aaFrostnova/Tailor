#!/usr/bin/env python3
"""Return optional ID-space guidance from the paper's measured-capacity model.

This is reporting-only post-processing. It neither selects an ID length nor
changes the measured watermark configuration, encoder, or verification logic.
The suggested range has not been validated for deployment at each ID length.
"""

import argparse
from fractions import Fraction
import json
import math
import numbers


def blank_guidance(status):
    """Return the guidance schema without a computed range.

    Exporters may use an explicit status such as ``pending`` or ``infeasible``
    when audited capacity evidence is unavailable. Missing evidence is distinct
    from ``empty``, which means valid inputs produced no positive ID length.
    """
    if not isinstance(status, str) or not status.strip():
        raise ValueError('status must be a nonempty string')
    return {
        'status': status,
        'capacity_bits_estimate': None,
        'fpr': None,
        'fpr_reserve_bits': None,
        'id_information_budget_bits': None,
        'id_bits_min': None,
        'id_bits_max': None,
        'id_space_size_min': None,
        'id_space_size_max': None,
        'selected_id_bits': None,
        'selection_by': 'user',
        'model_guidance_only': True,
        'range_deployment_validated': False,
    }


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(name + ' must be a finite real number, excluding bool')
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + ' must be a finite real number') from None
    if not math.isfinite(value):
        raise ValueError(name + ' must be a finite real number')
    return value


def recommend_id_space(capacity_bits_estimate, fpr):
    """Suggest integer ID lengths 1..floor(C - log2(1/FPR)), if nonempty.

    C is the measured worst-attack capacity estimate for a validated fixed
    configuration, on the experiment's 100-bit channel. The recommendation uses
    the paper's model, not the current encoder's fixed 37-bit identity payload.
    At any user-chosen integer length ell in the range, the nominal label count
    is 2**ell. The maximum endpoint is not selected automatically.

    Label counts are decimal strings so JSON consumers cannot silently round
    integers above their numeric precision. No epsilon is added before floor.
    """
    capacity = _finite_number(capacity_bits_estimate, 'capacity_bits_estimate')
    probability = _finite_number(fpr, 'fpr')
    if not 0.0 <= capacity <= 100.0:
        raise ValueError('capacity_bits_estimate must be between 0 and 100 bits')
    if not 0.0 < probability < 1.0:
        raise ValueError('fpr must satisfy 0 < fpr < 1')

    # Avoid forming 1 / fpr: valid subnormal probabilities can overflow it.
    reserve = -math.log2(probability)
    budget = max(0.0, capacity - reserve)
    # Floor the exact difference of the two computed floating-point estimates.
    # A rounded subtraction alone could turn 100 - tiny_positive into 100.
    maximum = math.floor(Fraction.from_float(capacity) - Fraction.from_float(reserve))
    result = blank_guidance('available' if maximum >= 1 else 'empty')
    result.update({
        'capacity_bits_estimate': capacity,
        'fpr': probability,
        'fpr_reserve_bits': reserve,
        'id_information_budget_bits': budget,
    })
    if maximum >= 1:
        result.update({
            'id_bits_min': 1,
            'id_bits_max': maximum,
            'id_space_size_min': '2',
            'id_space_size_max': str(2 ** maximum),
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-bits', type=float, required=True,
                        help='Measured worst-attack capacity estimate, 0..100 bits')
    parser.add_argument('--fpr', type=float, required=True,
                        help='Requested FPR, strictly between 0 and 1')
    args = parser.parse_args(argv)
    try:
        result = recommend_id_space(args.capacity_bits, args.fpr)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
