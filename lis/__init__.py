"""lis -- Learned Iterative Solvers for AC power flow.

Reusable code behind the notebooks.  The narrative lives in ``notebooks/``; this package
holds only what is worth calling more than once.
"""

from lis.powerflow import (  # noqa: F401
    Case, NewtonResult, build_jacobian, build_ybus, bus_types, flat_start,
    load_case, make_sbus, mismatch, newton_pf,
)

__version__ = "0.1.0"
