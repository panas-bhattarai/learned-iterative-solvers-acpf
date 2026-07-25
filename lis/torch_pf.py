r"""Batched, differentiable AC power flow residual in PyTorch.

Notebook 05 needs to train an R2N2 on the *nonlinear* power flow residual, which means
backpropagating through repeated evaluations of :math:`\mathbf{f}` across a whole batch of
contingencies at once.  That rules out the SciPy-sparse implementation used elsewhere:
different contingencies have different sparsity patterns, so they cannot share a sparse
structure, and `np.add.reduceat` has no autograd.

We therefore use the dense real polar form — the same equations as
:class:`lis.jacobian_free.PolarResidual`, validated against it:

.. math::
    P_i = |V_i|\sum_k |V_k|\big(G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}\big), \qquad
    Q_i = |V_i|\sum_k |V_k|\big(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}\big)

Dense costs :math:`O(n_{bus}^2)` per evaluation instead of :math:`O(\mathrm{nnz})`, which
for `case118` is 118² = 13924 against 476 nonzeros.  That is the price of batching on a
GPU, and it is worth paying here because we are training, not deploying — the deployed
solver in notebooks 02–03 stays sparse.
"""

from __future__ import annotations

import numpy as np
import torch

from lis.powerflow import Case

__all__ = ["TorchPF"]

torch.set_default_dtype(torch.float64)


class TorchPF:
    r"""A batch of power flow problems sharing one bus-type layout.

    All cases in the batch must have the same slack/PV/PQ partition — true for an N-1
    family on a fixed grid, where only :math:`\mathbf{Y}_{bus}` changes.  Each batch member
    carries its own :math:`\mathbf{G}`, :math:`\mathbf{B}` and scheduled injections.
    """

    def __init__(self, cases: list[Case], device: str = "cpu"):
        ref, pv, pq = cases[0].ref, cases[0].pv, cases[0].pq
        for c in cases:
            if not (np.array_equal(c.ref, ref) and np.array_equal(c.pv, pv)
                    and np.array_equal(c.pq, pq)):
                raise ValueError("all cases in a batch must share the bus-type partition")

        self.ref, self.pv, self.pq = ref, pv, pq
        self.pvpq = np.r_[pv, pq]
        self.n_bus = cases[0].n_bus
        self.n = len(pv) + 2*len(pq)
        self.device = device

        y = np.stack([c.ybus.toarray() for c in cases])
        self.G = torch.tensor(y.real, device=device)
        self.B = torch.tensor(y.imag, device=device)
        s = np.stack([c.sbus for c in cases])
        self.sp = torch.tensor(s.real[:, self.pvpq], device=device)
        self.sq = torch.tensor(s.imag[:, pq], device=device)

        v0 = np.stack([c.v0 for c in cases])
        self.va0 = torch.tensor(np.angle(v0), device=device)
        self.vm0 = torch.tensor(np.abs(v0), device=device)

    # -- packing -----------------------------------------------------------------
    def x0(self) -> torch.Tensor:
        r"""Flat start packed into the Newton unknown vector :math:`x`."""
        return torch.cat([self.va0[:, self.pvpq], self.vm0[:, self.pq]], dim=1)

    def unpack(self, x: torch.Tensor):
        va = self.va0.clone()
        vm = self.vm0.clone()
        k = len(self.pvpq)
        va = va.index_copy(1, torch.as_tensor(self.pvpq, device=x.device), x[:, :k])
        vm = vm.index_copy(1, torch.as_tensor(self.pq, device=x.device), x[:, k:])
        return va, vm

    # -- residual ----------------------------------------------------------------
    def residual(self, x: torch.Tensor) -> torch.Tensor:
        r"""Power mismatch :math:`\mathbf{f}(x)` for the whole batch."""
        va, vm = self.unpack(x)
        theta = va.unsqueeze(2) - va.unsqueeze(1)              # (batch, n_bus, n_bus)
        c, s = torch.cos(theta), torch.sin(theta)
        p = vm * torch.einsum('bik,bk->bi', self.G*c + self.B*s, vm)
        q = vm * torch.einsum('bik,bk->bi', self.G*s - self.B*c, vm)
        return torch.cat([p[:, self.pvpq] - self.sp, q[:, self.pq] - self.sq], dim=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Dense Jacobian per batch element, by autograd. Used only for validation."""
        rows = []
        for i in range(self.n):
            e = torch.zeros(self.n, device=x.device); e[i] = 1.0
            xr = x.clone().requires_grad_(True)
            r = self.residual(xr)
            g, = torch.autograd.grad((r*e).sum(), xr, create_graph=False)
            rows.append(g)
        return torch.stack(rows, dim=1)
