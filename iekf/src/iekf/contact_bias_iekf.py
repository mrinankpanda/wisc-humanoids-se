import numpy as np
from dataclasses import dataclass

from iekf.right_invariant import RIEKF
from iekf.utils.lie_group import so3_exp, so3_left_jacobian, so3_gamma_2, sek3_adjoint

# Error-state index constants: [phi, v, p, d_L, d_R, b_L, b_R]
IDX_PHI = slice(0, 3)
IDX_V = slice(3, 6)
IDX_P = slice(6, 9)
IDX_DL = slice(9, 12)
IDX_DR = slice(12, 15)
IDX_BL = slice(15, 18)
IDX_BR = slice(18, 21)
DIM_ERR = 21


@dataclass
class ContactBiasState:
    X: np.ndarray  # (7,7) SE_4(3) pose, same layout as RobotState.X
    b_L: np.ndarray  # (3,) world-frame left foot FK bias
    b_R: np.ndarray  # (3,) world-frame right foot FK bias
    P: np.ndarray  # (21,21) error covariance


class ContactBiasIEKF:
    """RIEKF extended with additive world-frame foot-position bias states.

    Composes a base 15-state RIEKF and augments the error state with a
    6-dim bias block (3 per foot), modeled as a random walk. The pose
    propagation math is duplicated from RIEKF._predict (rather than
    calling it directly) because it needs bias-corrected gyro/accel
    inputs that RIEKF's own _predict doesn't accept.
    """

    def __init__(self, fk_model=None, dt=None, gravity=None, noise_params=None):
        self.base = RIEKF(fk_model, dt=dt, gravity=gravity, noise_params=noise_params)
        self.noise = self.base.noise
        self.dt = self.base.dt
        self.fk_model = fk_model

    def make_state(self, R=None, v=None, p=None, dl=None, dr=None,
                    b_L=None, b_R=None, P=None):
        X = np.eye(7)
        if R is not None:
            X[:3, :3] = np.asarray(R)
        if v is not None:
            X[:3, 3] = np.asarray(v).ravel()
        if p is not None:
            X[:3, 4] = np.asarray(p).ravel()
        if dl is not None:
            X[:3, 5] = np.asarray(dl).ravel()
        if dr is not None:
            X[:3, 6] = np.asarray(dr).ravel()

        b_L = np.zeros(3) if b_L is None else np.asarray(b_L).ravel()
        b_R = np.zeros(3) if b_R is None else np.asarray(b_R).ravel()
        P = np.eye(DIM_ERR) if P is None else P

        return ContactBiasState(X=X, b_L=b_L, b_R=b_R, P=P)

    def compute_F(self, dt):
        F = np.eye(DIM_ERR)
        F[:15, :15] = self.base.compute_F(dt)
        return F

    def compute_Q(self, dt):
        Q = np.zeros((DIM_ERR, DIM_ERR))
        Q[:15, :15] = self.base.compute_Q(dt)
        Q[IDX_BL, IDX_BL] = self.noise.bias_cov * dt
        Q[IDX_BR, IDX_BR] = self.noise.bias_cov * dt
        return Q

    def build_H(self, foot):
        H = np.zeros((3, DIM_ERR))
        H[:, IDX_P] = -np.eye(3)
        if foot == "left":
            H[:, IDX_DL] = np.eye(3)
            H[:, IDX_BL] = np.eye(3)
        elif foot == "right":
            H[:, IDX_DR] = np.eye(3)
            H[:, IDX_BR] = np.eye(3)
        else:
            raise ValueError(f"Unknown foot: {foot!r}")
        return H

    def _foot_terms(self, state, foot):
        R = state.X[:3, :3]
        p = state.X[:3, 4]
        if foot == "left":
            d, b = state.X[:3, 5], state.b_L
        elif foot == "right":
            d, b = state.X[:3, 6], state.b_R
        else:
            raise ValueError(f"Unknown foot: {foot!r}")
        return R, p, d, b

    def predict_measurement(self, state, foot):
        """Predicted body-frame FK measurement consistent with the current estimate."""
        R, p, d, b = self._foot_terms(state, foot)
        return R.T @ (d + b - p)

    def innovation(self, state, y_fk_body, foot):
        """World-frame residual: (R @ y_body + p) - (d_hat + b_hat)."""
        R, p, d, b = self._foot_terms(state, foot)
        y_world = R @ y_fk_body + p
        return y_world - d - b

    def propagate(self, state, gyro_bias, accel_bias, imu, dt):
        R = state.X[:3, :3]
        v = state.X[:3, 3]
        p = state.X[:3, 4]
        dl = state.X[:3, 5]
        dr = state.X[:3, 6]

        omega = imu.gyro - gyro_bias
        a = imu.accel - accel_bias
        g = self.base.g

        R_new = R @ so3_exp(omega * dt)
        v_new = v + (R @ so3_left_jacobian(omega * dt) @ a + g) * dt
        p_new = p + v * dt + (R @ so3_gamma_2(omega * dt) @ a + 0.5 * g) * dt**2

        X_new = np.eye(7)
        X_new[:3, :3] = R_new
        X_new[:3, 3] = v_new
        X_new[:3, 4] = p_new
        X_new[:3, 5] = dl
        X_new[:3, 6] = dr

        F = self.compute_F(dt)

        Adj_full = np.eye(DIM_ERR)
        Adj_full[:15, :15] = sek3_adjoint(state.X, k=4)
        F_adj = F @ Adj_full

        # Base block is a continuous noise density -> discretize with *dt.
        Q_density = np.zeros((DIM_ERR, DIM_ERR))
        Q_density[:15, :15] = self.base.compute_Q(dt)
        Q_disc = F_adj @ Q_density @ F_adj.T * dt

        # Bias block from compute_Q(dt) is already discrete -- add directly.
        Q_disc[IDX_BL, IDX_BL] += self.noise.bias_cov * dt
        Q_disc[IDX_BR, IDX_BR] += self.noise.bias_cov * dt

        P_new = F @ state.P @ F.T + Q_disc

        return ContactBiasState(X=X_new, b_L=state.b_L, b_R=state.b_R, P=P_new)
    
    def reset_contact(self, state, foot, d_new):
        """Called on new foot touchdown: sets the contact position to
        d_new, zeroes the bias estimate, and decorrelates both from the
        rest of the state (P off-diagonal blocks -> 0). Diagonal blocks
        get fresh priors rather than carrying over stale covariance from
        the previous contact.
        """
        if foot == "left":
            idx_d, idx_b = IDX_DL, IDX_BL
        elif foot == "right":
            idx_d, idx_b = IDX_DR, IDX_BR
        else:
            raise ValueError(f"Unknown foot: {foot!r}")

        X_new = state.X.copy()
        if foot == "left":
            X_new[:3, 5] = d_new
        else:
            X_new[:3, 6] = d_new

        b_L = np.zeros(3) if foot == "left" else state.b_L
        b_R = np.zeros(3) if foot == "right" else state.b_R

        P_new = state.P.copy()
        for idx in (idx_d, idx_b):
            P_new[idx, :] = 0
            P_new[:, idx] = 0

        P_new[idx_d, idx_d] = self.noise.contact_cov
        # TODO: pick an actual bias-reset prior...needs more research
        P_new[idx_d, idx_d] = np.eye(3) * 0.02**2   # ~2 cm — flat, predictable ground
        P_new[idx_b, idx_b] = np.eye(3) * 0.02**2   # match it for now, keep it simple

        return ContactBiasState(X=X_new, b_L=b_L, b_R=b_R, P=P_new)