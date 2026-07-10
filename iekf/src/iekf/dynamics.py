import numpy as np

from iekf.utils.lie_group import so3_exp, so3_left_jacobian, so3_gamma_2


class IEKFDynamics:
    """Pure pose propagation (no correction, no covariance). This is
    the same integration math used in RIEKF._predict / ContactBiasIEKF
    .propagate, pulled out on its own for isolated free-fall / dynamics
    testing.
    """

    def __init__(self, gravity=None, noise_params=None):
        self.g = np.array([0.0, 0.0, -9.81]) if gravity is None else np.asarray(gravity)
        self.noise = noise_params

    def make_state(self, R=None, v=None, p=None, dl=None, dr=None):
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
        return X

    def unpack_state(self, X):
        R = X[:3, :3]
        v = X[:3, 3]
        p = X[:3, 4]
        dl = X[:3, 5]
        dr = X[:3, 6]
        return R, v, p, dl, dr

    def propagate_state(self, X, gyro_bias, accel_bias, imu, dt):
        R, v, p, dl, dr = self.unpack_state(X)

        omega = imu.gyro - gyro_bias
        a = imu.accel - accel_bias

        R_new = R @ so3_exp(omega * dt)
        v_new = v + (R @ so3_left_jacobian(omega * dt) @ a + self.g) * dt
        p_new = p + v * dt + (R @ so3_gamma_2(omega * dt) @ a + 0.5 * self.g) * dt**2

        X_new = np.eye(7)
        X_new[:3, :3] = R_new
        X_new[:3, 3] = v_new
        X_new[:3, 4] = p_new
        X_new[:3, 5] = dl
        X_new[:3, 6] = dr
        return X_new