import jax.numpy as np
import numpy as np
import torch


class Quaternion(object):

    def __init__(self, w=0, x=0, y=0, z=0):
        self.q = np.array([w, x, y, z])

    def attrsetter(attr):
        def set(self, value):
            setattr(self, attr, value)
        return set

    def __str__(self):
        return "({0}, {1}, {2}, {3})".format(*self.q)

    def __getitem__(self, key):
        return self.q[key]

    def __setitem__(self, key, value):
        self.q[key] = value

    def get_rotation_matrix(self):
        return np.array([
            [1 - 2 * self.q[2]**2 - 2 * self.q[3]**2,
             2 * self.q[1] * self.q[2] - 2 * self.q[0] * self.q[3],
             2 * self.q[1] * self.q[3] + 2 * self.q[0] * self.q[2]
            ],
            [2 * self.q[1] * self.q[2] + 2 * self.q[0] * self.q[3],
             1 - 2 * self.q[1]**2 - 2 * self.q[3]**2,
             2 * self.q[2] * self.q[3] - 2 * self.q[0] * self.q[1]],
            [2 * self.q[1] * self.q[3] - 2 * self.q[0] * self.q[2],
             2 * self.q[2] * self.q[3] + 2 * self.q[0] * self.q[1],
             1 - 2 * self.q[1]**2 - 2 * self.q[2]**2]
        ])

    def get_rotation_matrix_4d(self):
        return np.array([
            [1 - 2 * self.q[2]**2 - 2 * self.q[3]**2,
             2 * self.q[1] * self.q[2] - 2 * self.q[0] * self.q[3],
             2 * self.q[1] * self.q[3] + 2 * self.q[0] * self.q[2],
             0
            ],
            [2 * self.q[1] * self.q[2] + 2 * self.q[0] * self.q[3],
             1 - 2 * self.q[1]**2 - 2 * self.q[3]**2,
             2 * self.q[2] * self.q[3] - 2 * self.q[0] * self.q[1],
             0],
            [2 * self.q[1] * self.q[3] - 2 * self.q[0] * self.q[2],
             2 * self.q[2] * self.q[3] + 2 * self.q[0] * self.q[1],
             1 - 2 * self.q[1]**2 - 2 * self.q[2]**2,
             0],
             [0, 0, 0, 1]
        ])

    @staticmethod
    def SLERP(q1, q2, t):
        q1_ = np.array([q1[0], q1[1], q1[2], q1[3]])
        q2_ = np.array([q2[0], q2[1], q2[2], q2[3]])
        cosTheta = np.dot(q1_, q2_)
        if cosTheta < 1:
            theta = np.arccos(cosTheta)
            invSinTheta = 1. / np.sin(theta)
            c0 = np.sin((1 - t) * theta) * invSinTheta
            c1 = np.sin(t * theta) * invSinTheta
            q_ = c0 * q1_ + c1 * q2_
            return Quaternion(q_[0], q_[1], q_[2], q_[3])
        else:
            # In this case, the angle between the vectors is 0, so we can just return one of them
            return Quaternion(*q1)

    @staticmethod
    def torch_SLERP(q1, q2, t):
        cosTheta = torch.dot(q1, q2)
        if cosTheta < 1:
            theta = torch.arccos(cosTheta)
            invSinTheta = 1. / torch.sin(theta)
            c0 = torch.sin((1 - t) * theta) * invSinTheta
            c1 = torch.sin(t * theta) * invSinTheta
            q_ = c0 * q1 + c1 * q2
            return q_ / torch.norm(q_)
        else:
            return q1


    def mean(qs, ws=None):
        # Assert all quaternions are facing the same way
        for i in range(1, len(qs)):
            q_prev = np.array([qs[i-1][0], qs[i-1][1], qs[i-1][2], qs[i-1][3]])
            q = np.array([qs[i][0], qs[i][1], qs[i][2], qs[i][3]])
            cosTheta = np.dot(q_prev, q)
            if cosTheta < 0:
                qs[i] = Quaternion(-q[0], -q[1], -q[2], -q[3])

        if ws is None:
            ws = np.ones(len(qs)) / len(qs)
        else:
            assert np.abs(np.sum(ws) - 1.0) < 1e-4

        for i in range(1, len(qs)):
            t = ws[i] / (ws[i-1] + ws[i])
            qs[i] = Quaternion.SLERP(qs[i-1], qs[i], t)
            ws[i] = 1 - np.sum(ws[i+1:])
        return qs[-1]

    def prod(self, r):
        t0 = r[0] * self.q[0] - r[1]*self.q[1] - r[2] * self.q[2] - r[3] * self.q[3]
        t1 = r[0] * self.q[1] + r[1]*self.q[0] - r[2] * self.q[3] + r[3] * self.q[2]
        t2 = r[0] * self.q[2] + r[1]*self.q[3] + r[2] * self.q[0] - r[3] * self.q[1]
        t3 = r[0] * self.q[3] - r[1]*self.q[2] + r[2] * self.q[1] + r[3] * self.q[0]
        return np.array([t0, t1, t2, t3])

    @staticmethod
    def from_matrix(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr+1) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] - m[1, 0]) / S
            qz = (m[0, 2] - m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] - m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] - m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] - m[2, 0]) / S
            qy = (m[1, 2] - m[2, 1]) / S
            qz = 0.25 * S
        return Quaternion(qw, qx, qy, qz)
