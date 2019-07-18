import numpy as np

def cross_op(x):
    '''
    :param x: 3d vector
    '''
    u,v,w = x[0],x[1],x[2]
    cross_x_half = np.array([
        [0,-w,v],
        [0,0,-u],
        [0,0,0]])
    return cross_x_half-(cross_x_half.T)

def Vec2Rot(r):
    theta = np.linalg.norm(r, 2)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    R = np.cos(theta) * np.eye(3) + np.sin(theta) * cross_op(k) + (1 - np.cos(theta)) * np.outer(k, k)
    return R

class Optimizer:
    def __init__(self):
        pass

    def GenerateInput(self, n, m):
        P1 = np.random.rand(n,3)*100
        Q1 = np.random.rand(n,3)*100
        A0 = np.array([[1,3,9,-5],
                       [0,2,7,-3],
                       [0,0,1,-1]])
        A2 = []
        for i in range(m):
            A2.append(A0)
        A2 = np.array(A2)

        P2 = np.random.rand(m,3)*100
        C2 = np.random.rand(m,2)*100
        return P1, Q1, P2, C2, A2

    """
    Minimize sum_i |R p1i-q1i|^2 + sum_j |A2j [Rp2j; 1] - c2j|^2
    <=> Min  sum_i fi^T r + sum_j |Bj r + sj|^2
    <=> Min  r^THr + g^Tr
    steps:
        0. compute fi, Bj, sj
            fi = p1i+t-q1i
    """
    def OptimizeRotation(self, P1, Q1, P2, C2, A2):
        n1 = P1.shape[0]
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for i in range(n1):
            g += np.cross(Q1[i, :], P1[i, :])

        n2 = P2.shape[0]
        for j in range(n2):
            Aj = A2[j, :, :] # [3, 4]
            Ej = Aj[:3, :3] # [3, 3]
            ej = Aj[:3, 3] # [3]
            pj = P2[j, :]
            cj = C2[j, :]
            """ |Ej R pj + ej - cj|^2 """
            """ |Ej pj + Ej (rXpj) + ej - cj|^2 """
            sj = Ej.dot(pj) + ej - cj
            pjX = cross_op(pj)
            Bj = -Ej.dot(pjX)
            H += Bj.T.dot(Bj)
            g += Bj.T.dot(sj)

        r = -np.linalg.solve(H, g)
        R = Vec2Rot(r)
        return R

    """
    Minimize sum_i |p1i +t-q1i|^2 + sum_j |A2j [p2j+t; 1] - c2j|^2
    <=> Min  t^THt + g^Tt
    """
    def OptimizeTranslation(self, P1, Q1, P2, C2, A2):
        n1 = P1.shape[0]
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for i in range(n1):
            g += (P1[i, :]- Q1[i, :])
            H += np.eye(3)

        n2 = P2.shape[0]
        for j in range(n2):
            Aj = A2[j, :, :] # [3, 4]
            Ej = Aj[:3, :3] # [3, 3]
            ej = Aj[:3, 3] # [3]
            pj = P2[j, :]
            cj = C2[j, :]
            """ |Ej t + ej - cj + Ej pj|^2 """
            H += Ej.T.dot(Ej)
            g += Ej.T.dot(ej-cj+Ej.dot(pj))

        t = -np.linalg.solve(H, g)
        return t

    def Loss(self, P1, Q1, P2, C2, A2):
        loss = np.linalg.norm(P1-Q1, 'fro')**2
        n2 = P2.shape[0]
        for j in range(n2):
            Aj = A2[j, :, :]
            pj = P2[j, :]
            cj = C2[j, :]
            temp = Aj[:3, :3].dot(pj) + Aj[:3, 3] - cj
            loss += np.linalg.norm(temp, 2)**2
        return loss

