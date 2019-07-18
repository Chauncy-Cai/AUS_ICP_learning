from optimizer import *

def main():
    opt = Optimizer()
    n = 100
    m = 100
    P1, Q1, P2, C2, A2 = opt.GenerateInput(n, m)
    C2 = np.concatenate([C2, np.ones((m, 1))], axis=1)
    for i in range(50):
        loss = opt.Loss(P1, Q1, P2, C2, A2)
        R = opt.OptimizeRotation(P1, Q1, P2, C2, A2)
        delta_angle = np.arccos(np.clip((np.sum(np.diag(R))-1.0)/2, -1, 1))/np.pi*180.0
        P1 = (R.dot(P1.T)).T
        P2 = (R.dot(P2.T)).T
        t = opt.OptimizeTranslation(P1, Q1, P2, C2, A2)
        delta_t = np.linalg.norm(t, 2)
        P1 += t[np.newaxis, :]
        P2 += t[np.newaxis, :]
        print('iter=%d, loss=%f, angle=%f, delta_t=%f' % (i, loss, delta_angle, delta_t))

if __name__ == '__main__':
    main()
