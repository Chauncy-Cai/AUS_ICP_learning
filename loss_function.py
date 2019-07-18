#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/7/16 0:16
# @Author: csc
# @File  : loss_function.py
import numpy as np

def ocross(H,p):
    '''
    :param H: 3by3 matrix
    :param p: 3by1 matrix
    :return: H \otime p
    '''
    p=p.T
    #print(p)
    h0_cross_p = cross(H[0]).dot(p).T[0]
    h1_cross_p = cross(H[1]).dot(p).T[0]
    h2_cross_p = cross(H[2]).dot(p).T[0]
    #print("exam",h0_cross_p)
    H_cross_p_T = np.array(
                [h0_cross_p,
                 h1_cross_p,
                 h2_cross_p])
    return H_cross_p_T.T

def cross(x):
    '''
    :param x: 3d vector
    '''
    u,v,w = x[0],x[1],x[2]
    cross_x_half = np.array([
        [0,-w,v],
        [0,0,-u],
        [0,0,0]])
    return cross_x_half-(cross_x_half.T)

def r2R(r):##?????
    theta = np.linalg.norm(r, 2)
    k = r / theta
    R = np.cos(theta) * np.eye(3) + np.sin(theta) * cross(k) + (1 - np.cos(theta)) * np.outer(k, k)
    return R

def r2R1(r):
    tempr = np.eye(3)-cross(r)
    u,d,v = np.linalg.svd(tempr)
    newr = u.dot(v)
    #print(tempr,newr)
    #newr = tempr
    return newr

class lossFunction(object):
    '''
    P: numpy n*3;   pointset of P
    Q: numpy n*3;   pointset of Q(target point cloud)
    A: numpy m*3*4; the reflect matrix (from 3d to 2d)
    Pc: numpy m*3;  pointset of P (associate with image)
    C: numpy m*2;   points in image

    REQUIRE:P,Q match
            A,Pc,A match
    '''
    def __init__(self):
        self.n = -1
        self.P = -1
        self.Q = -1
        self.m = -1
        self.A = -1
        self.Pc_4 = -1
        self.C_3 = -1
        self.Pc = -1
        self.C = -1
        self.loss = -1
        self.optloss = -1
        self.midloss = -1
        self.Pnew = -1
        self.Pcnew = -1

    def usingSample(self,n,m):
        self.n=n
        self.m=m
        self.P = np.random.rand(n,3)*100
        self.Q = np.random.rand(n,3)*100
        self.A = np.array([np.array([[1,3,9,-5],
                  [0,2,7,-3],
                  [0,0,1,-1]])]*m)
        self.Pc = np.random.rand(m,3)*100
        self.C = np.random.rand(m,2)*100
        self.Pc_4 = np.append(self.Pc, np.ones((self.m, 1)), axis=1)
        self.C_3 = np.append(self.C, np.ones((self.m, 1)), axis=1)
        self.loss = self.calLoss()

    def inputdata(self, P, Q, A, Pc, C):
        self.n = len(P)
        self.P = P
        self.Q = Q
        self.m = len(Pc)
        self.A = A
        self.Pc_4 = np.append(Pc, np.ones((self.m, 1)), axis=1)
        self.C_3 = np.append(C, np.ones((self.m, 1)), axis=1)
        self.Pc = Pc
        self.C = C
        self.loss = self.calLoss()

    def optimize(self):
        R = self.optimizeR()
        assert (R.shape == (3, 3))
        #t = self.optimizeT(R)
        t = np.zeros((3,1))
        assert (t.shape == (3, 1))
        return [R, t]

    def optimizeR(self):
        #H=(R0'Ej')EjR0 and R0=I thus H=Ej'Ej
        U = -np.sum(np.cross(self.Q, self.P), axis=0)  # R0 = I
        U = np.array([U]).T
        #print(U)
        assert (U.shape == (3, 1))
        V = np.zeros((3, 1))
        W = np.zeros((3, 3))
        for j in range(self.m):
            A =self.A[j]
            E = A[:, 0:3]
            e = np.array([A[:, 3]]).T
            c = np.array([self.C_3[j]]).T
            p = np.array([self.P[j]])
            H = np.dot(E.T,E)
            #calulate V
            temp1 = p.dot(H)[0] #p.dot(H)1*3 => vector3
            v1 = -cross(temp1).dot(p.T)
            assert(v1.shape == (3,1))
            v2 = cross(p[0]).dot(E).dot(e-c)
            assert (v2.shape == (3, 1))
            V += -(v1+v2)
            #calculate W
            temp2 = cross(p[0]).dot(H)
            W += ocross(temp2,p)
        assert (V.shape == (3, 1))
        assert (W.shape == (3, 3))
        r = np.linalg.solve(W + W.T, 2*(U + V))  ## (W+W') t = -(U+V)'
        assert (r.shape == (3,1))
        # exam
        g = 2*np.linalg.inv(W+W.T).dot(V+U)
        print("minus",2*g.T.dot(U+V)-g.T.dot(W).dot(r))
        #self.examr(r)
        #self.beforeNlater(r)
        print(r.T[0])
        r = r2R1(r.T[0])
        return r

    def optimizeT(self, R):
        U = 2 * np.sum(self.P.dot(R.T) - self.Q, axis=0)
        U = np.array([U])
        assert (U.shape == (1, 3))
        V = np.zeros((1, 3))
        W = np.zeros((3, 3))
        for j in range(self.m):
            A = self.A[j]
            E = A[:, 0:3]
            e = np.array([A[:, 3]]).T
            c = np.array([self.C_3[j]]).T
            p = np.array([self.P[j]])
            V += 2 * p.dot(R.T).dot(E.T).dot(E) + 2 * (e - c).T.dot(E)
            W += E.T.dot(E)
        assert (V.shape == (1, 3))
        W += self.n * np.eye(3)
        #W is sym
        t = np.linalg.solve(2*W, -(U + V).T)  ## (W+W') t = -(U+V)'
        ##check
        #self.examt(R,t)
        return t

    def calLoss(self):
        pointCloudsLoss = np.linalg.norm(self.P - self.Q)**2
        pointCloud2ImageLoss = 0
        for i in range(self.m):
            pc4 = np.array([self.Pc_4[i]]).T
            c3 = np.array([self.C_3[i]]).T
            pointCloud2ImageLoss += np.linalg.norm(self.A[i].dot(pc4) - c3)**2
        sumLoss = pointCloudsLoss + pointCloud2ImageLoss
        return sumLoss

    def calOptloss(self, R, t):
        #print(self.P.shape,t.shape)
        PAfter = (self.P.dot(R.T)+t.T)
        PcAfter = (self.Pc.dot(R.T)+t.T)
        self.Pnew = PAfter
        self.Pcnew = PcAfter
        assert (PcAfter.shape == self.Pc.shape)
        Pc4After = np.append(PcAfter, np.ones((self.m, 1)), axis=1)
        pointCloudsLoss = np.linalg.norm(PAfter - self.Q)**2
        pointCloud2ImageLoss = 0
        for i in range(self.m):
            pointCloud2ImageLoss += np.linalg.norm(self.A[i].dot(Pc4After[i].T) - self.C_3[i].T)**2
        sumOptLoss = pointCloudsLoss + pointCloud2ImageLoss
        self.optloss = sumOptLoss
        return

    def calMidloss(self,R):
        PAfter = (self.P.dot(R.T))
        PcAfter = (self.Pc.dot(R.T))
        #PAfter = self.P
        #PcAfter = self.Pc
        #print(PcAfter.shape,self.Pc.shape)
        assert (PcAfter.shape==self.Pc.shape)
        Pc4After = np.append(PcAfter, np.ones((self.m, 1)), axis=1)
        pointCloudsLoss = np.linalg.norm(PAfter - self.Q)**2
        pointCloud2ImageLoss = 0
        for i in range(self.m):
            pointCloud2ImageLoss += np.linalg.norm(self.A[i].dot(Pc4After[i].T) - self.C_3[i].T)**2
        sumMidLoss = pointCloudsLoss + pointCloud2ImageLoss
        self.midloss = sumMidLoss
        return

    def showLoss(self,R,t):
        print("before_loss\t" + str(self.loss) + "\tmid_loss\t" + str(self.midloss) + "\topt_loss\t" + str(self.optloss))
        if self.loss <self.midloss:
            print("1  should be+ :"+str(self.loss - self.midloss))
        #if self.midloss < self.optloss:
        #    print("2  should be+ :" + str(self.midloss - self.optloss))
        return

    def examr(self,r):
        #print(r)
        item1=0
        item10 = 0
        #r=np.zeros((3))
        crossr = cross(r)
        for i in range(self.n):
            p = np.array([self.P[i]])
            q = np.array([self.Q[i]])
            item1 += -2*q.dot(crossr).dot(p.T)
            item10 += 2*r.T.dot(cross(q[0]).dot(p.T))
        #print("item1~",np.abs(item1-item10)<10-5)
        item2,item20=0,0
        item3,item30=0,0
        item4,item40 = 0,0
        for j in range(self.m):
            A =self.A[j]
            E = A[:, 0:3]
            e = np.array([A[:, 3]]).T
            c = np.array([self.C_3[j]]).T
            p = np.array([self.P[j]])
            H = np.dot(E.T,E)
            item2 += 2*p.dot(H).dot(crossr).dot(p.T)
            temp = p.dot(H)
            item20 +=-2*r.T.dot(cross(temp[0])).dot(p.T)
            item3 += p.dot(-crossr).dot(H).dot(crossr).dot(p.T)
            temp = cross(p[0]).dot(H)
            item30 +=-r.T.dot(ocross(temp,p)).dot(r)
            #另外一种算法
            item4 += 2*p.dot(crossr.T).dot(E).dot(e-c)
            item40 += 2*r.T.dot(cross(p[0])).dot(E).dot(e-c)
        #print("item2`",np.abs(item2-item20)<10-5)
        #print("item3`", np.abs(item3-item30)<10-5)
        #print("item4`", np.abs(item4-item40)<10-5)
        #print(item1,item2,item3,item4)
        print("sum item:", str(item1+item2+item3+item4))

    def beforeNlater(self,r):
        I = np.eye(3)
        crossr = cross(r)
        print(r.T[0])
        R = r2R1(r.T[0])
        #print(r2R1(r.T[0]))
        item0r=0
        item0o=0
        for i in range(self.n):
            p = np.array([self.P[i]])
            q = np.array([self.Q[i]])
            item0r += np.linalg.norm(R.dot(p.T)-q.T)**2
            item0o += np.linalg.norm(p.T - q.T) ** 2
            #item0r += -2*q.dot(R).dot(p.T)
            #item0o += -2*q.dot(I).dot(p.T)
        item1r=0
        item1o=0
        for j in range(self.m):
            A =self.A[j]
            E = A[:, 0:3]
            e = np.array([A[:, 3]]).T
            c = np.array([self.C_3[j]]).T
            p = np.array([self.P[j]])
            H = np.dot(E.T,E)
            #item1r += p.dot(R).dot(E.T).dot(E).dot(R).dot(p.T)+2*p.dot(R.T).dot(E.T).dot(e-c)
            #item1o += p.dot(E.T).dot(E).dot(p.T)+2*p.dot(E.T).dot(e-c)
            item1r +=p.dot((I+crossr).T).dot(H).dot(I+crossr).dot(p.T)
            item1r +=2*p.dot((I+crossr).T).dot(E).dot(e-c)
            item1o += p.dot((I).T).dot(H).dot(I).dot(p.T)
            item1o += 2 * p.dot((I).T).dot(E).dot(e - c)
        print("ori:" +str(item1o+item0o),"  new:"+str(item0r+item1r),"VAR:",str(item1o+item0o-(item0r+item1r)))