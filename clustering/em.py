from numpy.linalg import svd, inv, eig, norm
from math import exp, log


def initialize_mixtures(X, y, M):
    mix = []
    for i in range(len(X)):
        mix.append(M * int(y[i]) + random.randint(0,M-1))
    return np.array(mix)
def probabilities(X, mixtures, M):
    p = []
    d = len(X.T)
    n = len(X)
    ind = []
    for j in range(d):
        pj = np.zeros(10*M)
        p.append(pj)
    for m in range(10*M):
        indices = [i for i, x in enumerate(mixtures) if x == m]
        ind.append(len(indices))
        for i in indices:
            for j in range(d):
                p[j][m] += X[i][j]
        for j in range(d):
            p[j][m] = (p[j][m]+1)/(len(indices)+2)
    return np.array(p)
def likelihood(xi, pim, pm):
    S = 0
    for j in range(len(xi)):
        S += xi[j]*log(pm[j]) + (1-xi[j])*log(1-pm[j])
    return S + log(pim)
def weights(X, p, pi, M):
    n = len(X)
    d = len(X.T)
    l = []
    for m in range(10*M):
        l_m = []
        for i in range(n):
            l_m.append(likelihood(X[i], pi[m], p.T[m]))
        l.append(l_m)
    l = np.array(l)
    w = []
    for m in range(10*M):
        w_m = []
        for i in range(n):
            L = max(l.T[i])
            sum_demon = sum([exp(lm - L) for lm in l.T[i]])
            w_m.append((exp(l[m][i] - L))/sum_demon)
        w.append(w_m)
    return np.array(w)
def fix_log(X, p, pi, M):
    S = 0
    n = len(X)
    for i in range(n):
        sum_log = 0
        for m in range(10*M):
            sum_log += exp(likelihood(X[i], pi[m], p.T[m]))
        S += log(sum_log)
    return S
def iterations(X, p, pi, w, M):
    l = fix_log(test_data, p, pi, M)
    p, pi, w = m_step(X, p, pi, w, M)
    lnew = fix_log(test_data, p, pi, M)
    while (lnew-l > 10):
        l = lnew
        p, pi, w = m_step(X, p, pi, w, M)
        lnew = fix_log(test_data, p, pi, M)
    return p, pi, w
def e_step(X, mixtures, M):
    pi = np.zeros(M*10)
    for i in range(len(X)):
        pi[mixtures[i]] += 1
    pi = (pi+1)/(len(X)+2)

    p = []
    d = len(X.T)
    n = len(X)
    ind = []
    for j in range(d):
        pj = np.zeros(10*M)
        p.append(pj)
    for m in range(10*M):
        indices = [i for i, x in enumerate(mixtures) if x == m]
        ind.append(len(indices))
        for i in indices:
            for j in range(d):
                p[j][m] += X[i][j]
        for j in range(d):
            p[j][m] = (p[j][m]+1)/(len(indices)+2)
    p = np.array(p)

    w = weights(X, p, pi, M)
    return p, pi, w
def m_step(X, p, pi, w, M):
    n = len(X)
    d = len(X.T)
    for m in range(10*M):
        sumw = 0
        for i in range(n):
            sumw += w[m][i]
        pi[m] = (sumw+1) / (n+2)

    for j in range(d):
        for m in range(10*M):
            sumw = 0
            sumwx = 0
            for i in range(n):
                sumw += w[m][i]
                sumwx += w[m][i] * X[i][j]
        p[j][m] = (sumwx+1)/(sumw+2)
    w = weights(X, p, pi, M)
    return p, pi, w

def EM(X, Y, M):
    mixtures = initialize_mixtures(X, Y, M)
    p, pi, w = e_step(X, mixtures, M)
    if M>1:
        p, pi, w = iterations(X, p, pi, w, M)
    return p, pi
