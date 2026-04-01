"""Time OSQP examples using Python osqp (C-backed)."""
import osqp
import numpy as np
from scipy import sparse
import time

def timeit(label, fn, repeats=20):
    """Run fn() repeats times, report median."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times)//2]
    print(f"  {label:30s}  {med*1000:8.2f} ms  (median of {repeats})")
    return med

results = {}

# ============================================================
# Example 1: Setup and Solve (2x2 QP)
# ============================================================
P1 = sparse.csc_matrix([[4, 1], [1, 2]])
q1 = np.array([1., 1.])
A1 = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]], dtype=float)
l1 = np.array([1., 0., 0.])
u1 = np.array([1., 0.7, 0.7])

def ex1():
    prob = osqp.OSQP()
    prob.setup(P1, q1, A1, l1, u1, alpha=1.0, verbose=False)
    return prob.solve()

results['ex1'] = timeit('Ex1: Setup+Solve (2x2)', ex1)

# ============================================================
# Example 2: Update Vectors
# ============================================================
def ex2():
    prob = osqp.OSQP()
    prob.setup(P1, q1, A1, l1, u1, verbose=False)
    prob.solve()
    prob.update(q=np.array([2.,3.]), l=np.array([2.,-1.,-1.]), u=np.array([2.,2.5,2.5]))
    return prob.solve()

results['ex2'] = timeit('Ex2: Update Vectors', ex2)

# ============================================================
# Example 3: Update Matrices
# ============================================================
def ex3():
    prob = osqp.OSQP()
    prob.setup(P1, q1, A1, l1, u1, verbose=False)
    prob.solve()
    P_new = sparse.csc_matrix([[5, 1.5], [1.5, 1]])
    A_new = sparse.csc_matrix([[1.2, 1.1], [1.5, 0], [0, 0.8]])
    prob.update(Px=sparse.triu(P_new).data, Ax=A_new.data)
    return prob.solve()

results['ex3'] = timeit('Ex3: Update Matrices', ex3)

# ============================================================
# Example 4: Least Squares (50 vars)
# ============================================================
rng = np.random.RandomState(1)
m4, n4 = 30, 20
Ad4 = sparse.random(m4, n4, density=0.7, format='csc', random_state=rng)
b4 = rng.randn(m4)
P4 = sparse.block_diag([sparse.csc_matrix((n4, n4)), sparse.eye(m4)], format='csc')
q4 = np.zeros(n4 + m4)
A4 = sparse.vstack([
    sparse.hstack([Ad4, -sparse.eye(m4)]),
    sparse.hstack([sparse.eye(n4), sparse.csc_matrix((n4, m4))])
], format='csc')
l4 = np.concatenate([b4, np.zeros(n4)])
u4 = np.concatenate([b4, np.ones(n4)])

def ex4():
    prob = osqp.OSQP()
    prob.setup(P4, q4, A4, l4, u4, verbose=False)
    return prob.solve()

results['ex4'] = timeit('Ex4: Least Squares (50 vars)', ex4)

# ============================================================
# Example 5: MPC (172 vars, 15 solves)
# ============================================================
Ad5 = np.array([
    [1,0,0,0,0,0,0.1,0,0,0,0,0],[0,1,0,0,0,0,0,0.1,0,0,0,0],
    [0,0,1,0,0,0,0,0,0.1,0,0,0],[0.0488,0,0,1,0,0,0.0016,0,0,0.0992,0,0],
    [0,-0.0488,0,0,1,0,0,-0.0016,0,0,0.0992,0],[0,0,0,0,0,1,0,0,0,0,0,0.0992],
    [0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],[0.9734,0,0,0,0,0,0.0488,0,0,0.9846,0,0],
    [0,-0.9734,0,0,0,0,0,-0.0488,0,0,0.9846,0],[0,0,0,0,0,0,0,0,0,0,0,0.9846]])
Bd5 = np.array([
    [0,-0.0726,0,0.0726],[-0.0726,0,0.0726,0],[-0.0152,0.0152,-0.0152,0.0152],
    [0,-0.0006,0,0.0006],[0.0006,0,-0.0006,0],[0.0106,0.0106,0.0106,0.0106],
    [0,-1.4512,0,1.4512],[-1.4512,0,1.4512,0],[-0.3049,0.3049,-0.3049,0.3049],
    [0,-0.0236,0,0.0236],[0.0236,0,-0.0236,0],[0.2107,0.2107,0.2107,0.2107]])
nx5, nu5 = 12, 4; N5 = 10; u0 = 10.5916
umin5 = np.array([9.6,9.6,9.6,9.6]) - u0; umax5 = np.array([13,13,13,13]) - u0
xmin5 = np.array([-np.pi/6,-np.pi/6]+[-np.inf]*3+[-1]+[-np.inf]*6)
xmax5 = np.array([np.pi/6,np.pi/6]+[np.inf]*4+[np.inf]*6)
Q5 = np.diag([0,0,10,10,10,10,0,0,0,5,5,5]); R5 = 0.1*np.eye(4)
xr5 = np.array([0,0,1,0,0,0,0,0,0,0,0,0])

def build_mpc():
    P = sparse.block_diag([sparse.kron(sparse.eye(N5),Q5),Q5,sparse.kron(sparse.eye(N5),R5)],format='csc')
    q = np.concatenate([np.tile(-Q5@xr5,N5),-Q5@xr5,np.zeros(N5*nu5)])
    Ax = sparse.kron(sparse.eye(N5+1),-sparse.eye(nx5))+sparse.kron(sparse.diags([1],[-1],shape=(N5+1,N5+1)),sparse.csc_matrix(Ad5))
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1,N5)),sparse.eye(N5)]),sparse.csc_matrix(Bd5))
    Aeq = sparse.hstack([Ax,Bu])
    x0 = np.zeros(nx5)
    leq = np.concatenate([-x0,np.zeros(N5*nx5)]); ueq = leq.copy()
    Aineq = sparse.eye((N5+1)*nx5+N5*nu5)
    lineq = np.concatenate([np.tile(xmin5,N5+1),np.tile(umin5,N5)])
    uineq = np.concatenate([np.tile(xmax5,N5+1),np.tile(umax5,N5)])
    A = sparse.vstack([Aeq,Aineq],format='csc')
    l = np.concatenate([leq,lineq]); u = np.concatenate([ueq,uineq])
    return P, q, A, l, u

def ex5():
    P,q,A,l,u = build_mpc()
    prob = osqp.OSQP()
    prob.setup(P,q,A,l,u,warm_starting=True,verbose=False)
    x0 = np.zeros(12)
    for _ in range(15):
        res = prob.solve()
        ctrl = res.x[(N5+1)*nx5:(N5+1)*nx5+nu5]
        x0 = Ad5@x0 + Bd5@ctrl
        l[:nx5] = -x0; u[:nx5] = -x0
        prob.update(l=l, u=u)
    return res

results['ex5'] = timeit('Ex5: MPC (172 vars, 15 steps)', ex5, repeats=10)

# ============================================================
# Example 6: Huber Fitting (310 vars)
# ============================================================
rng6 = np.random.RandomState(1)
n6, m6 = 10, 100
Ad6 = sparse.random(m6, n6, density=0.5, format='csc', random_state=rng6)
x_true6 = rng6.randn(n6) / np.sqrt(n6)
ind95_6 = rng6.rand(m6) > 0.95
b6 = Ad6 @ x_true6 + 10*rng6.rand(m6)*ind95_6 + 0.5*rng6.randn(m6)*(1-ind95_6)

P6 = sparse.block_diag([sparse.csc_matrix((n6,n6)),2*sparse.eye(m6),sparse.csc_matrix((2*m6,2*m6))],format='csc')
q6 = np.concatenate([np.zeros(m6+n6),2*np.ones(2*m6)])
A6 = sparse.vstack([
    sparse.hstack([Ad6,-sparse.eye(m6),-sparse.eye(m6),sparse.eye(m6)]),
    sparse.hstack([sparse.csc_matrix((m6,n6)),sparse.csc_matrix((m6,m6)),sparse.eye(m6),sparse.csc_matrix((m6,m6))]),
    sparse.hstack([sparse.csc_matrix((m6,n6)),sparse.csc_matrix((m6,m6)),sparse.csc_matrix((m6,m6)),sparse.eye(m6)])
],format='csc')
l6 = np.concatenate([b6,np.zeros(2*m6)])
u6 = np.concatenate([b6,np.inf*np.ones(2*m6)])

def ex6():
    prob = osqp.OSQP()
    prob.setup(P6, q6, A6, l6, u6, verbose=False)
    return prob.solve()

results['ex6'] = timeit('Ex6: Huber (310 vars)', ex6)

# ============================================================
# Example 7: SVM (1010 vars)
# ============================================================
rng7 = np.random.RandomState(1)
n7, m7 = 10, 1000; N7 = 500
A_upp7 = sparse.random(N7, n7, density=0.5, format='csc', random_state=rng7)
A_low7 = sparse.random(N7, n7, density=0.5, format='csc', random_state=np.random.RandomState(2))
Ad7 = sparse.vstack([A_upp7/np.sqrt(n7)+(A_upp7!=0)/n7, A_low7/np.sqrt(n7)-(A_low7!=0)/n7], format='csc')
b7 = np.concatenate([np.ones(N7),-np.ones(N7)])
P7 = sparse.block_diag([sparse.eye(n7),sparse.csc_matrix((m7,m7))],format='csc')
q7 = np.concatenate([np.zeros(n7),np.ones(m7)])
A7 = sparse.vstack([
    sparse.hstack([sparse.diags(b7)@Ad7,-sparse.eye(m7)]),
    sparse.hstack([sparse.csc_matrix((m7,n7)),sparse.eye(m7)])
],format='csc')
l7 = np.concatenate([-np.inf*np.ones(m7),np.zeros(m7)])
u7 = np.concatenate([-np.ones(m7),np.inf*np.ones(m7)])

def ex7():
    prob = osqp.OSQP()
    prob.setup(P7, q7, A7, l7, u7, verbose=False)
    return prob.solve()

results['ex7'] = timeit('Ex7: SVM (1010 vars)', ex7)

# ============================================================
# Example 8: Lasso (11 solves, 1020 vars)
# ============================================================
rng8 = np.random.RandomState(1)
n8, m8 = 10, 1000
Ad8 = sparse.random(m8, n8, density=0.5, format='csc', random_state=rng8)
x_true8 = (rng8.randn(n8) > 0.8) * rng8.randn(n8) / np.sqrt(n8)
b8 = Ad8 @ x_true8 + 0.5 * rng8.randn(m8)
gammas8 = np.linspace(1, 10, 11)
P8 = sparse.block_diag([sparse.csc_matrix((n8,n8)),sparse.eye(m8),sparse.csc_matrix((n8,n8))],format='csc')
q8 = np.zeros(2*n8+m8)
A8 = sparse.vstack([
    sparse.hstack([Ad8,-sparse.eye(m8),sparse.csc_matrix((m8,n8))]),
    sparse.hstack([sparse.eye(n8),sparse.csc_matrix((n8,m8)),-sparse.eye(n8)]),
    sparse.hstack([sparse.eye(n8),sparse.csc_matrix((n8,m8)),sparse.eye(n8)])
],format='csc')
l8 = np.concatenate([b8,-np.inf*np.ones(n8),np.zeros(n8)])
u8 = np.concatenate([b8,np.zeros(n8),np.inf*np.ones(n8)])

def ex8():
    prob = osqp.OSQP()
    prob.setup(P8, q8, A8, l8, u8, warm_starting=True, verbose=False)
    for g in gammas8:
        prob.update(q=np.concatenate([np.zeros(n8+m8),g*np.ones(n8)]))
        prob.solve()

results['ex8'] = timeit('Ex8: Lasso (1020v, 11 solves)', ex8, repeats=10)

# Summary
print("\n" + "="*60)
print(f"  Python osqp {osqp.__version__} (C-backed) — Timing Summary")
print("="*60)
total = sum(results.values())
print(f"  Total (sum of medians): {total*1000:.2f} ms")
