"""Generate reference solutions for OSQP examples using Python osqp."""
import osqp
import numpy as np
from scipy import sparse
import json

results = {}

# ============================================================
# Example 1: Setup and Solve (from osqp.org)
# ============================================================
P = sparse.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

prob = osqp.OSQP()
prob.setup(P, q, A, l, u, alpha=1.0, verbose=False)
res = prob.solve()
results['setup_and_solve'] = {
    'x': res.x.tolist(),
    'y': res.y.tolist(),
    'obj': float(res.info.obj_val),
    'status': res.info.status,
}
print(f"Example 1 (Setup & Solve): x={res.x}, obj={res.info.obj_val:.6f}, status={res.info.status}")

# ============================================================
# Example 2: Update Vectors
# ============================================================
prob2 = osqp.OSQP()
prob2.setup(P, q, A, l, u, verbose=False)
res2a = prob2.solve()
results['update_vectors_before'] = {
    'x': res2a.x.tolist(),
    'y': res2a.y.tolist(),
    'obj': float(res2a.info.obj_val),
    'status': res2a.info.status,
}

q_new = np.array([2, 3])
l_new = np.array([2, -1, -1])
u_new = np.array([2, 2.5, 2.5])
prob2.update(q=q_new, l=l_new, u=u_new)
res2b = prob2.solve()
results['update_vectors_after'] = {
    'x': res2b.x.tolist(),
    'y': res2b.y.tolist(),
    'obj': float(res2b.info.obj_val),
    'status': res2b.info.status,
}
print(f"Example 2 (Update Vectors): x={res2b.x}, obj={res2b.info.obj_val:.6f}, status={res2b.info.status}")

# ============================================================
# Example 3: Update Matrices
# ============================================================
prob3 = osqp.OSQP()
prob3.setup(P, q, A, l, u, verbose=False)
res3a = prob3.solve()

P_new = sparse.csc_matrix([[5, 1.5], [1.5, 1]])
A_new = sparse.csc_matrix([[1.2, 1.1], [1.5, 0], [0, 0.8]])
prob3.update(Px=sparse.triu(P_new).data, Ax=A_new.data)
res3b = prob3.solve()
results['update_matrices'] = {
    'x': res3b.x.tolist(),
    'y': res3b.y.tolist(),
    'obj': float(res3b.info.obj_val),
    'status': res3b.info.status,
}
print(f"Example 3 (Update Matrices): x={res3b.x}, obj={res3b.info.obj_val:.6f}, status={res3b.info.status}")

# ============================================================
# Example 4: Least Squares
# ============================================================
np.random.seed(1)
m_ls = 30
n_ls = 20
Ad_ls = sparse.random(m_ls, n_ls, density=0.7, format='csc', random_state=np.random.RandomState(1))
b_ls = np.random.RandomState(1).randn(m_ls)
# Regenerate with same seed sequence as MATLAB rng(1) won't match numpy, 
# so we just solve the example to confirm structure works.
# For true comparison we feed the same data in MATLAB.

P_ls = sparse.block_diag([sparse.csc_matrix((n_ls, n_ls)), sparse.eye(m_ls)])
q_ls = np.zeros(n_ls + m_ls)
A_ls = sparse.vstack([
    sparse.hstack([Ad_ls, -sparse.eye(m_ls)]),
    sparse.hstack([sparse.eye(n_ls), sparse.csc_matrix((n_ls, m_ls))])
])
l_ls = np.concatenate([b_ls, np.zeros(n_ls)])
u_ls = np.concatenate([b_ls, np.ones(n_ls)])

prob4 = osqp.OSQP()
prob4.setup(P_ls, q_ls, A_ls, l_ls, u_ls, verbose=False)
res4 = prob4.solve()
results['least_squares'] = {
    'obj': float(res4.info.obj_val),
    'status': res4.info.status,
    'x_first5': res4.x[:5].tolist(),
}
print(f"Example 4 (Least Squares): obj={res4.info.obj_val:.6f}, status={res4.info.status}")

# ============================================================
# Example 5: MPC
# ============================================================
Ad_mpc = np.array([
    [1,0,0,0,0,0,0.1,0,0,0,0,0],
    [0,1,0,0,0,0,0,0.1,0,0,0,0],
    [0,0,1,0,0,0,0,0,0.1,0,0,0],
    [0.0488,0,0,1,0,0,0.0016,0,0,0.0992,0,0],
    [0,-0.0488,0,0,1,0,0,-0.0016,0,0,0.0992,0],
    [0,0,0,0,0,1,0,0,0,0,0,0.0992],
    [0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0.9734,0,0,0,0,0,0.0488,0,0,0.9846,0,0],
    [0,-0.9734,0,0,0,0,0,-0.0488,0,0,0.9846,0],
    [0,0,0,0,0,0,0,0,0,0,0,0.9846],
])
Bd_mpc = np.array([
    [0,-0.0726,0,0.0726],
    [-0.0726,0,0.0726,0],
    [-0.0152,0.0152,-0.0152,0.0152],
    [0,-0.0006,-0.0000,0.0006],
    [0.0006,0,-0.0006,0],
    [0.0106,0.0106,0.0106,0.0106],
    [0,-1.4512,0,1.4512],
    [-1.4512,0,1.4512,0],
    [-0.3049,0.3049,-0.3049,0.3049],
    [0,-0.0236,0,0.0236],
    [0.0236,0,-0.0236,0],
    [0.2107,0.2107,0.2107,0.2107],
])
nx, nu = Bd_mpc.shape
u0 = 10.5916
umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
umax = np.array([13, 13, 13, 13]) - u0
xmin = np.array([-np.pi/6, -np.pi/6, -np.inf, -np.inf, -np.inf, -1] + [-np.inf]*6)
xmax = np.array([np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf] + [np.inf]*6)
Q = np.diag([0,0,10,10,10,10,0,0,0,5,5,5])
QN = Q.copy()
R = 0.1 * np.eye(4)
x0_mpc = np.zeros(12)
xr = np.array([0,0,1,0,0,0,0,0,0,0,0,0])
N_mpc = 10

P_mpc = sparse.block_diag([
    sparse.kron(sparse.eye(N_mpc), Q), QN,
    sparse.kron(sparse.eye(N_mpc), R)
])
q_mpc = np.concatenate([np.tile(-Q @ xr, N_mpc), -QN @ xr, np.zeros(N_mpc * nu)])
Ax_mpc = sparse.kron(sparse.eye(N_mpc+1), -sparse.eye(nx)) + \
         sparse.kron(sparse.diags([1], [-1], shape=(N_mpc+1, N_mpc+1)), sparse.csc_matrix(Ad_mpc))
Bu_mpc = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N_mpc)), sparse.eye(N_mpc)]), sparse.csc_matrix(Bd_mpc))
Aeq = sparse.hstack([Ax_mpc, Bu_mpc])
leq = np.concatenate([-x0_mpc, np.zeros(N_mpc * nx)])
ueq = leq.copy()
Aineq = sparse.eye((N_mpc+1)*nx + N_mpc*nu)
lineq = np.concatenate([np.tile(xmin, N_mpc+1), np.tile(umin, N_mpc)])
uineq = np.concatenate([np.tile(xmax, N_mpc+1), np.tile(umax, N_mpc)])
A_mpc = sparse.vstack([Aeq, Aineq])
l_mpc = np.concatenate([leq, lineq])
u_mpc = np.concatenate([ueq, uineq])

prob5 = osqp.OSQP()
prob5.setup(P_mpc, q_mpc, A_mpc, l_mpc, u_mpc, warm_starting=True, verbose=False)

nsim = 15
x0_sim = np.zeros(12)
mpc_ctrls = []
for i in range(nsim):
    res5 = prob5.solve()
    if res5.info.status != 'solved':
        print(f"MPC step {i}: OSQP did not solve: {res5.info.status}")
        break
    ctrl = res5.x[(N_mpc+1)*nx:(N_mpc+1)*nx+nu]
    mpc_ctrls.append(ctrl.tolist())
    x0_sim = Ad_mpc @ x0_sim + Bd_mpc @ ctrl
    l_mpc[:nx] = -x0_sim
    u_mpc[:nx] = -x0_sim
    prob5.update(l=l_mpc, u=u_mpc)

results['mpc'] = {
    'final_state': x0_sim.tolist(),
    'num_steps': len(mpc_ctrls),
    'first_ctrl': mpc_ctrls[0] if mpc_ctrls else [],
    'last_ctrl': mpc_ctrls[-1] if mpc_ctrls else [],
}
print(f"Example 5 (MPC): {nsim} steps, final x3={x0_sim[2]:.6f} (target=1.0)")

# Save results for MATLAB comparison
with open('/tmp/osqp_reference_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nReference results saved to /tmp/osqp_reference_results.json")
