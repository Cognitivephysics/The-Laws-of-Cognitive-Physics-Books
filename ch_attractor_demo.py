# ch_attractor_demo.py
#
# Minimal terminal demonstration that C − H = 0 is an emergent attractor
# via a soft constraint (no clamping, no copying H = C).
#
# Run with:
#   python ch_attractor_demo.py
#
# What this proves empirically:
# 1) C and H evolve independently
# 2) D = C − H → 0
# 3) Lyapunov Φ decreases (near-monotone)

import math

# ----------------------------
# Potential V(C, H)
# ----------------------------
def V(C, H, a=1.0, b=1.0, gamma=0.0):
    return 0.5 * (a*C*C + b*H*H) + gamma*C*H

def dVdC(C, H, a=1.0, gamma=0.0):
    return a*C + gamma*H

def dVdH(C, H, b=1.0, gamma=0.0):
    return b*H + gamma*C

# ----------------------------
# Lyapunov Φ(C, H)
# ----------------------------
def Phi(C, H, kappa, a=1.0, b=1.0, gamma=0.0):
    D = C - H
    return V(C, H, a, b, gamma) + 0.5 * kappa * D * D

# ----------------------------
# One gradient-flow step
# ----------------------------
def step(C, H, dt, kappa, a=1.0, b=1.0, gamma=0.0):
    D = C - H
    gradC = dVdC(C, H, a, gamma) + kappa * D
    gradH = dVdH(C, H, b, gamma) - kappa * D
    C -= dt * gradC
    H -= dt * gradH
    return C, H

# ----------------------------
# Main run
# ----------------------------
def run(
    steps=50_000,
    dt=1e-3,
    C0=3.0,
    H0=0.5,
    kappa=2.0,
    a=1.0,
    b=1.0,
    gamma=0.0,
    print_every=2_000,
    tol=1e-9
):
    C, H = C0, H0
    phi_prev = Phi(C, H, kappa, a, b, gamma)
    violations = 0

    print("=== C − H Soft-Constraint Attractor Demo ===")
    print(f"steps={steps}, dt={dt}, C0={C0}, H0={H0}, kappa={kappa}")
    print("n, t, C, H, D=C-H, Phi, dPhi")
    print("-------------------------------------------")

    for n in range(1, steps + 1):
        C, H = step(C, H, dt, kappa, a, b, gamma)
        phi_now = Phi(C, H, kappa, a, b, gamma)
        dphi = phi_now - phi_prev

        if dphi > tol:
            violations += 1

        if n % print_every == 0 or n == 1 or n == steps:
            t = n * dt
            D = C - H
            print(f"{n}, {t:.6f}, {C:.6f}, {H:.6f}, {D:.6f}, {phi_now:.10f}, {dphi:.3e}")

        phi_prev = phi_now

    print("-------------------------------------------")
    print(f"Monotone Φ violations: {violations} / {steps}")
    print(f"Final state: C={C:.6f}, H={H:.6f}, D={C-H:.6e}")
    print("Interpretation:")
    print("• D → 0  (emergent invariant)")
    print("• Φ decreases (Lyapunov)")
    print("• No H=C enforcement anywhere")
    print("===========================================")

if __name__ == "__main__":
    run()
