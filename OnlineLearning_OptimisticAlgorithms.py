import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
m, n = 5, 5  # Tailles des stratégies
T = 2000     # Nombre d'itérations
np.random.seed(42)
A = np.random.rand(m, n) * 2 - 1  # Matrice de gain aléatoire dans [-1,1]

# -------------------------------------------------------------------
# Algorithmes standards
# -------------------------------------------------------------------

def OMD_entropic(A, T, gamma=0.1, optimistic=False):
    a = np.ones(m)/m
    b = np.ones(n)/n
    R1, R2 = np.zeros(T), np.zeros(T)
    sum_grad1, sum_grad2 = 1e-8, 1e-8
    prev_grad1, prev_grad2 = np.zeros(m), np.zeros(n)

    for t in range(T):
        # Joueur 1
        grad1 = A @ b
        sum_grad1 += np.max(np.abs(grad1))**2
        gamma_t = gamma / np.sqrt(sum_grad1)
        
        if optimistic:
            update = 2 * grad1 - prev_grad1 if t > 0 else grad1
            a = a * np.exp(gamma_t * update)
        else:
            a = a * np.exp(gamma_t * grad1)
            
        a /= a.sum()
        prev_grad1 = grad1.copy()

        # Joueur 2
        grad2 = a @ A
        sum_grad2 += np.max(np.abs(grad2))**2
        gamma_t = gamma / np.sqrt(sum_grad2)
        
        if optimistic:
            update = 2 * grad2 - prev_grad2 if t > 0 else grad2
            b = b * np.exp(-gamma_t * update)
        else:
            b = b * np.exp(-gamma_t * grad2)
            
        b /= b.sum()
        prev_grad2 = grad2.copy()

        R1[t] = np.max(A @ b) - a @ A @ b
        R2[t] = a @ A @ b - np.min(a @ A)
    
    return np.cumsum(R1), np.cumsum(R2)

def regret_matching(A, T, plus=False, optimistic=False):
    R1, R2 = np.zeros((m, T)), np.zeros((n, T))
    a, b = np.ones(m)/m, np.ones(n)/n
    prev_r1, prev_r2 = np.zeros(m), np.zeros(n)

    for t in range(1, T):
        # Joueur 1
        r1 = A @ b - a @ A @ b
        if optimistic:
            R1[:, t] = R1[:, t-1] + (2*r1 - prev_r1 if t > 1 else r1)
        else:
            R1[:, t] = R1[:, t-1] + r1
            
        if plus:
            R1[:, t] = np.maximum(R1[:, t], 0)
            
        a = R1[:, t] / np.sum(R1[:, t]) if np.sum(R1[:, t]) > 0 else np.ones(m)/m
        prev_r1 = r1.copy()

        # Joueur 2
        r2 = -(a @ A - a @ A @ b)
        if optimistic:
            R2[:, t] = R2[:, t-1] + (2*r2 - prev_r2 if t > 1 else r2)
        else:
            R2[:, t] = R2[:, t-1] + r2
            
        if plus:
            R2[:, t] = np.maximum(R2[:, t], 0)
            
        b = R2[:, t] / np.sum(R2[:, t]) if np.sum(R2[:, t]) > 0 else np.ones(n)/n
        prev_r2 = r2.copy()
    
    return np.max(np.cumsum(R1, axis=1)), np.max(np.cumsum(R2, axis=1))

# -------------------------------------------------------------------
# Exécution et visualisation
# -------------------------------------------------------------------

algorithms = {
    "OMD Entropique": lambda A, T: OMD_entropic(A, T, optimistic=False),
    "O-OMD Entropique": lambda A, T: OMD_entropic(A, T, optimistic=True),
    "RM+": lambda A, T: regret_matching(A, T, plus=True, optimistic=False),
    "O-RM+": lambda A, T: regret_matching(A, T, plus=True, optimistic=True),
}

results = {}
for name, algo in tqdm(algorithms.items()):
    results[name] = algo(A, T)

# Visualisation
plt.figure(figsize=(14, 8))
colors = ['blue', 'orange', 'green', 'red']
linestyles = ['-', '--', '-.', ':']

for idx, (name, (R1, R2)) in enumerate(results.items()):
    plt.semilogy(R1, label=f"{name} - Joueur 1", 
             color=colors[idx], linestyle=linestyles[0], linewidth=2)
    plt.semilogy(R2, label=f"{name} - Joueur 2", 
             color=colors[idx], linestyle=linestyles[1], alpha=0.7)

plt.xlabel("Itérations (t)", fontsize=12)
plt.ylabel("Regret cumulé", fontsize=12)
plt.title(f"Comparaison des algorithmes standards vs optimistes\nm={m}, n={n}, T={T}", fontsize=14)
plt.legend(fontsize=10, ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()