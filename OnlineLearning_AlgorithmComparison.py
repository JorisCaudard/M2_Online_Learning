import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
m, n = 10, 10  # Tailles des stratégies
T = 5000       # Nombre d'itérations
np.random.seed(42)
A = np.random.rand(m, n) * 2 - 1  # Matrice de gain aléatoire dans [-1,1]

# -------------------------------------------------------------------
# Algorithmes implémentés
# -------------------------------------------------------------------

def OMD_entropic(A, T, gamma=0.1):
    """Online Mirror Descent avec régulariseur entropique"""
    a = np.ones(m)/m
    b = np.ones(n)/n
    R1, R2 = np.zeros(T), np.zeros(T)
    sum_grad1, sum_grad2 = 1e-8, 1e-8  # Évite la division par zéro
    
    for t in range(T):
        # Joueur 1
        grad1 = A @ b
        sum_grad1 += np.max(np.abs(grad1))**2
        gamma_t = gamma / np.sqrt(sum_grad1)
        a = a * np.exp(gamma_t * grad1)
        a /= a.sum()
        
        # Joueur 2
        grad2 = a @ A
        sum_grad2 += np.max(np.abs(grad2))**2
        gamma_t = gamma / np.sqrt(sum_grad2)
        b = b * np.exp(-gamma_t * grad2)
        b /= b.sum()
        
        # Regrets cumulés
        R1[t] = np.max(A @ b) - a @ A @ b
        R2[t] = a @ A @ b - np.min(a @ A)
    
    return np.cumsum(R1), np.cumsum(R2)

def OMD_euclidean(A, T, gamma=0.1):
    """Online Mirror Descent avec régulariseur Euclidien (projection L2)"""
    a = np.ones(m)/m
    b = np.ones(n)/n
    R1, R2 = np.zeros(T), np.zeros(T)
    
    for t in range(T):
        # Joueur 1
        grad1 = A @ b
        a = a + gamma * grad1
        a = np.clip(a, 0, None)  # Projection sur le simplexe
        a /= a.sum()
        
        # Joueur 2
        grad2 = a @ A
        b = b - gamma * grad2
        b = np.clip(b, 0, None)
        b /= b.sum()
        
        R1[t] = np.max(A @ b) - a @ A @ b
        R2[t] = a @ A @ b - np.min(a @ A)
    
    return np.cumsum(R1), np.cumsum(R2)

def regret_matching(A, T, plus=False):
    """Regret Matching ou RM+"""
    R1, R2 = np.zeros((m, T)), np.zeros((n, T))
    a, b = np.ones(m)/m, np.ones(n)/n
    
    for t in range(1, T):
        # Joueur 1
        r1 = A @ b - a @ A @ b
        R1[:, t] = R1[:, t-1] + r1
        if plus:
            R1[:, t] = np.maximum(R1[:, t], 0)
        a = R1[:, t] / np.sum(R1[:, t]) if np.sum(R1[:, t]) > 0 else np.ones(m)/m
        
        # Joueur 2
        r2 = -(a @ A - a @ A @ b)
        R2[:, t] = R2[:, t-1] + r2
        if plus:
            R2[:, t] = np.maximum(R2[:, t], 0)
        b = R2[:, t] / np.sum(R2[:, t]) if np.sum(R2[:, t]) > 0 else np.ones(n)/n
    
    return np.max(np.cumsum(R1, axis=1), axis=0), np.max(np.cumsum(R2, axis=1), axis=0)

def exponential_weights(A, T, gamma=0.01):
    """Poids exponentiels classiques (pas fixe)"""
    a = np.ones(m)/m
    b = np.ones(n)/n
    R1, R2 = np.zeros(T), np.zeros(T)
    
    for t in range(T):
        # Joueur 1
        grad1 = A @ b
        a = a * np.exp(gamma * grad1)
        a /= a.sum()
        
        # Joueur 2
        grad2 = a @ A
        b = b * np.exp(-gamma * grad2)
        b /= b.sum()
        
        R1[t] = np.max(A @ b) - a @ A @ b
        R2[t] = a @ A @ b - np.min(a @ A)
    
    return np.cumsum(R1), np.cumsum(R2)

# -------------------------------------------------------------------
# Exécution et visualisation
# -------------------------------------------------------------------

algos = {
    "OMD Entropique": OMD_entropic,
    "OMD Euclidien": OMD_euclidean,
    "RM": lambda A, T: regret_matching(A, T, plus=False),
    "RM+": lambda A, T: regret_matching(A, T, plus=True),
    "Exponential Weights": exponential_weights
}

results = {}
for name, algo in tqdm(algos.items()):
    results[name] = algo(A, T)

# Visualisation
plt.figure(figsize=(14, 7))
for name in algos.keys():
    R1, R2 = results[name]
    plt.semilogy(R1, label=f"{name} - Joueur 1", ls='-')
    plt.semilogy(R2, label=f"{name} - Joueur 2", ls='--')

plt.xlabel("Itérations (t)", fontsize=12)
plt.ylabel("Regret cumulé", fontsize=12)
plt.title("Comparaison des algorithmes sur un jeu à somme nulle\n" + 
          f"Matrice {m}x{n}, T={T} itérations", fontsize=14)
plt.legend(fontsize=10, ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
