import numpy as np
import math
import os
import trimesh
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

def gaussian_kernel(X: np.ndarray, beta: float, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Calcule la matrice de noyau gaussien G_ij = exp(-||X_i - Y_j||² / (2*beta²))."""
    if Y is None:
        Y = X
    X = np.asarray(X, dtype=np.float64, order="C")
    Y = np.asarray(Y, dtype=np.float64, order="C")
    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True).T
    d2 = XX + YY - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-d2 / (2.0 * beta * beta))


def normalize_points(P: np.ndarray) -> np.ndarray:
    """Centre et met à l’échelle un nuage de points."""
    P = P - P.mean(0, keepdims=True)
    s = np.linalg.norm(P.std(0)) + 1e-12
    return P / s


def load_mesh(filepath, num_points=5000):
    mesh = trimesh.load(filepath, process=False)
    if isinstance(mesh, trimesh.points.PointCloud):
        if mesh.vertices.shape[0] > num_points:
            idx = np.random.choice(mesh.vertices.shape[0], num_points, replace=False)
            points = mesh.vertices[idx]
        else:
            points = mesh.vertices
        return points
    else:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points

@dataclass
class EMState:
    TY: np.ndarray
    P: Optional[np.ndarray]
    P1: np.ndarray
    Pt1: np.ndarray
    PX: np.ndarray
    Np: float
    sigma2: float
    diff: float
    q: float


class EMRegistration:

    def __init__(self, X, Y, max_iterations=100, tolerance=1e-5, w=0.0):
        self.X = np.ascontiguousarray(X, np.float64)
        self.Y = np.ascontiguousarray(Y, np.float64)
        self.N, self.D = self.X.shape
        self.M = self.Y.shape[0]
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.w = float(w)

        # Variance initiale (Eq. 12 du papier)
        muX, muY = self.X.mean(0), self.Y.mean(0)
        XY = np.sum((self.X - muX)**2) * self.M + np.sum((self.Y - muY)**2) * self.N
        sigma2 = XY / (self.D * self.M * self.N)

        self.state = EMState(
            TY=self.Y.copy(),
            P=None,
            P1=np.ones(self.M),
            Pt1=np.ones(self.N),
            PX=np.zeros_like(self.Y),
            Np=float(self.M),
            sigma2=sigma2,
            diff=np.inf,
            q=np.inf
        )

    def expectation(self):
        """E-step : calcul des correspondances floues P."""
        TY = self.state.TY
        TY2 = np.sum(TY * TY, axis=1, keepdims=True)
        X2 = np.sum(self.X * self.X, axis=1, keepdims=True).T
        d2 = TY2 + X2 - 2.0 * (TY @ self.X.T)
        np.maximum(d2, 0.0, out=d2)

        c = (2.0 * math.pi * self.state.sigma2) ** (self.D / 2.0)
        den_out = (self.w / (1.0 - self.w)) * (self.M / c)
        P = np.exp(-d2 / (2.0 * self.state.sigma2))
        P /= (P.sum(0, keepdims=True) + den_out)

        self.state.P = P
        self.state.P1 = P.sum(1)
        self.state.Pt1 = P.sum(0)
        self.state.PX = P @ self.X
        self.state.Np = self.state.P1.sum()

    def update_variance(self):
        prev_sigma2 = self.state.sigma2
        xPx = (self.state.Pt1 * np.sum(self.X * self.X, axis=1)).sum()
        yPy = (self.state.P1 * np.sum(self.state.TY * self.state.TY, axis=1)).sum()
        trPXY = np.sum(self.state.TY * self.state.PX)
        sigma2 = (xPx - 2.0 * trPXY + yPy) / (self.state.Np * self.D)
        if sigma2 <= 0:
            sigma2 = self.tolerance / 10.0
        self.state.diff = abs(sigma2 - prev_sigma2)
        self.state.sigma2 = sigma2

    def update_transform(self):
        raise NotImplementedError

    def register(self):
        for _ in range(self.max_iterations):
            self.expectation()
            self.update_transform()
            self.update_variance()
            if self.state.diff < self.tolerance:
                break
        return self.state.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        return None


class DeformableRegistration(EMRegistration):
   

    def __init__(self, X, Y, alpha=2.0, beta=2.0, **kw):
        super().__init__(X, Y, **kw)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha et beta doivent être > 0")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.G = gaussian_kernel(self.Y, self.beta)
        self.W = np.zeros((self.M, self.D))

    def update_transform(self):
       
        P1, PX, Y, sigma2 = self.state.P1, self.state.PX, self.Y, self.state.sigma2
        A = (P1[:, None] * self.G) + self.alpha * sigma2 * np.eye(self.M)
        B = PX - (P1[:, None] * Y)
        self.W = np.linalg.solve(A, B)
        self.state.TY = Y + self.G @ self.W

    def transform_point_cloud(self, Y_new):
        
        return Y_new + gaussian_kernel(Y_new, self.beta, self.Y) @ self.W

    def get_registration_parameters(self):
        return self.G, self.W


if __name__ == "__main__":
    path1 = "/content/chair_0891.off" 
    path2 = "/content/chair_0892.off" 

    if not (os.path.exists(path1) and os.path.exists(path2)):
        raise FileNotFoundError(f"Fichiers non trouvés : {path1} | {path2}")

    
    Y = normalize_points(load_mesh(path1, num_points=5000))  # Source
    X = normalize_points(load_mesh(path2, num_points=5000))  # Cible
    print(f"Nuage source Y : {Y.shape}, cible X : {X.shape}")

    
    reg = DeformableRegistration(X=X, Y=Y, alpha=2.0, beta=2.0)
    TY, (G, W) = reg.register()
    print(f"Alignement terminé | sigma² = {reg.state.sigma2:.6g}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="blue", s=3, label="Cible X")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c="green", s=3, label="Source Y")
    ax.scatter(TY[:, 0], TY[:, 1], TY[:, 2], c="red", s=3, label="Y transformé")
    ax.legend()
    ax.set_title("CPD non-rigide — implémentation papier (avec échantillonnage 5000 pts)")
    plt.show()