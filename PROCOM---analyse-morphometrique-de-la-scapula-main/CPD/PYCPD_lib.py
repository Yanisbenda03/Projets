import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pycpd import DeformableRegistration
import open3d as o3d

import os
import trimesh

base_dir = os.path.dirname(os.path.abspath(__file__))   
data_dir = os.path.join(base_dir, "data")


output_dir = os.path.join(base_dir, "results_CPD")
os.makedirs(output_dir, exist_ok=True)

path1 = os.path.join(data_dir, "Scap2.stl")
path2 = os.path.join(data_dir, "Scap23.stl")

#Chargement des fichiers source/cible


def load_mesh(filepath, n_points=6000):
    mesh = trimesh.load(filepath, process=False)
    
    # Nuage de points
    if isinstance(mesh, trimesh.points.PointCloud):
        total_points = len(mesh.vertices)
        print(f"Total points in PointCloud: {total_points}")
        if total_points <= n_points:
            return mesh.vertices
        else:
            indices = np.random.choice(total_points, n_points, replace=False)
            return mesh.vertices[indices]

    # Maillage
    elif isinstance(mesh, trimesh.Trimesh):
        # Obtenir les sommets uniques
        total_points = len(mesh.vertices)
        print(f"Total unique vertices in Trimesh: {total_points}")
        
        if total_points >= n_points:
            # Échantillonnage aléatoire des sommets uniques
            indices = np.random.choice(total_points, n_points, replace=False)
            return mesh.vertices[indices]
        else:
            # points, _ = trimesh.sample.sample_surface(mesh, n_points)
            # return points
            return mesh.vertices


if not (os.path.exists(path1) and os.path.exists(path2)):
    raise FileNotFoundError(
        f"The specified files were not found: {path1}, {path2}"
    )

Y = load_mesh(path1)  # source
X = load_mesh(path2)  # cible

# normalisation 
Y = (Y - Y.mean(0)) / np.linalg.norm(Y.std(0))
X = (X - X.mean(0)) / np.linalg.norm(X.std(0))

print(f"Nuage source Y : {Y.shape}, cible X : {X.shape}")

# Application de la transformation

X = np.ascontiguousarray(X.astype(np.float64))
Y = np.ascontiguousarray(Y.astype(np.float64))

reg = DeformableRegistration(X=X, Y=Y, alpha=15.0, beta=2.0)
TY, (G, W) = reg.register()

print("Alignement terminé | sigma^2 =", reg.sigma2)

#Visualisation (nuages de points)

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], X[:,2], c='blue', s=3, label='Cible X')
# ax.scatter(Y[:,0], Y[:,1], Y[:,2], c='green', s=3, label='Source Y')
# ax.scatter(TY[:,0], TY[:,1], TY[:,2], c='red', s=3, label='Y transformé')
# ax.legend()
# ax.set_title("CPD non-rigide ")
# plt.show()


# Sauvegarde des nuages de points 


def save_point_cloud_to_ply(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

target_ply_path = os.path.join(output_dir, "target_cloud_X.ply")
transformed_ply_path = os.path.join(output_dir, "transformed_cloud_TY.ply")

save_point_cloud_to_ply(X, target_ply_path)
save_point_cloud_to_ply(TY, transformed_ply_path)

print(f"Nuages de points sauvegardés dans le dossier '{output_dir}'")
