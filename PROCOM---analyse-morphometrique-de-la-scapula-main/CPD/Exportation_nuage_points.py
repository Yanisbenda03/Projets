import scipy.io as sio
import numpy as np
import trimesh

import os


base_dir = os.path.dirname(os.path.abspath(__file__))   
data_dir = os.path.join(base_dir, "data")

output_dir = os.path.join(base_dir, "results_CPD")
os.makedirs(output_dir, exist_ok=True)

path1 = os.path.join(data_dir, "Scap2.stl")
path2 = os.path.join(data_dir, "Scap23.stl")


# --------------------- FONCTION SAUVEGARDE ---------------------
def save_pointcloud(points, filepath):
    """
    Sauvegarde un nuage de points dans un fichier .ply via trimesh.
    """
    cloud = trimesh.points.PointCloud(points)
    cloud.export(filepath)
    print(f"[OK] Nuage de points enregistré : {filepath}")


# --------------------- CHARGEMENT MESH ---------------------
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
        total_points = len(mesh.vertices)
        print(f"Total unique vertices in Trimesh: {total_points}")
        
        if total_points >= n_points:
            indices = np.random.choice(total_points, n_points, replace=False)
            return mesh.vertices[indices]
        else:
            return mesh.vertices


# --------------------- VÉRIFICATION FICHIERS ---------------------
if not (os.path.exists(path1) and os.path.exists(path2)):
    raise FileNotFoundError(f"The specified files were not found: {path1}, {path2}")

# --------------------- CHARGEMENT ---------------------
Y = load_mesh(path1)  # source
X = load_mesh(path2)  # cible

# --------------------- NORMALISATION ---------------------
Y = (Y - Y.mean(0)) / np.linalg.norm(Y.std(0))
X = (X - X.mean(0)) / np.linalg.norm(X.std(0))

# --------------------- SAUVEGARDE ---------------------
save_pointcloud(Y, os.path.join(output_dir, "Y_source_normalized.ply"))
save_pointcloud(X, os.path.join(output_dir, "X_target_normalized.ply"))


print("Sauvegarde terminée.")
