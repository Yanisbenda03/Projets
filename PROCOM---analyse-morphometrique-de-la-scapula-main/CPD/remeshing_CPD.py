import open3d as o3d
import numpy as np
import os

# -----------------------------------------------
#  Reconstruction de maillage TY et X avec open3d
#  2 méthodes : Poisson et Ball Pivoting
# -----------------------------------------------

# --- CHARGEMENT DES DONNÉES ---
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(base_dir), "CPD", "results_CPD")

path_1 = os.path.join(results_dir, "target_cloud_X.ply")
path_2 = os.path.join(results_dir, "transformed_cloud_TY.ply")

# Charger les nuages de points
pcd_X = o3d.io.read_point_cloud(path_1)
pcd_TY = o3d.io.read_point_cloud(path_2)

# Convertir en arrays NumPy
X = np.asarray(pcd_X.points)
TY = np.asarray(pcd_TY.points)

print(f"Nuages de points chargés : X ({X.shape}), TY ({TY.shape})")


def make_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Crée un point cloud open3d à partir d'un array (N,3)
    et estime les normales.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Estimation des normales
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


# POISSON

def poisson_mesh(points: np.ndarray, depth: int = 8) -> o3d.geometry.TriangleMesh:
    """
    Reconstruction de surface par Poisson à partir d'un nuage de points.
    depth contrôle la résolution (plus grand = plus fin, plus lent).
    """
    pcd = make_pointcloud(points)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Nettoyage grossier : enlève les sommets de faible densité
    densities = np.asarray(densities)
    # seuil quantile (à ajuster si besoin)
    seuil = np.quantile(densities, 0.05)
    mask = densities < seuil
    mesh.remove_vertices_by_mask(mask)

    mesh.compute_vertex_normals()
    return mesh


# BALL PIVOTING

def ball_pivot_mesh(points: np.ndarray,
                    radii=(0.03, 0.06, 0.09)) -> o3d.geometry.TriangleMesh:
    """
    Reconstruction de surface par Ball Pivoting à partir d'un nuage de points.
    radii : rayon(s) des boules (en unités de ton nuage normalisé).
    """
    pcd = make_pointcloud(points)

    radii_vec = o3d.utility.DoubleVector(list(radii))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii_vec
    )
    mesh.compute_vertex_normals()
    return mesh



# --- CHOIX DE LA MÉTHODE DE REMAILLAGE ---
# Décommentez le bloc correspondant à la méthode que vous souhaitez utiliser.

# # OPTION A : Visualisation POISSON
print("Lancement du remaillage par la méthode de Poisson...")
mesh_TY = poisson_mesh(TY, depth=8)
# mesh_X  = poisson_mesh(X, depth=8)

mesh_TY.paint_uniform_color([1.0, 0.0, 0.0])  # rouge : source transformée
# mesh_X.paint_uniform_color([0.0, 1.0, 0.0])   # vert : cible

# o3d.visualization.draw_geometries(
#     [mesh_TY, mesh_X],
#     window_name="Reconstruction Poisson - TY (rouge) et X (vert)"
# )

o3d.visualization.draw_geometries(
    [mesh_TY],
    window_name="Reconstruction Poisson - TY (rouge) et X (vert)"
)



# OPTION B : Visualisation BALL PIVOTING
# print("Lancement du remaillage par la méthode Ball Pivoting...")
# mesh_TY = ball_pivot_mesh(TY, radii=(0.05, 0.1, 0.15))
# # mesh_X  = ball_pivot_mesh(X,  radii=(0.03, 0.06, 0.09))

# mesh_TY.paint_uniform_color([1.0, 0.0, 0.0])  # rouge : source transformée
# # mesh_X.paint_uniform_color([0.0, 1.0, 0.0])   # vert : cible

# # o3d.visualization.draw_geometries(
# #     [mesh_TY, mesh_X],
# #     window_name="Ball Pivoting - TY (rouge) et X (vert)"
# # )

# o3d.visualization.draw_geometries(
#     [mesh_TY],
#     window_name="Ball Pivoting - TY (rouge) et X (vert)"
# )