import numpy as np
import time
import icp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# Constants
N = 10  # number of random points in the dataset
num_tests = 100  # number of test iterations
dim = 3  # number of dimensions of the points
noise_sigma = 0.01  # standard deviation error to be added
translation = 0.1  # max translation of the test set
rotation = 0.1  # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim) * translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:, 0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(
            C[:, 0:3], A, atol=6 * noise_sigma
        )  # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6 * noise_sigma)  # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6 * noise_sigma)  # R and R1 should be inverses

    print("best fit time: {:.3}".format(total_time / num_tests))

    return


def test_icp():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim) * translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(B, A, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:, 0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        assert np.mean(distances) < 6 * noise_sigma  # mean error should be small
        assert np.allclose(
            T[0:3, 0:3].T, R, atol=6 * noise_sigma
        )  # T and R should be inverses
        assert np.allclose(
            -T[0:3, 3], t, atol=6 * noise_sigma
        )  # T and t should be inverses

    print("icp time: {:.3}".format(total_time / num_tests))

    return


def test_icp_visual():

    N = 50
    dim = 3
    noise_sigma = 0.01
    translation = 0.3
    rotation = 0.5

    # Génère un nuage de points de référence A
    A = np.random.rand(N, dim)

    # Crée un nuage B transformé
    B = np.copy(A)
    t = np.random.rand(dim) * translation
    B += t

    # Applique une rotation aléatoire
    def rotation_matrix(axis, theta):
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        return np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )

    R = rotation_matrix(np.random.rand(dim), rotation)
    B = np.dot(R, B.T).T

    # Ajoute du bruit
    B += np.random.randn(N, dim) * noise_sigma

    # Exécution de l’ICP
    T, distances, iterations = icp.icp(B, A, tolerance=1e-6)

    # Transforme B pour l'aligner sur A
    B_h = np.ones((N, 4))
    B_h[:, 0:3] = B
    B_aligned = (np.dot(T, B_h.T).T)[:, 0:3]

    # ---- VISUALISATION ----
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(A[:, 0], A[:, 1], A[:, 2], c="g", label="Cible (A)")
    ax1.scatter(B[:, 0], B[:, 1], B[:, 2], c="r", label="Source (B avant ICP)")
    ax1.set_title("Avant ICP")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(A[:, 0], A[:, 1], A[:, 2], c="g", label="Cible (A)")
    ax2.scatter(
        B_aligned[:, 0], B_aligned[:, 1], B_aligned[:, 2], c="b", label="B après ICP"
    )
    ax2.set_title(f"Après ICP ({iterations} itérations)")
    ax2.legend()

    plt.show()


def visualize_with_open3d(A, B, B_aligned):

    # Création des nuages colorés
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(A)
    pcd_A.paint_uniform_color([0, 1, 0])  # vert = cible (référence)

    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(B)
    pcd_B.paint_uniform_color([1, 0, 0])  # rouge = avant ICP

    pcd_Ba = o3d.geometry.PointCloud()
    pcd_Ba.points = o3d.utility.Vector3dVector(B_aligned)
    pcd_Ba.paint_uniform_color([0, 0, 1])  # bleu = après ICP

    # Crée un repère pour mieux se repérer dans la scène
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # Fenêtre de visualisation améliorée
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ICP Bunny", width=1000, height=700)
    for obj in [pcd_A, pcd_B, pcd_Ba, coord_frame]:
        vis.add_geometry(obj)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    opt.show_coordinate_frame = True

    vis.run()
    vis.destroy_window()


def main():
    # Charger le lapin
    pcd = o3d.io.read_point_cloud(
        r"C:\Cours\IMT Atlantique\Troisième Année\Projet PROCOM\data\bunny\data\bun000.ply"
    )
    A = np.asarray(pcd.points)

    # Créer une version transformée
    R = pcd.get_rotation_matrix_from_xyz((0.4, 0.2, 0.3))
    t = np.array([0.05, 0.02, 0.03])
    B = (A @ R.T) + t

    # Appliquer ICP
    T, distances, iterations = icp.icp(B, A, tolerance=1e-6)
    print(
        f"ICP converged in {iterations} iterations with mean error {np.mean(distances):.6f}"
    )

    # Transformer B selon T
    B_h = np.ones((B.shape[0], 4))
    B_h[:, 0:3] = B
    B_aligned = (T @ B_h.T).T[:, 0:3]

    # Visualiser
    visualize_with_open3d(A, B, B_aligned)


if __name__ == "__main__":
    main()
