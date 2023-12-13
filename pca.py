from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np


class PcaHelper:
    data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.data = data

    def __get_cov_matrix(self) -> np.ndarray:
        return np.cov(self.data, rowvar=False)

    @staticmethod
    def __sort_eigens_desc(eigen_values: np.ndarray, eigen_vectors: np.ndarray):
        sorted_eigen_values = eigen_values[::-1]
        sorted_eigen_vectors = eigen_vectors[:, ::-1]
        return sorted_eigen_values, sorted_eigen_vectors

    @staticmethod
    def __evaluate_PVEs(eigen_values: np.ndarray) -> np.ndarray:
        variance = np.sum(eigen_values)
        return eigen_values / variance

    @staticmethod
    def __min_max_scale(matrix: np.ndarray) -> np.ndarray:
        max_value = np.max(matrix)
        min_value = np.min(matrix)
        min_max_diff = max_value - min_value

        scaled_matrix = (matrix - min_value) / min_max_diff
        return scaled_matrix

    def get_sorted_eigens(self):
        cov_matrix = self.__get_cov_matrix()

        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        sorted_eigen_values, sorted_eigen_vectors = self.__sort_eigens_desc(
            eigen_values, eigen_vectors
        )

        return sorted_eigen_values, sorted_eigen_vectors

    def get_PVEs(self, k: int = None) -> np.ndarray:
        cov_matrix = self.__get_cov_matrix()

        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        sorted_eigen_values, sorted_eigen_vectors = self.__sort_eigens_desc(
            eigen_values, eigen_vectors
        )

        pves = self.__evaluate_PVEs(sorted_eigen_values)
        if k is None:
            return pves

        return pves[:k]

    def get_cumulative_PVEs(self) -> np.ndarray:
        pves = self.get_PVEs()
        return np.cumsum(pves)

    def get_k_PCs(self, k: int) -> np.ndarray:
        cov_matrix = self.__get_cov_matrix()

        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        _, sorted_eigen_vectors = self.__sort_eigens_desc(eigen_values, eigen_vectors)

        return sorted_eigen_vectors[:, :k]

    def plot_k_PCs(self, k: int):
        k_PCs = self.get_k_PCs(k)
        _, axs = plt.subplots(2, 5, figsize=(10, 5))

        for i, ax in enumerate(axs.flat):
            reshaped_component = k_PCs[:, i].reshape(28, 28)
            mm_scaled_component = self.__min_max_scale(reshaped_component)

            ax.axis("off")
            ax.imshow(mm_scaled_component, cmap="gray")
            ax.set_title(f"Component: {i + 1}")
        plt.tight_layout()
        plt.show()

    def project_images_on_2PCs(self, images: np.ndarray, labels: np.ndarray):
        _, eigen_vectors = self.get_sorted_eigens()
        pcs2 = eigen_vectors[:, :2]

        required_images = images[:100]
        required_labels = labels[:100]
        projected_images = np.dot(required_images, pcs2)
        color_labels = plt.cm.tab10(required_labels)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            projected_images[:, 0],
            projected_images[:, 1],
            c=color_labels[required_labels],
            cmap="viridis",
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Projection of images on the first 2 PCs")
        plt.legend(
            handles=[
                Patch(color=color_labels[i], label=f"Digit {i}") for i in range(10)
            ],
            title="Digits",
        )
        plt.show()

    def reconstruct_image(self, image: np.ndarray, k: int, mean):
        _, eigenvectors = self.get_sorted_eigens()
        data = np.dot(image - mean, eigenvectors[:, :k])
        reconstructed_image = np.dot(data, eigenvectors[:, :k].T) + mean
        return reconstructed_image

    def plot_multiple_reconstructions(self, images: np.ndarray, Ks: list[int]):
        image = images[0]
        mean = np.mean(images, axis=0)
        
        plt.figure(figsize=(10, 7))
        for index, k in enumerate(Ks, 1):
            reconstructed_image = self.reconstruct_image(image, k, mean)
            reshaped_image = reconstructed_image.reshape(28, 28)
            plt.subplot(2, 3, index)
            plt.imshow(reshaped_image, cmap="gray")
            plt.title(f"Reconstructed k={k}")
        plt.show()
