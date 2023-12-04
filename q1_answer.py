from q1_script import labels, images
import numpy as np
from matplotlib import pyplot as plt


print("The shape of the images is: ", images.shape)
print("The shape of the labels is: ", labels.shape)

# apply PCA manually 
# 1. compute the mean of the data
mean = np.mean(images, axis=0)
# 2. subtract the mean from the data
mean_subtracted = images - mean
# 3. compute the covariance matrix
covariance_matrix = np.cov(mean_subtracted.T)
# 4. compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# 5. sort the eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
# 6. select the first k eigenvectors
k = 20
selected_eigenvectors = sorted_eigenvectors[:, :k]
# 7. compute the new data matrix
new_data = np.dot(selected_eigenvectors.T, mean_subtracted.T).T
# 8. reconstruct the data
reconstructed_data = np.dot(selected_eigenvectors, new_data.T).T + mean

# plot the pca result
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(reconstructed_data[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()