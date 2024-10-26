import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('thiennhien.jpg', cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))  
sobel_y = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))  
sobel_combined = cv2.magnitude(sobel_x, sobel_y) 

log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])

log_image = cv2.filter2D(image, -1, log_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh xám')
plt.subplot(1, 3, 2), plt.imshow(sobel_combined, cmap='gray'), plt.title('Biên với Sobel')
plt.subplot(1, 3, 3), plt.imshow(log_image, cmap='gray'), plt.title('Biên với LoG')
plt.tight_layout()
plt.show()
