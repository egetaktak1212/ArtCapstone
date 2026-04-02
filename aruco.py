import cv2
import numpy as np
import matplotlib.pyplot as plt

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 200

fig, axes = plt.subplots(2, 3, figsize=(8, 6))

for i, marker_id in enumerate(range(1, 7)):
    marker_image = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        marker_size
    )
    
    cv2.imwrite(f"marker_{marker_id}.png", marker_image)

    ax = axes[i // 3, i % 3]
    ax.imshow(marker_image, cmap='gray', interpolation='nearest')
    ax.set_title(f'ID {marker_id}')
    ax.axis('off')

plt.tight_layout()
plt.show()
