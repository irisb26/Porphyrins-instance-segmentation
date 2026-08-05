import napari
import numpy as np
import tifffile
import sys #system 

#save image/labels from jupyter notebook in temporary files, read them here
img_path = sys.argv[1]
labels_path = sys.argv[2]
output_path = sys.argv[3]

# Load existing data
image_data = tifffile.imread(img_path)
initial_labels = tifffile.imread(labels_path)

# Build napari viewer
viewer = napari.Viewer()
viewer.add_image(image_data, name="Raw Image")
labels_layer = viewer.add_labels(initial_labels, name="StarDist labels")

# Block execution until GUI window is closed
napari.run()

# Save final edited labels upon closing
tifffile.imwrite(output_path, labels_layer.data)
print(f"Saved modified labels to {output_path}")
