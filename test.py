import numpy as np
from PIL import Image

# Load the masked image
masked_image = Image.open("path_to_masked_image.png")

# Convert the image to a NumPy array
masked_array = np.array(masked_image)

# Find non-black pixels (assuming mask is pure black)
non_black_pixels = np.any(masked_array != 0, axis=-1)

# Get bounding box of non-black pixels
non_black_coordinates = np.argwhere(non_black_pixels)
(top, left), (bottom, right) = (
    non_black_coordinates.min(0),
    non_black_coordinates.max(0) + 1,
)

# Crop the image using the bounding box
cropped_image = masked_image.crop((left, top, right, bottom))

# Save or display the cropped image
cropped_image.show()


# TODO AFTER THAT POSSIBLY BREAK DOWN INTO SMALLER PATCH TO CATCH LOCAL.
