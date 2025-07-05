import cv2
import numpy as np

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.5):
    """Apply Gamma Correction."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def white_balance(image):
    """White Balance using Gray World Assumption."""
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

# Load the low-light image
image_path = "image1.jpg"  # Change to your output path
image = cv2.imread(image_path)

# Apply Enhancements
enhanced_image = apply_clahe(image)          # Step 1: Improve Contrast
enhanced_image = gamma_correction(enhanced_image, gamma=1.8)  # Step 2: Adjust Brightness
enhanced_image = white_balance(enhanced_image)  # Step 3: Fix Colors

# Save the final output
cv2.imwrite("output_enhanced.png", enhanced_image)

print("✅ Enhancement Completed! Check 'output_enhanced.png'")