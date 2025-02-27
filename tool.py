import cv2
import numpy as np

def create_visualization(gray, blurred, blurred2, edges, result_img):
    """Create a visualization of image processing steps.
    
    Args:
        gray: Grayscale image
        blurred: First blurred image
        blurred2: Second blurred image
        edges: Edge detection result
        result_img: Final result image with markings
    """
    # Scale images for better visualization
    scale_percent = 50  # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # Resize all images
    gray_resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
    blurred_resized = cv2.resize(blurred, dim, interpolation=cv2.INTER_AREA)
    blurred2_resized = cv2.resize(blurred2, dim, interpolation=cv2.INTER_AREA)
    edges_resized = cv2.resize(edges, dim, interpolation=cv2.INTER_AREA)
    result_resized = cv2.resize(result_img, dim, interpolation=cv2.INTER_AREA)
    
    # Convert edges to 3-channel for proper display
    edges_colored = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    gray_colored = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    blurred_colored = cv2.cvtColor(blurred_resized, cv2.COLOR_GRAY2BGR)
    blurred2_colored = cv2.cvtColor(blurred2_resized, cv2.COLOR_GRAY2BGR)
    
    # Create horizontal stack of images
    vis = np.hstack([gray_colored, blurred_colored, blurred2_colored, edges_colored, result_resized])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    cv2.putText(vis, 'Gray', (width//2 - 30, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Blurred', (width + width//2 - 50, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Blurred2', (2*width + width//2 - 50, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Edges', (3*width + width//2 - 40, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Result', (4*width + width//2 - 40, y_pos), font, 1, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Image Processing Steps", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()