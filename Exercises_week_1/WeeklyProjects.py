'''
# Weekly project part 1
Using the image "appletree.jpg"
1) Can you segment the apples from the tree?

2) Can you get the computer to count how many there are?  
How close can you get to the ground truth? (there are 26 apples in the image)

3) Can you change the color of one of them?

4) Can you segment the leaves?
'''

# 1)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils 

path = "/Users/madsyar/Desktop/1. semester/Perception for autonomous systems/Exercises/Exercises_week_1/appletree.jpg"

bgr_img = cv2.imread(path)
b,g,r = cv2.split(bgr_img)       # get b,g,r
image = cv2.merge([r,g,b])
# plt.imshow(image)
# plt.show()

# 1)
def calculate_saturation(image):
    max_val = np.max(image, axis=2)
    min_val = np.min(image, axis=2)
    saturation = max_val - min_val
    return saturation

def isolate_red_apples(image, saturation_threshold=50, red_threshold=100):
    saturation = calculate_saturation(image)
    
    red_mask = (image[:, :, 0] > red_threshold) & (image[:, :, 1] < 100) & (image[:, :, 2] < 100)
    saturation_mask = saturation > saturation_threshold
    
    combined_mask = red_mask & saturation_mask
    combined_mask = combined_mask.astype(np.uint8) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask

# 2)
def count_apples(image):
    mask = isolate_red_apples(image)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 300  # Adjust this value based on your image
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    output = image.copy()
    cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)  # Draw contours in green
    
    num_apples = len(filtered_contours)
    
    plt.imshow(output)  # Convert BGR to RGB for display
    plt.title(f'Number of Apples: {num_apples}')
    plt.show()
    
    return num_apples, filtered_contours

# 3)
# Function to color one apple
def color_apple(image, contour, color=(0, 255, 0)):
    # Create a mask for the selected apple
    apple_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    
    # Draw the selected apple contour on the mask
    cv2.drawContours(apple_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Create an image to overlay the color
    colored_overlay = np.zeros_like(image)
    
    # Apply the color to the regions specified by the mask
    for i in range(3):  # Iterate over the 3 color channels
        colored_overlay[:, :, i] = np.where(apple_mask == 255, color[i], colored_overlay[:, :, i])
    
    # Blend the colored overlay with the original image
    colored_image = cv2.addWeighted(image, 1.0, colored_overlay, 0.5, 0)
    
    return colored_image

# Run the functions
num_apples, filtered_contours = count_apples(image)
print(f'Number of apples detected: {num_apples}')

if num_apples > 0:
    # Color the first apple (index 0) if there are any apples detected
    if len(filtered_contours) > 2:  # Ensure there are enough apples
        colored_image = color_apple(image, filtered_contours[7], color=(0, 255, 255))  # Yellow color
        plt.imshow(colored_image)  
        plt.title('Colored Apple')
        plt.show()
    else:
        print("Not enough apples detected to color.")
else:
    print("No apples detected to color.")

# 4)
# Function to isolate leaves
def isolate_leaves(image, green_threshold=100, saturation_threshold=50):
    # Calculate saturation
    max_val = np.max(image, axis=2)
    min_val = np.min(image, axis=2)
    saturation = max_val - min_val

    # Define thresholds for green color and saturation
    green_mask = (image[:, :, 1] > green_threshold) & (image[:, :, 0] < 100) & (image[:, :, 2] < 100)
    saturation_mask = saturation > saturation_threshold

    # Combine the masks
    combined_mask = green_mask & saturation_mask
    combined_mask = combined_mask.astype(np.uint8) * 255

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask

# Function to display the segmented leaves
def show_segmented_leaves(image):
    mask = isolate_leaves(image)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Draw contours in green

    # Display the result
    plt.imshow(output)  # Convert BGR to RGB for display
    plt.title('Segmented Leaves')
    plt.show()

# Show the segmented leaves
# show_segmented_leaves(image)


'''
# Weekly project part 2
1) Remove the greenscreen and replace the background in "itssp.png".
2) Can you improve the edge with eroding/dilating?
'''

# 1)
path2 = "ittsp.png"

bgr_img = cv2.imread(path2)
b,g,r = cv2.split(bgr_img)       # get b,g,r
image = cv2.merge([r,g,b])

plt.imshow(image)
plt.show()

# 1)

# Create a mask where the green screen is (green channel should dominate)
green_mask = (g > 150) & (r < 150) & (b < 150)  # Adjust thresholds if necessary

# Convert mask to uint8 format
green_mask = green_mask.astype(np.uint8) * 255

# Invert the mask to get the foreground (person + clock)
foreground_mask = cv2.bitwise_not(green_mask)

# Apply the mask to the original image
foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

# Replace the green background with a new color (e.g., white) or a background image
new_background = np.ones_like(image) * np.array((255, 100, 000), dtype=np.uint8) 
new_background = cv2.bitwise_and(new_background, new_background, mask=green_mask)

# Combine the foreground with the new background
final_image = cv2.add(foreground, new_background)

# Display the result
plt.imshow(final_image)
plt.title('Green Screen Removed')
plt.show()

# 2)

'''
Yes, you can improve the edges of your mask by using erosion and dilation. 
These operations can smooth out the boundaries, remove small imperfections, and refine the mask to make the 
transition between the foreground (person and clock) and background more natural
'''
# Create a mask where the green screen is (green channel should dominate)
green_mask = (g > 150) & (r < 150) & (b < 150)  # Adjust thresholds if necessary
green_mask = green_mask.astype(np.uint8) * 255

# Apply morphological operations to smooth edges
kernel = np.ones((5, 5), np.uint8)  # Kernel size (tune it based on your image size)

# Step 1: Erosion (shrink small artifacts)
green_mask = cv2.erode(green_mask, kernel, iterations=1)

# Step 2: Dilation (expand back after erosion)
green_mask = cv2.dilate(green_mask, kernel, iterations=2)

# Invert the mask to get the foreground (person + clock)
foreground_mask = cv2.bitwise_not(green_mask)

# Apply the mask to the original image to extract the foreground
foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

# Replace the green background with a new color (e.g., white) or a background image
new_background = np.ones_like(image) * np.array([255, 100, 000], dtype=np.uint8)  # Example color background
new_background = cv2.bitwise_and(new_background, new_background, mask=green_mask)

# Combine the foreground with the new background
final_image = cv2.add(foreground, new_background)

# Display the result
plt.imshow(final_image)
plt.title('Green Screen Removed with Improved Edges')
plt.show()
