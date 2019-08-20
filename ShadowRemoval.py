from skimage import measure
from cv2 import cv2 as cv
import numpy as np
import sys

'''

This software is implemented according to the methods presented in:

- Murali, Saritha, and V. K. Govindan. 
"Removal of shadows from a single image." 
the Proceedings of First International Conference on Futuristic Trends 
in Computer Science and Engineering. Vol. 4.

- Murali, Saritha, and V. K. Govindan. 
"Shadow detection and removal from a single image using LAB color space." 
Cybernetics and information technologies 13.1 (2013): 95-103.

'''

__all__ = ['ShadowRemover']

class ShadowRemover:

    # Applies median filtering over given point
    @staticmethod
    def manuel_median_filter(img_o_i, point, filter_size):
        # Obtain window indices
        indices = [[x, y] for x in range(point[1] - filter_size // 2, point[1] + filter_size // 2 + 1)
                   for y in range(point[0] - filter_size // 2, point[0] + filter_size // 2 + 1)]

        indices = list(filter(lambda x: not (x[0] < 0 or x[1] < 0 or
                                             x[0] >= img_o_i.shape[0] or
                                             x[1] >= img_o_i.shape[1]), indices))

        pixel_values = [0, 0, 0]
        # Find the median of pixel values
        for channel in range(3):
            pixel_values[channel] = list(img_o_i[index[0], index[1], channel] for index in indices)
        pixel_values = list(np.median(pixel_values, axis=1))

        return pixel_values


    # Applies median filtering on given contour pixels, the filter size is adjustable
    @staticmethod
    def edge_median_filter(shadow_clear_img__h_s_v, contours_list, filter_size=21):
        temp_img = np.copy(shadow_clear_img__h_s_v)

        for partition in contours_list:
            for point in partition:
                temp_img[point[0][1]][point[0][0]] = ShadowRemover.manuel_median_filter(shadow_clear_img__h_s_v, point[0], filter_size)

        return cv.cvtColor(temp_img, cv.COLOR_HSV2BGR)

    @staticmethod
    def removeShadows(imgName,
                      saveName,
                      region_adjustment_kernel_size=10,
                      shadow_dilation_iteration=25,
                      shadow_dilation_kernel_size=17,
                      blob_threshold = 1000):

        orgImage = cv.imread(imgName)
        # If the image is in BGRA color space, convert it to BGR
        if orgImage.shape[2] == 4:
            orgImage = cv.cvtColor(orgImage, cv.COLOR_BGRA2BGR)
        convertedImg = cv.cvtColor(orgImage, cv.COLOR_BGR2LAB)
        shadowClearImg = np.copy(orgImage)  # Used for constructing corrected image

        # Calculate the mean values of A and B across all pixels
        means = [np.mean(convertedImg[:, :, i]) for i in range(3)]
        thresholds = [means[i] - (np.std(convertedImg[:, :, i]) / 3) for i in range(3)]

        # If mean is below 256 (which is I think the max value for a channel)
        # Apply threshold using only L

        mask = cv.inRange(convertedImg, (0, 0, 0), (thresholds[0], 256, thresholds[2]))

        kernel_size = (region_adjustment_kernel_size, region_adjustment_kernel_size)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
        cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, mask)
        cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, mask)

        # We need connected components
        # Initialize the labels of the blobs in our binary image
        labels = measure.label(mask)

        # Now, we will iterate over each label's pixels
        for label in np.unique(labels):
            if not label == 0:
                temp_filter = np.zeros(mask.shape, dtype="uint8")
                temp_filter[labels == label] = 255

                # Only consider blobs with size above threshold
                if cv.countNonZero(temp_filter) >= blob_threshold:
                    shadow_indices = np.where(temp_filter == 255)

                    # Calculate average LAB values in current shadow region
                    shadow_average_LAB = np.mean(convertedImg[shadow_indices[0], shadow_indices[1], :], axis=0)

                    # TODO: Apply dilation few times, in order to obtain non-shadow pixels around shadow region
                    # Play with the parameters for optimization
                    non_shadow_kernel_size = (shadow_dilation_kernel_size, shadow_dilation_kernel_size)
                    non_shadow_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, non_shadow_kernel_size)
                    non_shadow_temp_filter = cv.dilate(temp_filter, non_shadow_kernel, iterations=shadow_dilation_iteration)

                    # Get the new set of indices and remove shadow indices from them
                    non_shadow_temp_filter = cv.bitwise_xor(non_shadow_temp_filter, temp_filter)
                    non_shadow_indices = np.where(non_shadow_temp_filter == 255)

                    temp = cv.findContours(temp_filter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                    # Contours are used for extracting the edges of the current shadow region
                    contours, hierarchy = cv.findContours(temp_filter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                    # Q: Rather than asking for RGB constants individually, why not adjust L only?
                    # A: L component isn't enough to REVIVE the colors that were under the shadow.

                    # Find the average of the non-shadow areas
                    
                    # Get the average LAB from border areas
                    border_average_LAB = np.mean(convertedImg[non_shadow_indices[0], non_shadow_indices[1], :], axis=0)
                    # Calculate ratios that are going to be used on clearing the current shadow region
                    # This is different for each region, therefore calculated each time
                    lab_ratio = border_average_LAB / shadow_average_LAB

                    # Adjust LAB THIS DOESN'T REVIVE THE COLOR INFO, ONLY ADDS ILLUMINANCE

                    shadowClearImg = cv.cvtColor(shadowClearImg, cv.COLOR_BGR2LAB)
                    shadowClearImg[shadow_indices[0], shadow_indices[1]] = np.uint8(
                        shadowClearImg[shadow_indices[0],
                                        shadow_indices[1]] * lab_ratio)
                    shadowClearImg = cv.cvtColor(shadowClearImg, cv.COLOR_LAB2BGR)

                    shadowClearImg = cv.blur(shadowClearImg, (5, 5))

                    

        
        cv.imwrite(saveName, shadowClearImg)

