import cv2
import numpy as np


def resize_image(filepath):
    # This resizes the image but maintains the aspect ratio
    image = cv2.imread(filepath)

    (h, w) = image.shape[:2]

    new_width = 800

    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))


def get_yellow_mask(image):
    # Convert colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find the yellow and create a mask
    yellow_lower = np.array([10, 100, 100])
    yellow_upper = np.array([60, 255, 255])

    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    return mask


def mask_yellow(image, mask):
    yellow_ratio = (cv2.countNonZero(mask)) / (image.size / 3)

    humanize_ratio = np.round(yellow_ratio * 100, 2)

    # Don't mask if the background is yellow. This causes a lot of issues
    if humanize_ratio < 5:
        mask_invert = cv2.bitwise_not(mask)
        image = cv2.bitwise_and(image, image, mask=mask_invert)

    return image


def filter_image(image):

    yellow_mask = get_yellow_mask(image)
    image = mask_yellow(image, yellow_mask)

    return image


def get_params():
    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 60

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.72

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    return params


def find_circles(image):
    params = get_params()

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)

    return keypoints


def circle_domino_dots(image, keypoints):
    blank = np.zeros((1, 1))
    circles = cv2.drawKeypoints(
        image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return circles


def add_text_and_circles_to_image(image, circles, text):
    cv2.putText(circles, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)


def display(circles):
    cv2.imshow("Filtering Score", circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_score(filepath):
    image = resize_image(filepath)
    image = filter_image(image)

    keypoints = find_circles(image)

    circles = circle_domino_dots(image, keypoints)
    number_of_circles = len(keypoints)

    # Uncomment these lines out to display the image(s) while testing

    # add_text_and_circles_to_image(
    #     image, circles, text=f"Domino Score: {str(number_of_circles)}"
    # )
    # display(circles)

    return number_of_circles
