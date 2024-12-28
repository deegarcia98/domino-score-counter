import cv2
import numpy as np


def resize_image(filepath):
    image = cv2.imread(filepath)

    (h, w) = image.shape[:2]

    new_width = 800

    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))


def filter_image(image):

    # image = np.array(255 * (image / 255) ** 3.2, dtype="uint8")
    # image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    # alpha = 1.5
    # beta = -200
    # image = cv2.convertScaleAbs(image, beta=beta)
    # cv2.imshow("after abs", image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Background area using Dilation
    bg = cv2.dilate(closing, kernel, iterations=1)

    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)

    return bg


def get_params():
    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    # params.minArea = 90
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


def get_score(filepath):
    image = resize_image(filepath)
    image = filter_image(image)

    params = get_params()

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(
        image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    number_of_blobs = len(keypoints)

    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return number_of_blobs
