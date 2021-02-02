from utils.images_stack import *
from utils.color_filter import *

MAX_FEATURES = 2000
GOOD_MATCH_PERCENT = 0.8


def align_images(img_new, img_reference):
    # Detect ORB features and  descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    key_points1, descriptors1 = orb.detectAndCompute(img_new, None)
    key_points2, descriptors2 = orb.detectAndCompute(img_reference, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = key_points1[match.queryIdx].pt
        points2[i, :] = key_points2[match.trainIdx].pt

    # Find homography
    homography, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channel = img_reference.shape
    img_aligned = cv.warpPerspective(img_new, homography, (width, height))

    return img_aligned


def colony_health(ref_whole_colored, ref_whole_mask, ref_red_mask, ref_white_mask):

    hsv_trackbars_create("Red Filter")
    hsv_trackbars_create("White Filter")
    while True:
        # ________________________________________________________________ Capture Frame
        # _, frame = capture.read()
        frame = cv.imread("images/b-water-1-dark.jpg")

        # ________________________________________________________________ Color Filtering
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Bounds
        red_lower_bound, red_upper_bound = hsv_trackbars_pos(name="Red Filter")[0]
        white_lower_bound, white_upper_bound = hsv_trackbars_pos(name="White Filter")[0]

        # Masks
        frame_red_mask = cv.inRange(frame_hsv, lowerb=red_lower_bound, upperb=red_upper_bound)
        frame_white_mask = cv.inRange(frame_hsv, lowerb=white_lower_bound, upperb=white_upper_bound)
        frame_whole_mask = frame_red_mask + frame_white_mask

        # Frame Masked
        frame_red_masked = cv.bitwise_and(frame, frame, mask=frame_red_mask)
        frame_white_masked = cv.bitwise_and(frame, frame, mask=frame_white_mask)
        frame_whole_masked = cv.bitwise_and(frame, frame, mask=frame_whole_mask)

        # ________________________________________________________________ Color Filtering Stack
        color_filters_stack = stack_images(0.4, [
            [frame, frame_red_masked, frame_red_mask],
            [frame, frame_white_masked, frame_white_mask],
            [frame, frame_whole_masked, frame_whole_mask]
        ])
        cv.imshow("Color Filters Stack", color_filters_stack)

        # ________________________________________________________________ Color Filtering Complete
        if hsv_trackbars_pos(name="Red Filter")[1] == 1 and hsv_trackbars_pos(name="White Filter")[1] == 1:

            # ________________________________________________________________ Images Alignment
            # Alignment is performed on the "before" and "after" images without the background
            frame_aligned = align_images(frame_whole_masked, ref_whole_colored)
            alignment_stack = stack_images(0.5, [[ref_whole_colored, frame_aligned]])
            cv.imshow("Alignment Stack", alignment_stack)

            # ________________________________________________________________ Re-compute the Aligned Masks
            frame_aligned_hsv = cv.cvtColor(frame_aligned, cv.COLOR_BGR2HSV)
            frame_red_mask = cv.inRange(frame_aligned_hsv, lowerb=red_lower_bound, upperb=red_upper_bound)
            frame_white_mask = cv.inRange(frame_aligned_hsv, lowerb=white_lower_bound, upperb=white_upper_bound)
            frame_whole_mask = frame_red_mask + frame_white_mask

            kernel = np.ones((9, 9), np.uint8)
            # ________________________________________________________________ Recovery
            # Dilate frame's white mask (For Enhanced Difference)
            frame_white_mask = cv.dilate(frame_white_mask, kernel, iterations=1)

            recovered_areas = ref_white_mask - frame_white_mask
            recovered_areas = np.where(recovered_areas > 0, recovered_areas, 0)

            # Opening Morphology to Reduce Noise
            recovered_areas = cv.morphologyEx(recovered_areas, cv.MORPH_OPEN, kernel)
            _, recovered_areas = cv.threshold(recovered_areas, 100, 255, cv.THRESH_BINARY)

            recovery_stack = [ref_white_mask, frame_white_mask, recovered_areas]

            contours, _ = cv.findContours(recovered_areas, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv.contourArea(contour) > 600:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(frame_aligned, (x - 5, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            # ________________________________________________________________ Bleach
            # Dilate frame's red mask (For Enhanced Difference)
            frame_red_mask = cv.dilate(frame_red_mask, kernel, iterations=1)

            bleached_areas = ref_red_mask - frame_red_mask
            bleached_areas = np.where(bleached_areas > 0, bleached_areas[:], 0)

            # Opening Morphology to Reduce Noise
            bleached_areas = cv.morphologyEx(bleached_areas, cv.MORPH_OPEN, kernel)
            _, bleached_areas = cv.threshold(bleached_areas, 100, 255, cv.THRESH_BINARY)

            bleach_stack = [ref_red_mask, frame_red_mask, bleached_areas]

            contours, _ = cv.findContours(bleached_areas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv.contourArea(contour) > 600:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(frame_aligned, (x - 5, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 2)

            # ________________________________________________________________ Damage
            frame_whole_mask = cv.dilate(frame_whole_mask, kernel, iterations=1)

            damaged_areas = ref_whole_mask - frame_whole_mask
            damaged_areas = np.where(damaged_areas <= 0, 0, damaged_areas)

            # Opening Morphology to Reduce Noise
            damaged_areas = cv.morphologyEx(damaged_areas, cv.MORPH_OPEN, kernel)
            _, damaged_areas = cv.threshold(damaged_areas, 100, 255, cv.THRESH_BINARY)

            damage_stack = [ref_whole_mask, frame_whole_mask, damaged_areas]

            contours, _ = cv.findContours(damaged_areas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv.contourArea(contour) > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(frame_aligned, (x - 5, y - 10), (x + w + 10, y + h + 10), (0, 255, 255), 2)

            # ________________________________________________________________ Growth
            frame_whole_mask = cv.erode(frame_whole_mask, kernel, iterations=2)
            growth_areas = frame_whole_mask - ref_whole_mask
            growth_areas = np.where(growth_areas <= 0, 0, growth_areas)

            # Opening Morphology to Reduce Noise
            growth_areas = cv.morphologyEx(growth_areas, cv.MORPH_OPEN, kernel)
            _, growth_areas = cv.threshold(growth_areas, 100, 255, cv.THRESH_BINARY)

            growth_stack = [frame_whole_mask, ref_whole_mask, growth_areas]

            contours, _ = cv.findContours(growth_areas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv.contourArea(contour) > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(frame_aligned, (x - 5, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

            operations_stack = stack_images(0.4, [recovery_stack, bleach_stack, damage_stack, growth_stack])
            cv.imshow("Operations", operations_stack)
            cv.imshow("Comparison", frame_aligned)
        cv.waitKey(1)


if __name__ == '__main__':
    # IP Webcam
    ip = "192.168.1.4"
    port = "8080"
    # capture = cv.VideoCapture(f"https://{ip}:{port}/video")

    a_nobg = cv.imread("images/a-nobg.png")
    a_red = cv.imread("images/a-red.jpg")
    a_white = cv.imread("images/a-white.jpg")
    a_whole = a_red + a_white
    a_red = cv.cvtColor(a_red, cv.COLOR_BGR2GRAY)
    a_white = cv.cvtColor(a_white, cv.COLOR_BGR2GRAY)
    a_whole = cv.cvtColor(a_whole, cv.COLOR_BGR2GRAY)

    colony_health(a_nobg, a_whole, a_red, a_white)
