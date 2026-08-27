import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. WS10 CAMERA CALIBRATION
# ============================================================

K = np.array([
    [534.075, 0.0,   641.53],
    [0.0,    534.305, 355.714],
    [0.0,    0.0,     1.0]
], dtype=np.float64)


R = np.array([
    [-0.999708678846212,  0.0180780476635436, -0.015991923961726],
    [-0.0224052471724579, -0.448689103870014,  0.893407014169614],
    [ 0.00897565250521704, -0.893505048816737, -0.448963434368879]
], dtype=np.float64)


T = np.array([
    [3.460003],
    [-0.8772764],
    [5.762192]
], dtype=np.float64)


# ============================================================
# 2. GLOBAL CAMERA POSITION
# ============================================================

CAMERA_POSITION = np.array([
    3.43,
    5.44,
    1.82
], dtype=np.float64)


# ============================================================
# 3. CONFIDENCE THRESHOLDS
# ============================================================

PERSON_CONFIDENCE = 0.50
ANKLE_CONFIDENCE = 0.50


# ============================================================
# 4. WORKER SELECTION PARAMETERS
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
#
# Worker classification is now based primarily on
# PERSON BBOX OVERLAP with the moving car.
#
# Ankles are NOT mandatory anymore.
# ------------------------------------------------------------

# Minimum percentage of PERSON bounding box that must
# overlap the car polygon.
#
# Example:
#
# 0%   -> person completely outside car
# 10%  -> small overlap
# 50%  -> half of person bbox inside car
# 100% -> person bbox completely inside car
#
MIN_CAR_OVERLAP = 1.0


# ============================================================
# 5. LOAD YOLO POSE MODEL
# ============================================================

model = YOLO("yolo11n-pose.pt")


# ============================================================
# 6. VIDEO
# ============================================================

video_path = (
    r"ws10\converted_mp4"
    r"\xsens_002_WS10_2023_09_21_in_camera_cropped_L_view.mp4"
)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("ERROR: Could not open video")
    exit()

print("Video opened successfully.")


# ============================================================
# 7. PIXEL → GROUND POSITION
# ============================================================

def pixel_to_ground(u, v):

    pixel = np.array([
        float(u),
        float(v),
        1.0
    ], dtype=np.float64)


    # --------------------------------------------------------
    # Pixel → camera ray
    # --------------------------------------------------------

    ray_camera = np.linalg.inv(K) @ pixel


    # --------------------------------------------------------
    # Camera ray → world ray
    # --------------------------------------------------------

    ray_world = R.T @ ray_camera

    ray_world = np.asarray(
        ray_world
    ).reshape(3)


    # --------------------------------------------------------
    # Camera world position
    # --------------------------------------------------------

    camera_position = np.asarray(
        CAMERA_POSITION
    ).reshape(3)


    # --------------------------------------------------------
    # Ground plane
    #
    # Z = 0
    # --------------------------------------------------------

    if abs(ray_world[2]) < 1e-8:

        return None


    lam = (
        -camera_position[2] /
        ray_world[2]
    )


    # --------------------------------------------------------
    # Intersection must be in front of camera
    # --------------------------------------------------------

    if lam <= 0:

        return None


    ground_point = (
        camera_position +
        lam * ray_world
    )


    X = float(
        ground_point[0]
    )

    Y = float(
        ground_point[1]
    )

    Z = float(
        ground_point[2]
    )


    return X, Y, Z


# ============================================================
# 8. SELECT CAR ANCHOR POINTS
# ============================================================

ret, first_frame = cap.read()

if not ret:

    print("ERROR: Could not read first frame")

    cap.release()

    exit()


# ------------------------------------------------------------
# Store manually selected points
# ------------------------------------------------------------

selected_points = []


def mouse_callback(event, x, y, flags, param):

    global selected_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(selected_points) < 4:

            selected_points.append([
                float(x),
                float(y)
            ])

            print(
                f"Point {len(selected_points)} selected: "
                f"({x}, {y})"
            )


# ============================================================
# SHOW FIRST FRAME
# ============================================================

selection_frame = first_frame.copy()


cv2.namedWindow(
    "SELECT CAR ANCHOR POINTS"
)

cv2.setMouseCallback(
    "SELECT CAR ANCHOR POINTS",
    mouse_callback
)


print()
print("================================================")
print("SELECT CAR ANCHOR POINTS")
print("================================================")
print("Click FOUR points on the CAR:")
print()
print("1. C1 = TOP-LEFT")
print("2. C2 = TOP-RIGHT")
print("3. C3 = BOTTOM-RIGHT")
print("4. C4 = BOTTOM-LEFT")
print()
print("Press ENTER after selecting all 4 points.")
print("Press R to reset points.")
print("================================================")


while True:

    display = selection_frame.copy()


    # --------------------------------------------------------
    # Draw selected points
    # --------------------------------------------------------

    for i, point in enumerate(
        selected_points
    ):

        x = int(point[0])
        y = int(point[1])


        cv2.circle(
            display,
            (x, y),
            8,
            (255, 0, 255),
            -1
        )


        cv2.putText(
            display,
            f"C{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )


    # --------------------------------------------------------
    # Draw temporary polygon
    # --------------------------------------------------------

    if len(selected_points) >= 2:

        pts = np.array(
            selected_points,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [pts],
            False,
            (255, 0, 255),
            2
        )


    # --------------------------------------------------------
    # Four points selected
    # --------------------------------------------------------

    if len(selected_points) == 4:

        pts = np.array(
            selected_points,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [pts],
            True,
            (255, 0, 255),
            3
        )


        cv2.putText(
            display,
            "CAR ANCHOR READY - PRESS ENTER",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            display,
            f"Select {4 - len(selected_points)} more point(s)",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


    cv2.imshow(
        "SELECT CAR ANCHOR POINTS",
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # R = reset
    # --------------------------------------------------------

    if key == ord("r"):

        selected_points = []

        print("Points reset.")


    # --------------------------------------------------------
    # ENTER = continue
    # --------------------------------------------------------

    elif key == 13:

        if len(selected_points) == 4:

            break

        print(
            "Please select all 4 points first."
        )


cv2.destroyWindow(
    "SELECT CAR ANCHOR POINTS"
)


# ============================================================
# 9. INITIALIZE RIGID CAR TRACKING
# ============================================================

old_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------------------------
# ORIGINAL CAR SHAPE
#
# These points NEVER change relative to one another.
# ------------------------------------------------------------

original_car_points = np.array(
    selected_points,
    dtype=np.float32
)


# ------------------------------------------------------------
# Current car position
# ------------------------------------------------------------

car_translation = np.array(
    [0.0, 0.0],
    dtype=np.float32
)


# ------------------------------------------------------------
# Current points used by optical flow
# ------------------------------------------------------------

car_points = (
    original_car_points.copy()
    .reshape(-1, 1, 2)
)


# ============================================================
# LUCAS-KANADE PARAMETERS
# ============================================================

lk_params = dict(

    winSize=(31, 31),

    maxLevel=3,

    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)


# ============================================================
# 10. FUNCTIONS FOR CAR ANCHOR
# ============================================================

def get_car_polygon(points):

    return np.array(
        points,
        dtype=np.int32
    ).reshape(-1, 1, 2)


def get_car_bbox(points):

    points = np.asarray(
        points
    )

    x_min = int(
        np.min(points[:, 0])
    )

    y_min = int(
        np.min(points[:, 1])
    )

    x_max = int(
        np.max(points[:, 0])
    )

    y_max = int(
        np.max(points[:, 1])
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


# ============================================================
# PERSON BBOX → CAR DISTANCE
# ============================================================

def bbox_to_car_distance(
    bbox,
    polygon
):

    """
    Distance from PERSON BOUNDING BOX to CAR polygon.

    If any part of the person's bbox overlaps
    the car polygon:

        distance = 0

    Otherwise:

        distance = minimum distance from bbox
                   corners/center to car polygon.
    """

    x1, y1, x2, y2 = bbox


    # --------------------------------------------------------
    # Points representing the bbox
    # --------------------------------------------------------

    points = [

        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),

        (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )
    ]


    distances = []


    for point in points:

        distance = cv2.pointPolygonTest(
            polygon,
            (
                float(point[0]),
                float(point[1])
            ),
            True
        )

        distances.append(
            float(distance)
        )


    # --------------------------------------------------------
    # If any point is inside car
    # --------------------------------------------------------

    if any(
        d >= 0
        for d in distances
    ):

        return 0.0


    # --------------------------------------------------------
    # Otherwise closest point
    # --------------------------------------------------------

    return min(
        abs(d)
        for d in distances
    )


# ============================================================
# PERSON BBOX / CAR POLYGON OVERLAP
# ============================================================

def bbox_polygon_overlap(
    bbox,
    polygon
):

    """
    Calculate percentage of PERSON bounding box
    that overlaps the moving CAR polygon.

    This is now the MAIN criterion for worker selection.
    """

    x1, y1, x2, y2 = bbox


    if x2 <= x1 or y2 <= y1:

        return 0.0


    # --------------------------------------------------------
    # Person mask
    # --------------------------------------------------------

    mask_person = np.zeros(
        first_frame.shape[:2],
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Car mask
    # --------------------------------------------------------

    mask_car = np.zeros(
        first_frame.shape[:2],
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Draw person bbox
    # --------------------------------------------------------

    cv2.rectangle(
        mask_person,
        (x1, y1),
        (x2, y2),
        255,
        -1
    )


    # --------------------------------------------------------
    # Draw car polygon
    # --------------------------------------------------------

    cv2.fillPoly(
        mask_car,
        [polygon],
        255
    )


    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    intersection = cv2.bitwise_and(
        mask_person,
        mask_car
    )


    intersection_area = cv2.countNonZero(
        intersection
    )


    person_area = cv2.countNonZero(
        mask_person
    )


    if person_area == 0:

        return 0.0


    return (
        intersection_area /
        person_area
    ) * 100.0


# ============================================================
# 11. FUNCTION TO SELECT CAR ANCHORS AGAIN
# ============================================================

def select_car_anchors(frame):

    global selected_points


    selection_frame = frame.copy()

    selected_points = []


    cv2.namedWindow(
        "SELECT CAR ANCHOR POINTS"
    )

    cv2.setMouseCallback(
        "SELECT CAR ANCHOR POINTS",
        mouse_callback
    )


    while True:

        display = selection_frame.copy()


        # ----------------------------------------------------
        # Draw points
        # ----------------------------------------------------

        for i, point in enumerate(
            selected_points
        ):

            x = int(point[0])
            y = int(point[1])


            cv2.circle(
                display,
                (x, y),
                8,
                (255, 0, 255),
                -1
            )


            cv2.putText(
                display,
                f"C{i + 1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )


        # ----------------------------------------------------
        # Temporary polygon
        # ----------------------------------------------------

        if len(selected_points) >= 2:

            pts = np.array(
                selected_points,
                dtype=np.int32
            )

            cv2.polylines(
                display,
                [pts],
                False,
                (255, 0, 255),
                2
            )


        # ----------------------------------------------------
        # Instruction
        # ----------------------------------------------------

        cv2.putText(
            display,
            "Select C1 C2 C3 C4 - ENTER",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        cv2.imshow(
            "SELECT CAR ANCHOR POINTS",
            display
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("r"):

            selected_points = []


        elif key == 13:

            if len(selected_points) == 4:

                break


    cv2.destroyWindow(
        "SELECT CAR ANCHOR POINTS"
    )


    return np.array(
        selected_points,
        dtype=np.float32
    )


# ============================================================
# 12. PROCESS VIDEO
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    # ========================================================
    # CURRENT FRAME GRAYSCALE
    # ========================================================

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # CAR TRACKING
    #
    # ONE TRANSLATION FOR THE ENTIRE CAR
    # ========================================================

    new_points, status, error = (
        cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            car_points,
            None,
            **lk_params
        )
    )


    # ========================================================
    # CHECK TRACKING
    # ========================================================

    tracking_ok = False


    if (
        new_points is not None
        and status is not None
    ):

        status = status.flatten()


        old_points_2d = (
            car_points.reshape(
                4,
                2
            )
        )


        new_points_2d = (
            new_points.reshape(
                4,
                2
            )
        )


        valid_old = (
            old_points_2d[
                status == 1
            ]
        )


        valid_new = (
            new_points_2d[
                status == 1
            ]
        )


        if len(valid_old) >= 3:

            tracking_ok = True


    # ========================================================
    # IF TRACKING IS GOOD
    # ========================================================

    if tracking_ok:

        # ----------------------------------------------------
        # Anchor displacements
        # ----------------------------------------------------

        displacements = (
            valid_new -
            valid_old
        )


        # ----------------------------------------------------
        # ONE COMMON TRANSLATION
        # ----------------------------------------------------

        dx = float(
            np.median(
                displacements[:, 0]
            )
        )


        dy = float(
            np.median(
                displacements[:, 1]
            )
        )


        translation = np.array(
            [dx, dy],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Update total translation
        # ----------------------------------------------------

        car_translation += translation


        # ----------------------------------------------------
        # Build current car
        #
        # RIGID TRANSLATION ONLY
        # ----------------------------------------------------

        current_car_points = (
            original_car_points +
            car_translation
        )


        # ----------------------------------------------------
        # Update optical-flow points
        # ----------------------------------------------------

        car_points = (
            current_car_points
            .copy()
            .reshape(4, 1, 2)
            .astype(np.float32)
        )


    else:

        # ----------------------------------------------------
        # Tracking lost
        # ----------------------------------------------------

        current_car_points = (
            original_car_points +
            car_translation
        )


        cv2.putText(
            frame,
            "CAR TRACKING LOST - PRESS R",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )


    # ========================================================
    # CURRENT CAR POLYGON
    # ========================================================

    car_polygon = get_car_polygon(
        current_car_points
    )


    # ========================================================
    # CURRENT CAR BBOX
    # ========================================================

    car_x1, car_y1, car_x2, car_y2 = (
        get_car_bbox(
            current_car_points
        )
    )


    # ========================================================
    # DRAW MOVING CAR
    # ========================================================

    cv2.polylines(
        frame,
        [car_polygon],
        True,
        (255, 0, 255),
        3
    )


    # ========================================================
    # DRAW C1 C2 C3 C4
    # ========================================================

    for i, point in enumerate(
        current_car_points
    ):

        x = int(point[0])
        y = int(point[1])


        cv2.circle(
            frame,
            (x, y),
            7,
            (255, 0, 255),
            -1
        )


        cv2.putText(
            frame,
            f"C{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2
        )


    # ========================================================
    # LABEL MOVING CAR
    # ========================================================

    cv2.putText(
        frame,
        "MOVING CAR ANCHOR",
        (
            car_x1,
            max(
                25,
                car_y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )


    # ========================================================
    # SHOW TRANSLATION
    # ========================================================

    cv2.putText(
        frame,
        f"Translation: "
        f"X={car_translation[0]:.1f}px "
        f"Y={car_translation[1]:.1f}px",
        (30, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )


    # ========================================================
    # YOLO POSE DETECTION
    # ========================================================

    results = model(
        frame,
        verbose=False,
        conf=PERSON_CONFIDENCE
    )


    result = results[0]


    # ========================================================
    # PERSON PROCESSING
    # ========================================================

    target_worker = None

    # --------------------------------------------------------
    # We now want HIGHER overlap to win.
    # --------------------------------------------------------

    best_overlap = -1.0


    if (
        result.keypoints is not None
        and result.boxes is not None
    ):

        keypoints = (
            result.keypoints.xy
            .cpu()
            .numpy()
        )


        keypoint_conf = (
            result.keypoints.conf
            .cpu()
            .numpy()
        )


        person_conf = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        person_boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )


        # ====================================================
        # CHECK EVERY PERSON
        # ====================================================

        for person_id, person in enumerate(
            keypoints
        ):

            if person_id >= len(
                person_conf
            ):

                continue


            confidence = float(
                person_conf[
                    person_id
                ]
            )


            if confidence < PERSON_CONFIDENCE:

                continue


            # ------------------------------------------------
            # PERSON BBOX
            # ------------------------------------------------

            if person_id >= len(
                person_boxes
            ):

                continue


            bx1, by1, bx2, by2 = (
                person_boxes[
                    person_id
                ].astype(int)
            )


            person_bbox = (
                bx1,
                by1,
                bx2,
                by2
            )


            # =================================================
            # CAR OVERLAP
            #
            # THIS IS NOW THE MAIN WORKER TEST
            # =================================================

            overlap = (
                bbox_polygon_overlap(
                    person_bbox,
                    car_polygon
                )
            )


            # ------------------------------------------------
            # Person must actually overlap car.
            #
            # A person standing near the car but outside
            # receives:
            #
            # overlap = 0%
            #
            # and is therefore NOT selected.
            # ------------------------------------------------

            if overlap < MIN_CAR_OVERLAP:

                continue


            # =================================================
            # ANKLE DETECTION
            #
            # OPTIONAL NOW
            # =================================================

            ankle_available = False

            ankle_point = None


            if (
                len(person) > 16
                and
                person_id < len(
                    keypoint_conf
                )
            ):

                left_ankle = person[15]

                right_ankle = person[16]


                left_conf = float(
                    keypoint_conf[
                        person_id
                    ][15]
                )


                right_conf = float(
                    keypoint_conf[
                        person_id
                    ][16]
                )


                # ------------------------------------------------
                # If both ankles are visible, use them.
                # ------------------------------------------------

                if (
                    left_conf >= ANKLE_CONFIDENCE
                    and
                    right_conf >= ANKLE_CONFIDENCE
                ):

                    ankle_u = float(
                        (
                            left_ankle[0] +
                            right_ankle[0]
                        ) / 2
                    )


                    ankle_v = float(
                        (
                            left_ankle[1] +
                            right_ankle[1]
                        ) / 2
                    )


                    ankle_point = (
                        ankle_u,
                        ankle_v
                    )


                    ankle_available = True


            # =================================================
            # POSITION FALLBACK
            # =================================================

            if ankle_available:

                position_point = ankle_point

                position_source = "ANKLE"


            else:

                # ------------------------------------------------
                # Ankles are hidden by car door/window.
                #
                # Use bottom-center of person's bbox.
                # ------------------------------------------------

                position_point = (
                    (
                        bx1 + bx2
                    ) / 2.0,

                    float(by2)
                )

                position_source = "BBOX"


            # =================================================
            # CAR GAP
            # =================================================

            if ankle_available:

                car_distance = (
                    bbox_to_car_distance(
                        person_bbox,
                        car_polygon
                    )
                )

            else:

                # ------------------------------------------------
                # Since this person already overlaps the car,
                # the effective gap is zero.
                # ------------------------------------------------

                car_distance = 0.0


            # =================================================
            # SELECT BEST WORKER
            # =================================================

            # ------------------------------------------------
            # Highest overlap wins.
            #
            # This means:
            #
            # Worker inside door/window
            #       ↓
            # Large bbox overlap
            #       ↓
            # TARGET WORKER
            #
            # Worker merely near car
            #       ↓
            # 0% overlap
            #       ↓
            # OTHER
            # ------------------------------------------------

            if overlap > best_overlap:

                best_overlap = overlap


                target_worker = {

                    "id":
                        person_id,

                    "confidence":
                        confidence,

                    "bbox":
                        person_bbox,

                    "ankle":
                        ankle_point,

                    "position":
                        position_point,

                    "position_source":
                        position_source,

                    "distance":
                        car_distance,

                    "overlap":
                        overlap
                }


        # ====================================================
        # DRAW ALL PEOPLE
        # ====================================================

        for person_id, person in enumerate(
            keypoints
        ):

            if person_id >= len(
                person_conf
            ):

                continue


            confidence = float(
                person_conf[
                    person_id
                ]
            )


            if confidence < PERSON_CONFIDENCE:

                continue


            if person_id >= len(
                person_boxes
            ):

                continue


            bx1, by1, bx2, by2 = (
                person_boxes[
                    person_id
                ].astype(int)
            )


            person_bbox = (
                bx1,
                by1,
                bx2,
                by2
            )


            # ------------------------------------------------
            # Calculate overlap for display
            # ------------------------------------------------

            overlap = (
                bbox_polygon_overlap(
                    person_bbox,
                    car_polygon
                )
            )


            # ------------------------------------------------
            # Is this the selected worker?
            # ------------------------------------------------

            is_target = (
                target_worker is not None
                and
                person_id ==
                target_worker["id"]
            )


            # ------------------------------------------------
            # Color
            # ------------------------------------------------

            if is_target:

                color = (0, 255, 0)

            else:

                color = (0, 0, 255)


            # ------------------------------------------------
            # Draw bbox
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                color,
                2
            )


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if is_target:

                label = "TARGET WORKER"

            else:

                label = "OTHER"


            cv2.putText(
                frame,
                label,
                (
                    bx1,
                    max(
                        25,
                        by1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )


            # ------------------------------------------------
            # YOLO confidence
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (
                    bx1,
                    by2 + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )


            # ------------------------------------------------
            # Show overlap for every person
            #
            # This helps us debug the selection.
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"OVERLAP {overlap:.1f}%",
                (
                    bx1,
                    by2 + 40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )


    # ========================================================
    # TARGET WORKER
    # ========================================================

    if target_worker is not None:

        # ----------------------------------------------------
        # Position source
        # ----------------------------------------------------

        u, v = (
            target_worker["position"]
        )


        # ----------------------------------------------------
        # Pixel → world position
        # ----------------------------------------------------

        X_world = pixel_to_ground(
            u,
            v
        )


        # ----------------------------------------------------
        # Draw worker position point
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (
                int(u),
                int(v)
            ),
            9,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # World position
        # ----------------------------------------------------

        if X_world is not None:

            X, Y, Z = X_world


            cv2.putText(
                frame,
                f"WORKER POSITION: "
                f"X={X:.2f} m  "
                f"Y={Y:.2f} m",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Z={Z:.2f} m",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ====================================================
        # WORKER / CAR RELATIONSHIP
        # ====================================================

        cv2.putText(
            frame,
            f"CAR GAP: "
            f"{target_worker['distance']:.0f}px",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"CAR OVERLAP: "
            f"{target_worker['overlap']:.1f}%",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"YOLO CONF: "
            f"{target_worker['confidence']:.2f}",
            (30, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Show where position came from
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"POSITION SOURCE: "
            f"{target_worker['position_source']}",
            (30, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO WORKER IN CAR",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "WS10 Worker Position Prototype",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Q = quit
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # ========================================================
    # R = MANUALLY RESELECT CAR
    # ========================================================

    if key == ord("r"):

        print()
        print("==============================================")
        print("MANUAL CAR ANCHOR RESET")
        print("==============================================")


        # ----------------------------------------------------
        # Select anchors on CURRENT frame
        # ----------------------------------------------------

        new_car_points = (
            select_car_anchors(frame)
        )


        # ----------------------------------------------------
        # Replace original shape
        # ----------------------------------------------------

        original_car_points = (
            new_car_points.copy()
        )


        # ----------------------------------------------------
        # Reset translation
        # ----------------------------------------------------

        car_translation = np.array(
            [0.0, 0.0],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Current points
        # ----------------------------------------------------

        car_points = (
            original_car_points
            .copy()
            .reshape(4, 1, 2)
            .astype(np.float32)
        )


        # ----------------------------------------------------
        # Reset optical flow reference
        # ----------------------------------------------------

        old_gray = frame_gray.copy()


        print(
            "Car anchor successfully reset."
        )


        continue


    # ========================================================
    # PREPARE NEXT FRAME
    # ========================================================

    old_gray = frame_gray.copy()


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Finished.")


# python pose_prototype.py