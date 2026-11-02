import cv2
import numpy as np
from statistics import mode

cap = cv2.VideoCapture(r"C:\Users\gshre\Downloads\Volleyball (1).mp4")

previous_gray = None

# Store player counts from different frames
team1_counts = []
team2_counts = []

# Store ball positions
ball_path = []

frame_number = 0
sample_rate = 10

# Approximate horizontal net position
NET_Y_RATIO = 0.39

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    h, w = frame.shape[:2]

    # BLACK OUT THE BACKGROUND BALL

    ball_radius = int(0.012 * w)

    # ball on the table 
    cv2.circle(frame, (int(0.384 * w), int(0.167 * h)), ball_radius, (0, 0, 0), -1)

    # 1. BALL TRACKING USING HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 100, 150])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_ball = None
    best_area = 0

    # find jerseys and mark them
    jersey_mask = np.zeros(mask.shape, np.uint8)
    for c in contours:
        if cv2.contourArea(c) >= 800:
            cv2.drawContours(jersey_mask, [c], -1, 255, -1)
    jersey_mask = cv2.dilate(jersey_mask, None, iterations=5)

    for c in contours:

        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if 30 < area < 800 and circularity > 0.60:

            # skip anything touching a jersey
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if jersey_mask[int(cy), int(cx)] == 255:
                continue

            if area > best_area:
                best_area = area
                best_ball = c

    if best_ball is not None:

        (x, y), radius = cv2.minEnclosingCircle(best_ball)

        center = (int(x), int(y))

        ball_path.append(center)

        if len(ball_path) > 15:
            ball_path.pop(0)

    # Draw ball trajectory
    for i in range(1, len(ball_path)):

        cv2.line(
            frame,
            ball_path[i-1],
            ball_path[i],
            (0, 0, 255),
            2
        )

    # 2. PLAYER COUNT USING FRAME

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous_gray is not None:

        # Process every 10th frame
        if frame_number % sample_rate == 0:

            difference = cv2.absdiff(gray, previous_gray)

            _, thresh = cv2.threshold(
                difference, 25, 255, cv2.THRESH_BINARY
            )

            thresh = cv2.dilate(thresh, None, iterations=3)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            team1 = 0
            team2 = 0

            net_y = int(h * NET_Y_RATIO)

            for c in contours:

                area = cv2.contourArea(c)

                if area > 1500:

                    x, y, bw, bh = cv2.boundingRect(c)

                    cy = y + bh // 2

                    if cy < net_y:
                        team1 += 1
                    else:
                        team2 += 1

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + bw, y + bh),
                        (0, 255, 0),
                        2
                    )

            team1_counts.append(team1)
            team2_counts.append(team2)

            cv2.putText(
                frame, "Team 1: " + str(team1),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2
            )

            cv2.putText(
                frame, "Team 2: " + str(team2),
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2
            )

    # Update previous frame EVERY iteration
    previous_gray = gray.copy()

    cv2.imshow("Volleyball Tracking", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. FINAL PLAYER COUNT USING MODE

if team1_counts and team2_counts:

    print("Final Player Count")

    print("Team 1:", mode(team1_counts))
    print("Team 2:", mode(team2_counts))

else:
    print("No frames were sampled.")