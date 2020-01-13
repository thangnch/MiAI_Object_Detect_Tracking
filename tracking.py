import cv2
import sys

# Tao doi tuong tracking
my_track_method = cv2.TrackerMOSSE_create()

# Doc file video
cap = cv2.VideoCapture("race.mp4")

# Doc frame dau tien de nguoi dung chon doi truong can track
ret , frame = cap.read()
if not ret:
    print('Khong tim thay file video')
    sys.exit()

# Chon doi tuong va init tracking
select_box = cv2.selectROI(frame, showCrosshair=True)
my_track_method.init(frame, select_box)

while True:
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        # Neu khong doc duoc tiep thi out
        break

    # Update tracker
    ret, select_box = my_track_method.update(frame)

    if ret:
        # Neu nhu tracking duoc thanh cong
        tl, br  = (int(select_box[0]), int(select_box[1])), (int(select_box[0] + select_box[2]), int(select_box[1] + select_box[3]))
        cv2.rectangle(frame, tl, br, (0, 255, 0), 2, 2)
    else:
        # Neu nhu khong tim thay doi tuong
        cv2.putText(frame, "Object can not be tracked!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Hien thi thong tin va video
    cv2.putText(frame, "MiAI Demo Object Tracking", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2);
    cv2.imshow("Video", frame)

    # Nhan dien thao tac nhan phim
    key = cv2.waitKey(1) & 0xff

    # Neu nhan q thi thoat
    if key == ord('q'):
        break
    if key == ord('s'):
        # Select lai ROI moi
        select_box = cv2.selectROI(frame, showCrosshair=True)
        my_track_method.clear()
        my_track_method = cv2.TrackerMOSSE_create()
        my_track_method.init(frame, select_box)
