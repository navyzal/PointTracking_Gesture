import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import pyautogui as pg
from pynput.mouse import Button, Controller
from math import atan2, degrees, dist, radians, sin, cos

# def getAngle(index_X, index_Y, wrist_X, wrist_Y):
#     degree = degrees(atan2(index_Y-wrist_Y, index_X-wrist_X))
#     print(degree)

#     pass

def getDelta(index, wrist, r_dist):
    global prev_position, scaleX, scaleY, imageWidth, imageHeight, accel

    rad = atan2((index[1]-wrist[1])*imageHeight, (index[0]-wrist[0])*imageWidth)

    x2 = wrist[0]*imageWidth + r_dist * cos(rad)
    y2 = wrist[1]*imageHeight + r_dist * sin(rad)
    
    x = (x2 - prev_position[0]) * scaleX * accel
    y = (y2 - prev_position[1]) * scaleY * accel

    global moveDelta
    x = 0 if abs(x) < moveDelta else x
    y = 0 if abs(y) < moveDelta else y

    return int(x), int(y)


def move(landmark, image, gesture):
    imageHeight, imageWidth, _ = image.shape

    # WRIST
    wrist = (landmark[mp_hands.HandLandmark.WRIST].x, landmark[mp_hands.HandLandmark.WRIST].y)
    # INDEX_FINGER_MCP
    index = (landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)

    global dragCount, isClicked, prev_position, mouse

    p1 = (int(wrist[0]*imageWidth), int(wrist[1]*imageHeight))
    p2 = (int((index[0]*imageWidth + (index[0]-wrist[0])*imageWidth)), int((index[1]*imageHeight + (index[1]-wrist[1])*imageHeight)))

    if gesture == 'pointing':
        if prev_position is None:
            prev_position = p2
        x, y = getDelta(index, wrist, dist(p1, p2))
        mouse.move(x,y)
        prev_position = p2
    elif gesture == 'grab':
        if dragCount <= dragCountThreshould:
            dragCount += 1
        else:
            if prev_position is None:
                prev_position = p2
            mouse.click(Button.left)
            mouse.press(Button.left)
            x, y = getDelta(index, wrist, dist(p1, p2))
            mouse.move(x,y)
            prev_position = p2
            isClicked = True
    elif gesture == 'five':
        if isClicked:
            mouse.release(Button.left)
            isClicked = False
        prev_position = None
        dragCount = 0
    else:
        pass

    if gesture == 'pointing':
        ix = int(landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * imageWidth)
        iy = int(landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * imageHeight)
        cv2.circle(image, (ix, iy), 20, (255,255,0), -1)
    elif gesture == 'click' or gesture == 'click_hold' or gesture == 'grab':
        color = (153,51,255) if dragCount >= dragCountThreshould else (255,255,0)
        ix = int((landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * imageWidth + int(landmark[mp_hands.HandLandmark.THUMB_TIP].x * imageWidth)) / 2)
        iy = int((landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * imageHeight + int(landmark[mp_hands.HandLandmark.THUMB_TIP].y * imageHeight)) / 2)
        cv2.circle(image, (ix,iy), 20, color, -1)

    return angle


def moveScale(landmark, image, gesture):
    img_height, img_width, _ = image.shape
    # WRIST
    wrist = (landmark[mp_hands.HandLandmark.WRIST].x, landmark[mp_hands.HandLandmark.WRIST].y)
    # INDEX_FINGER_MCP
    index = (landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)

    angle = degrees(atan2(index[1]-wrist[1], index[0]-wrist[0]))

    global dragCount, isClicked, scaleDelta, prev_position, scaleX, scaleY, mouse

    p1 = (int(wrist[0]*img_width), int(wrist[1]*img_height))
    p2 = (int((index[0]*img_width + (index[0]-wrist[0])*img_width)), int((index[1]*img_height + (index[1]-wrist[1])*img_height)))


    r_dist = dist(p1, p2)

    rad = atan2((index[1]-wrist[1])*img_height, (index[0]-wrist[0])*img_width)
    xx = wrist[0]*img_width + r_dist * cos(rad)
    yy = wrist[1]*img_height + r_dist * sin(rad)
    p2 = (int(xx), int(yy))

    if gesture == 'pointing' or gesture == 'grab':
        cv2.line(image, p1, p2, (255, 255, 0), 2)
        cv2.circle(image, p2, 20, (255,255,0),-1)

    if prev_position is None:
        prev_position = p2

    delta = dist(prev_position, p2)

    if delta < 10:
        p2 = prev_position
    else:
        prev_position = p2

    # Multiply by the screen vs. webcam ratio 
    x = p2[0] * scaleX
    y = p2[1] * scaleY

    if gesture == 'pointing':
        if isClicked:
            pg.mouseUp(x, y, button='left')
            dragCount = 0
            isClicked = False
        pg.moveTo(x, y)
    elif gesture == 'grab':
        if dragCount >= 5:
            pg.drag(x, y, button='left')
            global this_action
            this_action = 'DRAG'
        else:
            if not isClicked:
                pg.mouseDown(x, y, button='left')
                isClicked = True
            dragCount += 1
    elif gesture == 'five':
        if isClicked:
            pg.mouseUp(x, y, button='left')
            dragCount = 0
            isClicked = False
    else:
        pass

    return angle


pg.PAUSE = 0
pg.FAILSAFE = False

# actions = ['come', 'away', 'spin']
# actions = ['pointing', 'click', 'click_hold', 'grab']
# actions = ['five', 'pointing', 'click_hold', 'grab'] #model_v4.h5, model_v5.h5
# actions = ['five','flick_RHRL', 'flick_LHLR']  #model_v6_flick.h5, model_v6_flick02.h5
# actions = ['five', 'pointing', 'click'] #model_v6.h5
# actions = ['pointing', 'click'] #model_v7.h5, model_v8.h5
actions = ['five', 'pointing', 'grab'] #model_v9.h5

seq_length = 5
model = load_model('models/model_v9.h5')

#############################
# Webcam 1 vs. ScreenSize 1 #
#############################
this_action = '?'
scaleDelta = 1
dragCount = 0
isClicked = False
prev_position = None

#############################
# as the same as moune      #
#############################
moveDelta = 20
dragCountThreshould = 5
isMoving = False
accel = 1.3 # mouse pointer acceleration 

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

# Ratio between cam resolution and screen size
screenWidth, screenHeight = pg.size()
imageWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
imageHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scaleX = screenWidth/imageWidth
scaleY = screenHeight/imageHeight

prev_position = (int(screenWidth/2), int(screenHeight/2))
mouse = Controller()

seq = []
action_seq = []
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        img = img.copy()
        start = time.time()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                # degree = move(res.landmark, img, this_action)
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                # degree = moveScale(res.landmark, img, this_action)
                move(res.landmark, img, this_action)

                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                cv2.putText(img, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(img, f'{this_action.upper()} : conf[{round(conf,1)}], isClicked[{isClicked}]', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('Hand Gesture', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break



cap.release()
cv2.destroyAllWindows()