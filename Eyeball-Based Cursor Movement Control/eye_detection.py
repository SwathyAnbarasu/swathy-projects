import argparse
import math
import sched
import time

import cv2
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from scipy.constants import point
from scipy.spatial import distance as dist

import dlib


# @mayank408
# OOP Project
# 13/10/17

# Mouse Events Class

class Quartz:
    kCGEventRightMouseUp = None
    kCGEventOtherMouseUp = None
    kCGEventOtherMouseDown = None
    kCGEventRightMouseDown = None
    kCGEventMouseMoved = None
    kCGMouseButtonLeft = None
    kCGEventLeftMouseDown = None
    kCGEventLeftMouseUp = None
    kCGMouseEventClickState = None
    kCGHIDEventTap = None
    kCGEventLeftMouseDragged = None

    @classmethod
    def CGEventSetType(cls, theEvent, kCGEventLeftMouseUp):
        pass

    @classmethod
    def CGEventSetIntegerValueField(cls, theEvent, kCGMouseEventClickState, clickCount):
        pass

    @classmethod
    def CGEventPost(cls, kCGHIDEventTap, theEvent):
        pass

    @classmethod
    def CGEventCreateMouseEvent(cls, param: object, param1: object, param2: object, button: object) -> object:
        pass

    @classmethod
    def CGEventGetLocation(cls, param):
        pass

    @classmethod
    def CGEventCreate(cls, param):
        pass

    @classmethod
    def CGPointMake(cls, a, b):
        pass

    @classmethod
    def CGWarpMouseCursorPosition(cls, param):
        pass


def doubleClick(a, b, button=0, clickCount=None):
    print("Double click event")
    theEvent = Quartz.CGEventCreateMouseEvent(None, Mouse.down[button], (a, b), button)
    Quartz.CGEventSetIntegerValueField(theEvent, Quartz.kCGMouseEventClickState, clickCount)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, theEvent)
    Quartz.CGEventSetType(theEvent, Quartz.kCGEventLeftMouseUp)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, theEvent)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, theEvent)
    Quartz.CGEventSetType(theEvent, Quartz.kCGEventLeftMouseDown)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, theEvent)
    Quartz.CGEventSetType(theEvent, Quartz.kCGEventLeftMouseUp)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, theEvent)
    print("Double click event ended")


def __mouse_event(type, a, b):
    mouse_ = Quartz.CGEventCreateMouseEvent(None, type, (a, b), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, mouse_)


# noinspection PyCompatibility
def _mouse_event():
    pass


def press(a, b, button=0):
    event = Quartz.CGEventCreateMouseEvent(None, Mouse.down[button], (a, b), button)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def release(a, b, button=0):
    event = Quartz.CGEventCreateMouseEvent(None, Mouse.up[button], (a, b), button)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def torelative(a, b, curr_pos=None):
    Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    a += curr_pos.a
    b += curr_pos.b
    return [a, b]


def move_rel(a, b):
    [a, b] = torelative(a, b)
    moveEvent = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, Quartz.CGPointMake(a, b), 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, moveEvent)


def mouseEvent(type, posx, posy):
    theEvent = Quartz.CGEventCreateMouseEvent(None, type, (posx, posy), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, theEvent)


def mouse_event():

    pass


def move(a, b):
    mouse_event()
    Quartz.CGWarpMouseCursorPosition((a, b))


def position():
    Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return point.a, point.b


def click(button=0):
    a, b = position()
    press(a, b, button)
    release(a, b, button)


def click_pos(a, b, button=0):
    move(a, b)
    click(button)


def mousedrag(posx, posy):
    mouseEvent(Quartz.kCGEventLeftMouseDragged, posx, posy)


class Mouse:
    down = [Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown, Quartz.kCGEventOtherMouseDown]
    up = [Quartz.kCGEventLeftMouseUp, Quartz.kCGEventRightMouseUp, Quartz.kCGEventOtherMouseUp]
    [LEFT, RIGHT, OTHER] = [0, 1, 2]


# Detecting Blinking of eyes


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    Ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return Ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# defining two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

s = sched.scheduler(time.time, time.sleep)


def print_time():
    if ear > .3:
        pass


def print_some_times():
    print(time.time())
    s.enter(1, 1, print_time, ())
    s.run()
    print(time.time())


# initialize lib's face detector (HOG-based) and then create
# the facial landmark predictor
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("Starting live video stream...")

vs = VideoStream(src=0).start()

fileStream = False
time.sleep(1.0)
currentCount = 0

mouse = Mouse()
face_cascade = cv2.CascadeClassifier('res/caseharden_frontal face_default.xml')

while True:

    # if this is a file video stream, then we need to check if
    # there are any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)

    height, width, c = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.circle(frame, (int(width / 2), int(height / 2)), 4, (0, 0, 255), 2)
    cv2.circle(frame, (int(width / 2), int(height / 2)), 20, (128, 0, 128), 2)
    face = face_cascade.detectMultiScale(gray, 1.15)

    min_dis = 100000
    x = 0
    y = 0
    w = 0
    h = 0
    for (mx, my, mw, mh) in face:
        d = dist.euclidean((mx + w / 2, my + h / 2), (width / 2, height / 2))
        if d < min_dis:
            min_dis = d
            x = mx
            y = my
            w = mw
            h = mh

    if x != 0 and y != 0 and w != 0 and h != 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # to avoid mirror image
        x = width - (x + w)
        slope = math.atan(float((y + h / 2 - height / 2) / (x + w / 2 - width / 2)))

        r = int(width / 2 + 100 * math.cos(slope))
        t = int(height / 2 + 100 * math.sin(slope))

        cv2.line(frame, (int(width / 2), int(height / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 3, (255, 0, 0), 2)
        d = dist.euclidean((x + w / 2, y + h / 2), (width / 2, height / 2))
        c, e = position()

        if d > 20:
            speed = 5
            if x + w / 2 > width / 2:
                move(c + speed * math.cos(slope), e + speed * math.sin(slope))
            else:
                move(c - speed * math.cos(slope), e - speed * math.sin(slope))

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    prevcount = 0
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes\
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink

        if not leftEAR >= EYE_AR_THRESH - 0.12 and rightEAR > EYE_AR_THRESH - 0.12:
            print("Left Eye Blinked")
            m, n = position()
            click_pos(m, n, 0)
            time.sleep(1)

        elif not rightEAR >= EYE_AR_THRESH - 0.12 and leftEAR > EYE_AR_THRESH - 0.12:
            print("Right Eye Blinked")
            m, n = position()
            click_pos(m, n, 1)
            time.sleep(1)

        if leftEAR < EYE_AR_THRESH - 0.12 and rightEAR < EYE_AR_THRESH - 0.12:
            print("Both Eyes Blinked")
            m, n = position()
            doubleClick(m, n, 0)
            COUNTER += 1
            TOTAL += 1
            prevcount = COUNTER
            time.sleep(1)

        # the computed eye aspect ratio for the frame

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Left: {:.2f}".format(leftEAR), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Right: {:.2f}".format(rightEAR), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        break

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
