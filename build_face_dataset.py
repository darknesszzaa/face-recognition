from imutils.video import VideoStream
import imutils
import argparse
import cv2
import time
import requests
import json


class Application:
    def __init__(self, vs):

        # load OpenCV's Haar cascade for face detection from disk
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # initialize the video stream, allow the camera sensor to warm up,
        # and initialize the total number of example faces written to disk
        # thus far
        print("[INFO] starting video stream...")

        self.initial_user()

        print("[INFO] Press S on video for snapshot and press ESC for exit.")

        self.vs = vs
        time.sleep(2.0)
        self.total = 0
        # loop over the frames from the video stream
        while True:

            self.video_loop()

            self.key = cv2.waitKey(100) & 0xFF

            if self.key == 115:  # Press 'S' for save screenshot video
                self.take_snapshot()

                self.total += 1

            elif self.key == 27:
                break

        # print the total faces saved and do a bit of cleanup
        print("[INFO] {} face images stored".format(self.total))

        self.destructor()

    def initial_user(self):

        self.face_id = input("\n Enter User ID and press <return> ==>  ")
        self.response = requests.get(
            "http://covid.rvconnex.com/users/username/" + self.face_id,
            headers={
                "authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjA2LCJyb2xlIjoiSFIiLCJpYXQiOjE2MDQ3NzA3NTEsImV4cCI6MTYzNjMwNjc1MX0.FIJphYIuF8vpx2q4n2WcAZBj5xYMuQMITzGTJNHEa58"
            },
        )
        self.jsonResponse = json.loads(self.response.content)
        if self.response.status_code == 200:
            self.user_id = self.jsonResponse["id"]
            print(self.user_id)

        else:
            print("[INFO] User ID is invalid please try again.")
            self.initial_user()

    def video_loop(self):
        # grab the frame from the threaded video stream, clone it, (just
        # in case we want to write it to disk), and then resize the frame
        # so we can apply face detection faster
        self.frame = self.vs.read()
        self.orig = self.frame.copy()
        self.frame = imutils.resize(self.frame, width=800)
        # detect faces in the grayscale frame
        self.rects = self.detector.detectMultiScale(
            cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # loop over the face detections and draw them on the frame
        for (x, y, w, h) in self.rects:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", self.frame)

    def take_snapshot(self):
        cv2.imwrite(
            "dataset/User." + str(self.user_id) + "." + str(time.time()) + ".jpg",
            self.orig,
        )
        print("[INFO] Recorded User ID : " + str(self.face_id))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.vs.stop()
        cv2.destroyAllWindows()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", default="./",
#     help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
vs = VideoStream(src=0).start()
pba = Application(vs)