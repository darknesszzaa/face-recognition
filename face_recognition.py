from imutils.video import VideoStream
import imutils
import argparse
import cv2
import requests
import json


class User:
    def __init__(self, id, full_name, user_id):
        self.id = id
        self.full_name = full_name
        self.user_id = user_id

    def __eq__(self, id):
        return self.id == id


class Application:
    def __init__(self, vs):

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer/trainer.yml")
        self.cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # iniciate id counter
        self.id = 0

        self.names = []

        self.initial_user_list()

        print("[Info] Initialize and start realtime video capture.")
        # Initialize and start realtime video capture
        self.vs = vs

        # Define min window size to be recognized as a face
        self.minW = 64
        self.minH = 48

        while True:

            self.video_loop()

            k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        self.destructor()

    def initial_user_list(self):
        self.response = requests.get(
            "http://covid.rvconnex.com/users",
            headers={
                "authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjA2LCJyb2xlIjoiSFIiLCJpYXQiOjE2MDQ3NzA3NTEsImV4cCI6MTYzNjMwNjc1MX0.FIJphYIuF8vpx2q4n2WcAZBj5xYMuQMITzGTJNHEa58"
            },
        )
        self.jsonResponse = json.loads(self.response.content)
        for child in self.jsonResponse:
            self.names.append(User(child["id"], child["fullName"], child["userName"]))

        print("[Info] Initial user list completed.")

    def video_loop(self):
        self.img = self.vs.read()
        self.img = imutils.resize(self.img, width=800)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.faces = self.faceCascade.detectMultiScale(
            self.gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(self.minW), int(self.minH)),
        )

        for (x, y, w, h) in self.faces:

            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.id, self.confidence = self.recognizer.predict(
                self.gray[y : y + h, x : x + w]
            )

            # Check if confidence is less them 100 ==> "0" is perfect match
            if self.confidence < 80:
                for data in self.names:
                    if data.id == self.id:
                        self.id = data.full_name
                        self.url = "http://covid.rvconnex.com/health/auto-record"
                        self.myobj = {
                            "userId": data.user_id,
                            "temperature": 36.5,
                            "timeline": "Morning",
                            "workingStatus": 1,
                            "healthCondition": True,
                            "journey": "Record from face detection camera in office area.",
                        }
                        self.res = requests.post(
                            self.url,
                            data=json.dumps(self.myobj),
                            headers={
                                "Content-type": "application/json",
                                "authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjA2LCJyb2xlIjoiSFIiLCJpYXQiOjE2MDQ3NzA3NTEsImV4cCI6MTYzNjMwNjc1MX0.FIJphYIuF8vpx2q4n2WcAZBj5xYMuQMITzGTJNHEa58",
                            },
                        )
                        break
                self.confidence = "  {0}%".format(round(100 - self.confidence))
            else:
                self.id = "unknown"
                self.confidence = "  {0}%".format(round(100 - self.confidence))

            cv2.putText(
                self.img,
                str(self.id),
                (x - 75, y - 5),
                self.font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                self.img,
                str(self.confidence),
                (x + 5, y + h - 5),
                self.font,
                1,
                (255, 255, 0),
                1,
            )

            cv2.imshow("camera", self.img)

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.vs.stop()
        cv2.destroyAllWindows()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
vs = VideoStream(src=0).start()
pba = Application(vs)