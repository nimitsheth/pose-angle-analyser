import cv2
import mediapipe as mp
import time
import math
import imutils as imutils
import threading
# from pconst import const



starttime=0.00

elapsed=0.00

class poseDetector():
    start = False

    #starttime=0.0
    starttime = 0.0
    lasttime = starttime
    pTime = 0
    elapsed=0.0
    stretchtime = False
    rep_count = 0



    def __init__(self, mode=False, upBody=False, smooth=True,detectionCon=0.5, trackCon=0.5):


        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def stpwatch(self):
        #self.currenttime=time.time()
        """lasttime = time.time()
        self.elapse=round((time.time() - starttime), 0)"""
        cTime = time.time()

        elapse = cTime - self.pTime
        pTime = cTime



    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360



        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (2, 2, 2), 3)
            cv2.line(img, (x3, y3), (x2, y2), (2, 2, 2), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle))+"deg", (x2 - 50, y2 + 50),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            msg="hold it for 8sec"
            wrn="Keep your knee bent"
            success="Successfully completed rep count"





           #Detecting the angle<120 to start the timer


            if ((angle<120) and (self.start==False)and(self.stretchtime==False)) :
                cv2.putText(img,msg,(170,50),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                print("start")
                self.start = True
                #recording start time

                """try: const.starttime = time.time()
                except:print()"""
                self.starttime=time.time()


             #displaying the timer

            if (self.start):
                #self.elapsed = time.time() - const.starttime
                self.elapsed = time.time() - self.starttime


                cv2.putText(img, f"timer started :{round(self.elapsed,0)}", (170, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            #Displaying warning to keep leg bent
            if (self.start and angle > 100):

                cv2.putText(img, wrn, (170, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            #if timer hits 8sec rep successful

            if (self.start and round(self.elapsed,0)==8.0):
                self.start=False
                print("startsession"+str(self.start))
                self.stretchtime=True
                elapsed =0.0
                self.rep_count += 1
                cv2.putText(img, success + f"{self.rep_count}", (1, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 10, 0), 2)
                print("rep1"+str(self.rep_count))



            #stretchtime starts
            if(self.stretchtime and self.start==False):
                print("rep2" + str(self.rep_count))
                cv2.putText(img, success + f"{self.rep_count}", (1, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 10, 0), 2)

                cv2.putText(img, "stretch your leg", (5, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 10,0), 2)

            #if legs stretched i.e. angle>120 stretchtime over
            if(self.start==False and angle>150):


                self.stretchtime=False
                print("stretchtime2"+str(self.stretchtime))






            """if (self.start and angle > 120):
                self.start = False
                self.starttime = time.time()

                cv2.putText(img, "unsuccessful rep", (170, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)"""
        return angle

def main():
    cap = cv2.VideoCapture('E:/KneeBend.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img,draw=False)
        img = imutils.resize(img, width=720)
        lmList = detector.findPosition(img, draw=False)
        detector.findAngle(img,23,25,27,draw=True)
        if len(lmList) != 0:
            print(lmList[14])
           # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()

        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    starttime = 0.00

    elapsed = 0.00
    main()