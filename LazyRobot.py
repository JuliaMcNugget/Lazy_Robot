import pyrealsense2 as rs
import cv2, pickle
import numpy as np
import cv2.aruco as aruco
import time, maestro, pyautogui    #Maestro File Updated
import tkinter as tk
from tkinter import Label
import threading, edge_tts, asyncio, playsound
import PIL 

#--------------------------- Load saved descriptors ------------------------------------------------
with open("orb_descriptors_by_label.pkl", "rb") as f:
    descriptors_lables = pickle.load(f)
#     End -------------------------------------------------------------------------------------------
# ---------------------------- ArUco Set Up ---------------------------------------------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
marker_length = 100  # mm
label_dict = {}
i = 1
for label in descriptors_lables:
    label_dict[label] = i
    i= i+1
#TODO Create dictionary with Object to arUco 
# ===================================================================================================
#--------------------------- Set Up Orb ------------------------------------------------
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
keypoints = ""
descriptors = ""
#------------------------------Window Setup---------------------------------------------------
    #TODO Fix Image TK
window = tk.Tk()
window.title("Face Recognition")
eyes_image = Image.open("eyes.jpg")
eyes_img = ImageTk.PhotoImage(eyes_image)
stranger_image = Image.open("stranger_danger.jpg")
stranger_img = ImageTk.PhotoImage(stranger_image)
label = Label(window, image=eyes_img, text="Initializing...", compound="bottom")
label.pack()

switch_to_danger = False
recognized_name = "Initializing..."
last_detection_time = 0
detection_delay = 2  
#-----------------------------------------------------------------------------------------------------------
# ------- Initialize Robot --------------------------------------------------------
    #TODO Find the right list number for the arm and hands
Robot = maestro.Controller()
waist = 2
htilt = 3
hturn = 4
wheel_1 = 0
wheel_2 = 1
arm = 5
servos = [waist, htilt, hturn, wheel_1, wheel_2, arm]
servos_position = [6000, 6000, 6000, 6000, 6000]
center_x = 0.00
center_y = 0.00
#======================================================================
for servo in servos:
    Robot.setAccel(servo, 2)
    Robot.setRange(servo, 3000, 9000)
    Robot.setSpeed(servo, 100)
    Robot.setTarget(servo, 6000)  # Neutral
Robot.setTarget(waist, 4000)
Robot.setTarget(waist, 7000)

def detection():
    working = True
    while working:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
    
        if not color_frame:
            continue

        img_BW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = orb.detectAndCompute(img_BW, None)
        best_label = None
        best_score = 0
        if descriptors is not None:
            # Loop through each label’s saved descriptors
            for label, desc_list in descriptors_lables.items():
                for known_desc in desc_list:
                    if known_desc is None or len(known_desc) == 0:
                        continue

                    matches = bf.match(known_desc, descriptors)
                        #TODO verify the distance number
                    good_matches = [m for m in matches if m.distance < 20]

                    if len(good_matches) > best_score:
                        best_score = len(good_matches)
                        best_label = label
        frame_with_kp = cv2.drawKeypoints(frame, keypoints, None, color=(255, 0, 0))
        if label:
            cv2.putText(frame_with_kp, f"{label.upper()} DETECTED", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            working = False
                    # Show original image
        cv2.imshow("Camera View", frame_with_kp)
        cv2.waitKey(0)
        return best_label

def stop():
    print("Stopping all movement...")
    for servo in servos:
        Robot.setTarget(servo, 6000)
    time.sleep(0.5)

def forward(offset_z):
    duration = offset_z*.002
    print("Moving forward")
    Robot.setTarget(wheel_1, 7000)
    Robot.setTarget(wheel_2, 5000)
    time.sleep(duration)
    stop()

def backward(offset_z):
    duration = offset_z*.002
    print("Moving back")
    Robot.setTarget(wheel_1, 5000)
    Robot.setTarget(wheel_2, 7000)
    time.sleep(duration)
    stop()

def turn_left(duration):
    print("Turning left")
    Robot.setTarget(wheel_1, 8000)
    Robot.setTarget(wheel_2, 6000)
    time.sleep(duration)
    stop()


def turn_right(duration):
    print("Turning right")
    Robot.setTarget(wheel_1, 6000)
    Robot.setTarget(wheel_2, 4000)
    time.sleep(duration)
    stop()

# Camera pose estimator
def get_camera_position_from_marker(marker_world_pos, rvec, tvec):
    R_ct, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape((3, 1))
    R_tc = R_ct.T
    t_tc = -np.dot(R_tc, tvec)
    marker_x, marker_y = marker_world_pos
    marker_world = np.array([[marker_x], [marker_y], [0.0]])
    camera_world = marker_world + t_tc[0:3]
    return float(camera_world[0]), float(camera_world[1])

marker_world_positions = {1: (0, 4), 2: (4, 8), 3: (8, 4), 4: (4, 0)}
coordinate_updates = []
finished = False

# Load calibration
calib = np.load("calibration_data.npz")
camera_Matrix = calib["cameraMatrix"]
distCoeffs = calib["distCoeffs"]

# ArUco setup

pipeline = rs.pipeline()
pipeline.start()
# -------------------------Text Setup ---------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 0, 0)  # Red

#--------------------------- Recognition Set Up ------------------------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")  # Load trained model
labels = {}
pickle_file_path = 'labels.pickle'
try:
    with open(pickle_file_path, 'rb') as f:
        original_labels = pickle.load(f)
        labels = {v: k for k, v in original_labels.items()}
    print("Label IDs and Names Mapping:")
    for key, value in labels.items():
        print(f"ID: {value}, Name: {key}")
except FileNotFoundError:
    print(f"Error: The file '{pickle_file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#----------------------------- Facial Recognition Setup ---------------------------------------

def facial_recognition():
    global switch_to_danger, recognized_name, last_detection_time
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
    
        if not color_frame:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print("faces detected")

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                id_, conf = recognizer.predict(roi_gray)

                if conf >= 69:  # Confidence threshold for recognition might have to tweak this some
                    new_name = labels.get(id_, "Unknown")
                    if recognized_name != new_name:
                        recognized_name = new_name
                        switch_to_danger = False
                        last_detection_time = time.time()
                    
                else:
                    if recognized_name  != "Stranger Danger":
                       recognized_name = "Stranger Danger"
                       switch_to_danger = True
                       last_detection_time = time.time()
                
                

                cv2.rectangle(frame, (x, y), (x+w, y+h), (177, 73, 238), 2)
                text = recognized_name
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_y = y + text_size[1] + 5 + h
                cv2.putText(frame, text, (x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 5)
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 5)
                
                for (ex, ey, ew, eh) in smiles:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 2)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            if time.time() - last_detection_time > detection_delay:
                recognized_name = "No one detected"
                switch_to_danger = False

        window.after(100, update_gui)
        cv2.imshow("Main Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
# =========================================================================================================
def update_gui():
    global switch_to_danger, recognized_name
    if switch_to_danger:
        label.config(image=stranger_img, text="Stranger Danger", compound="bottom")
    else:
        label.config(image=eyes_img, text=recognized_name, compound="bottom")

def get_closest_marker(ids, tvecs):
    if ids is None:
        return None, None
    min_z = float("inf")
    best_id = None
    best_vec = None
    for i, marker_id in enumerate(ids.flatten()):
        z = tvecs[i][0][2]
        if z < min_z:
            min_z = z
            best_id = marker_id
            best_vec = tvecs[i][0]
    return best_id, best_vec

previous_ID = 0
current_ID = 0
#======================================================================\
                # Text to Speech
class Talk:
    def __init__(self, voice="en-US-ChristopherNeural"):
        self.voice = voice

    async def _speak_async(self, text):
        temp_file = "tts.wav"
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(temp_file)  ##this will save to the same directory, but I did create a tmp directory 
        playsound.playsound(temp_file)

    def say(self, text):
        threading.Thread(target=self._run_async, args=(text,), daemon=True).start()

    def _run_async(self, text):
        asyncio.run(self._speak_async(text))
def main_speak():
    speaker = Talk(voice="en-US-GuyNeural")  # Try other voices too
    speaker.say("Hello! I am your robot.")
    
    # Prevent script from exiting too fast
    import time
    time.sleep(3)
main_speak()
                # END
#=========================================================================================================
# Initial Process ----------------------------------------------------------
def arm_raise():
    Robot.setTarget(arm, 9000)

def arm_drop():
    Robot.setTarget(arm, 7000)


#TODO Boot Up Text
arm_raise()
acquired = 0
delivered = 1
distance = 0
label = facial_recognition()
#TODO Ask for object to identify
detection()
# TODO
while True:
    #TODO Find world position
    #TODO Drop Object
    #TODO "Cleaning complete Text"
    # Camera Image Process ============================================
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    
    if not color_frame:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #==================================================================
    # arUuco ID and positioning =======================================
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
    # I think the issue we're having is when we lose track of a marker
    # maybe noting that if the previous one is no longer in view, it
    # it should change to the next one
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_Matrix, distCoeffs)
        marker_id, tvec = get_closest_marker(ids, tvecs)
        for i, mid in enumerate(ids.flatten()):
            print(f"[MARKER DETECTED] ID: {marker_id}, DIstance (z): {tvecs[i][0][2]:.2f}mm")
        

        if marker_id is None:
            continue
        
        x_offset = tvec[0]  # left/right
        y_offset = tvec[1]  # up/down
        z_offset = tvec[2]  # forawrd/back

        #update_head_position(x_offset, y_offset) #changes happen in this function to clean up code
        # changed from tvecs to tvec since it updates in line 159
        cv2.drawFrameAxes(frame, camera_Matrix, distCoeffs, rvecs, tvec, 50)
    #==================================================================

     # will rotate in right direction until it finds the desired ArUco
        if (label_dict[label] != marker_id):
            if (marker_id==1 & label_dict[label] == 2 | marker_id==2 & label_dict[label] == 3 | 
                marker_id==3 & label_dict[label] == 4 | marker_id==4 & label_dict[label] == 1):
                servos_position[wheel_1] = 6700
                servos_position[wheel_2] = 5300

            else:
                servos_position[wheel_2] = 6700
                servos_position[wheel_1] = 5300
                acquired = 0
    # but it won't shut off until the x offset of the desired aruco is within reason
        elif ( label_dict[label] == marker_id &x_offset < 5):
            servos_position[wheel_1] = 6000
            servos_position[wheel_2] = 6000
            if acquired ==0:
                distance = z_offset
            acquired = 1
            
        # Camera movement happening ~ simultaneously =======
            # removed the head rotation for simplicity
            # The y offste is so it can always see it's distance away

        if y_offset != center_y:
            if y_offset > center_y:
                if servos_position[htilt] <= 3000:
                    servos_position[htilt]=3000
                else:
                    servos_position[htilt]=servos_position[htilt]-20
                    Robot.setTarget(htilt, servos_position[htilt])  
            
                
            if y_offset < center_y:
                if servos_position[htilt] >= 9000:
                    servos_position[htilt]=9000
               
                else:
                    servos_position[htilt]=servos_position[htilt]+20
                    Robot.setTarget(htilt, servos_position[htilt])
            
    # =================================================================
    #Changes based on distance away ===================================
        if(delivered == 0):
            if(z_offset > 560) & acquired ==1 : # Can be changed to less, but causes weird
            # Forward movement until desired distance away
                forward(z_offset)
                print("Forward")
                delivered = 1
    # TODO return to center
            else:
                stop()
                arm_drop()
                stop()
                time.sleep(3)
                delivered =1
                fix =0
        else:
            if(z_offset < distance): # Can be changed to less, but causes weird
            # Forward movement until desired distance away
                backward(z_offset)
                print("Backwards")
                delivered = 1
                
            else:
                stop()
                arm_drop()
                stop()
                time.sleep(3)
                delivered =1
                fix =0
#======================================================================

            if current_ID !=  marker_id:
                previous_ID = current_ID
                current_ID = marker_id

cv2.destroyAllWindows()
stop()      #Stop added at end to reset robot
