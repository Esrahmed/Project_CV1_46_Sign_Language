
import pickle
import cv2  # Assuming you have OpenCV installed
import numpy as np
import json  # Assuming you have the json library installed
import os

import cv2

import numpy as np

from matplotlib import pyplot as plt
import time
import mediapipe as mp




data_path = os.path.join("F:", "cv_datast", "WLASL_v0.3.json")

video_folder = os.path.join("F:", "cv_datast", "videos")

data_pickle_path=os.path.join("F:", "cv_datast", "data.pickle")
# Load data and labels
data = []
labels = []


def load_data():
  """
  Loads the WLASL dataset glossary and video information.

  
  """
  # Load glossary data (assuming it's in a JSON file)
  
  
  with open(f"{data_path}", "r") as f:
    glossary_data = json.load(f)  # Assuming you have the json library installed
  
  glossary = {}
  for entry in glossary_data:
    
    gloss = entry["gloss"]
    instances = entry["instances"]
    
    for dataItem in instances:
      
       video_id = dataItem["video_id"]
       glossary[video_id] = gloss
      
    

 
  return glossary

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities




########################################################################################




def mediapipe_detection(image, model):
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False
    if(image is None): 
       print("fi") 

    else:                # Image is no longer writeable
     results = model.process(image)

                      # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results




#########################################################################################




def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections






##############################################################################################





def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
        




###################################################################################################




def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])










###########################################################################################################


def preprocess_video(video_path,word,number,video_counter,last_word, words_counter, frame_size=(224, 224)):
  
  if(words_counter==100): return 0 ,video_counter,last_word, words_counter
  if(video_counter==6 and last_word==word): return 1 , video_counter,last_word, words_counter

  
  frames = []
           
  cap = cv2.VideoCapture(video_path)
  if(cap.isOpened()):
     print("words_counter:",words_counter)
     print("video_counter: ",video_counter)
     
    
     print(word,"\n")
     
    
     
     if(last_word!=word): 
        last_word=word
        words_counter=words_counter+1
        video_counter=1

     else: 
         video_counter=video_counter+1

      

    
  fl=0
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        

        # Read feed
        ret, frame = cap.read()

        # Make detections
        if(ret): image, results = mediapipe_detection(frame, holistic)
        # print(results)
        else: break
        # Draw landmarks
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)
       

        data.append(keypoints)
        
        labels.append(word)
        

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
       
    cap.release()
    cv2.destroyAllWindows()  
    



  return 1 ,video_counter,last_word, words_counter





##########################################################################################


from keyboard import is_pressed
video_counter=0
last_word=""
words_counter=0
number=0
glossary= load_data()  # Assuming separate info file
print(len(glossary))
for video_id, word in glossary.items():
  # print(word)
  video_pa = video_id+".mp4" # Assuming video format
  video_path=  os.path.join(video_folder,video_pa)
 
  
  fl,video_counter,last_word, words_counter= preprocess_video(video_path,word,number,video_counter,last_word,words_counter)
    
  if(fl==0): break
  if is_pressed('e'):  # Check for 'e' key press
    break  # Exit the loop
  

# Combine data and labels into a dictionary
data_dict = {"data": data, "labels": labels}

# Save the data dictionary to a pickle file
with open(data_pickle_path, "wb") as f:
  pickle.dump(data_dict, f)

print("Data and labels saved to data.pickle")