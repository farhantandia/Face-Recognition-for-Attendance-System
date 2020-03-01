# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle
from datetime import datetime
from time import gmtime, strftime, localtime
import pandas as pd
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import csv
from collections import defaultdict

cleaner = pd.read_csv('attendance-system.csv') 
cleaner.drop(cleaner.index, inplace=True)
cleaner.to_csv('attendance-system.csv', index=False)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
name_list=[]
proba_list=[]
proba = 0
count=0
now = datetime.now()
dictionaryin={}
dictionaryout={}



# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    dt_string = now.strftime("%d/%m/%Y")
    hr_string = strftime("%H:%M:%S", localtime())
    
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            cv2.putText(frame, "Sign In Status", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 2)
            
            cv2.putText(frame, "Sign Out Status", (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 2)            
            
            
            countitem=0
            for item in le.classes_:
                coordsy1=50+countitem*30
                countitem=countitem+1
                if item != 'unknown':
                    if item in dictionaryin.keys():
                        cv2.putText(frame,str(item), (10, coordsy1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        #os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 500))
                    else:
                        cv2.putText(frame,str(item), (10, coordsy1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)
            
            countitem2=0
            for item2 in dictionaryin.keys():
                coordsy2=300+countitem2*30
                countitem2=countitem2+1
                if item2 != 'unknown':
                    if item2 in dictionaryout.keys():
                        cv2.putText(frame,str(item2), (10, coordsy2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        #os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 500))
                    else:
                        cv2.putText(frame,str(item2), (10, coordsy2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)                

            
            #print(le.classes_)
    
    
    
    if proba >=0.70:        
        name_list.append(name)
        proba_list.append(proba)
        count=count+1
    
    if count==20:
    
        d = defaultdict(list)
        for key, value in zip(name_list, proba_list):
            d[key].append(value)
        occurence=dict(d)
        thisset=set(occurence)
        for x in thisset:
            occurance_individual=len(occurence[x])
            occurence[x]=sum(item for item in occurence[x])
        
        a=sum(occurence.values())

        for x in thisset:
            occurence[x]=occurence[x]/a
            
        attendance = {word for word, prob in occurence.items() if prob >= 0.3}
        #students = max(occurence, key=occurence.get)
        students = list(attendance)
                
        
        headers = ['Date','Name', 'Time Sign In','Time Sign Out']
        def write_csv(data):
            
            with open('attendance-system.csv', 'a') as outfile:
            
                outfile.truncate()
                file_is_empty = os.stat('attendance-system.csv').st_size == 0
                writer = csv.writer(outfile, lineterminator='\n',)
                if file_is_empty:
                    writer.writerow(headers)
                
                writer.writerow(data)
                
        #time.sleep(1)
        current_hour = datetime.now().second
        fps.stop()
        waktu=fps.elapsed()

        if waktu >= 0 and waktu <= 15 :
            print('Attendance system Open for sign in')
            for a in students:
                write_csv([dt_string,a,hr_string,''])
            
            records = pd.read_csv('attendance-system.csv') #Records dictionaryin for notification
            deduped = records.drop_duplicates(['Name'], keep='first')
            deduped =deduped.drop(columns=['Time Sign Out'])
            dictionaryin=deduped.set_index('Name').T.to_dict('list')
        
        elif waktu >=30 and waktu <=45:
            
            for a in students:
                write_csv([dt_string,a,'',hr_string])
            print('Attendance system Open for sign out')
            
            records = pd.read_csv('attendance-system.csv') #Records dictionaryout for notification
            signed_out=records.loc[records['Time Sign Out'].notna()]
            deduped_out = signed_out.drop_duplicates(['Name'], keep='first')
            deduped_out =deduped_out.drop(columns=['Time Sign In'])
            dictionaryout=deduped_out.set_index('Name').T.to_dict('list')
        else:
            print('Attendance system close until Next Course')

        print(dt_string,hr_string, students)
        


        name_list.clear()
        proba_list.clear()
        count=0
        
        
        
        
    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()

records = pd.read_csv('attendance-system.csv') 
deduped = records.drop_duplicates(['Name'], keep='first')
deduped = deduped.drop(columns=['Time Sign Out'])

signed_out=records.loc[records['Time Sign Out'].notna()]
deduped_out = signed_out.drop_duplicates(['Name'], keep='first')
deduped_out =deduped_out.drop(columns=['Time Sign In'])

mergedStuff = pd.merge(deduped, deduped_out, on=['Name'],suffixes=(' Sign In', ' Sign Out'))
attend_data = mergedStuff[mergedStuff.Name != 'unknown']
attend_data.to_csv('attendance-data.csv', index=False)

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
