import cv2 

#the program is based upon the cascade classifier method using the default haar casacades nad identifier that comes with opencv
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    # cap the video fram by frame
    ret ,frame = cap.read()
    # coverting the image to grey scale 
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectMultiScale is a function that takes multiple arguments:
    # 1.an image array: in the program the image is converted to a grey scale in the form
    #                 of an array of pixels
    # 2.scaleFactor: allows the program to dynamically resize the image to improve its
    #                 chance of detectibility by the algorithm
    # 3.minNeighbors: this cariable affects the quality of the detected faces higher value 
    #                 will lead to fewer detections, usaually a good value for it ranges from 3 ~ 6
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor = 1.5, minNeighbors = 5)
    for (x , y , w, h) in faces:
        print(x, y, w, h)
        region_of_intrest = frame[y:y+h, x:x+w]
        image_item = "my-image.png"
        cv2.imwrite(image_item, region_of_intrest)
        color1 = (0, 255, 0)
        color2 = (255, 0, 0)
        stroke = 2
        xcord = x + w
        ycord = y + h
        cv2.rectangle(frame, (x, y), (xcord, ycord), color1, stroke)
        eyes = eye_cascade.detectMultiScale(gray_scale, scaleFactor = 1.1, minNeighbors = 6)
        for(ex, ey, ew, eh) in eyes:
            eye_xcord = ex + ew
            eye_ycord = ey + eh
            cv2.rectangle(frame, (ex, ey), (eye_xcord, eye_ycord), color2, stroke )
    #display the frames
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#end the program and release the video capture from the webcam
cap.release()
cv2.destroyAllWindows()