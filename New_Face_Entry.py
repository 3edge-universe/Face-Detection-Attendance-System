import cv2                  # Importing the opencv
import numpy as np          # Import Numarical Python
import NameFind
import math
WHITE = [255, 255, 255]

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
glass_cas = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

#   import the Haar cascades for face ditection
def DetectEyes(Image):
    Theta = 0
    rows, cols = Image.shape
    glass = glass_cas.detectMultiScale(Image)                                               # This ditects the eyes
    for (sx, sy, sw, sh) in glass:
        if glass.shape[0] == 2:                                                             # The Image should have 2 eyes
            if glass[1][0] > glass[0][0]:
                DY = ((glass[1][1] + glass[1][3] / 2) - (glass[0][1] + glass[0][3] / 2))    # Height diffrence between the glass
                DX = ((glass[1][0] + glass[1][2] / 2) - glass[0][0] + (glass[0][2] / 2))    # Width diffrance between the glass
            else:
                DY = (-(glass[1][1] + glass[1][3] / 2) + (glass[0][1] + glass[0][3] / 2))   # Height diffrence between the glass
                DX = (-(glass[1][0] + glass[1][2] / 2) + glass[0][0] + (glass[0][2] / 2))   # Width diffrance between the glass

            if (DX != 0.0) and (DY != 0.0):                                                 # Make sure the the change happens only if there is an angle
                Theta = math.degrees(math.atan(round(float(DY) / float(DX), 2)))            # Find the Angle
                print ("Theta  " + str(Theta))

                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Theta, 1)                 # Find the Rotation Matrix
                Image = cv2.warpAffine(Image, M, (cols, rows))
                #cv2.imshow('ROTATED', Image)                                              # UNCOMMENT IF YOU WANT TO SEE THE

                Face2 = face_cascade.detectMultiScale(Image, 1.3, 5)                                # This detects a face in the image
                for (FaceX, FaceY, FaceWidth, FaceHeight) in Face2:
                    CroppedFace = Image[FaceY: FaceY + FaceHeight, FaceX: FaceX + FaceWidth]
                    return CroppedFace
			
def enter_face():			
	ID = NameFind.AddName()
	Count = 0
	cap = cv2.VideoCapture(0)                                                                           # Camera object

	while Count < 200:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                    # Convert the Camera to graySe
		if np.average(gray) > 110:                                                                      # Testing the brightness of the image
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)                                         # Detect the faces and store the positions
			for (x, y, w, h) in faces:                                                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
				FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]    # The Face is isolated and cropped
				Img = (DetectEyes(FaceImage))
				cv2.putText(gray, "FACE DETECTED: "+str(Count), (x+int(w/2), y-5), cv2.FONT_HERSHEY_DUPLEX, .4, WHITE)
				gray=cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),3)
				if Img is not None:
					frame = Img                                                                         # Show the detected faces
				else:
					frame = gray[y: y+h, x: x+w]
				cv2.imwrite("Newdataset/"+str(ID)+"/" + str(Count) + ".jpg", frame)
				cv2.waitKey(300)
				cv2.imshow("CAPTURED PHOTO", frame)                                                     # show the captured image
				Count = Count + 1
		cv2.imshow('Face Recognition System Capture Faces', gray)                                       # Show the video
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	print ('FACE CAPTURE FOR THE SUBJECT IS COMPLETE')
	cap.release()
	cv2.destroyAllWindows()

enter_face()