import cv2
import sys

# Path for image to be detected and classifier path
# imagePath = "/home/mayoogh/Downloads/photo2.jpg"
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
# cascPath = "haarcascade_eye.xml"

# Creating haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image and convert it to gray
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,               # Matrix where image are stored.
    scaleFactor=1.1,    # This specify how much the image size is reduced.
    minNeighbors=5,     # Minimum distance between nearby detected faces
    minSize=(30, 30),   # Minimum size of object (face) to be detected, here face smaller than this are ignored.
)

print("Detected {0} faces from the image.".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
