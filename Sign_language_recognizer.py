import cv2
from HandTrackingModule import HandDetector
from ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from PIL import ImageTk, Image

# Initialize variables
cap = cv2.VideoCapture(0)  # Camera ID == 0
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z"]

classifier = Classifier("keras_model.h5", "labels.txt")
use_code1 = True

# Create Tkinter window
window = tk.Tk()
window.title("American Sign Language Recognizer")

# Tkinter icon
window.iconbitmap("logo.ico")

# Create a label widget to display the OpenCV output
label = tk.Label(window)
label.pack()

# Define global variable for chart window
current_chart_window = None

# Define switch code function
def switch_to_code1():
    global use_code1
    use_code1 = True
    code_label.config(text="Sign Language: American Sign Language", font=("Arial", 14), fg="white", bg="black")

# Define show_chart function
def show_chart(chart_path):
    global current_chart_window

    # Close the current chart window if it exists
    if current_chart_window is not None:
        current_chart_window.destroy()

    chart = Image.open(chart_path)
    chart = chart.resize((400, 400), Image.LANCZOS)
    chartTk = ImageTk.PhotoImage(chart)

    # Create a new Tkinter window
    chart_window = tk.Toplevel(window)
    chart_window.title("Sign Language Charts")
    
    # Create a label widget to display the chart
    chart_label = tk.Label(chart_window, image=chartTk)
    chart_label.pack()

    # Update the current chart window
    current_chart_window = chart_window

    # Run the Tkinter event loop for the chart window
    chart_window.mainloop()

# Create a frame for the buttons
frame = tk.Frame(window)
frame.pack()

# Create switch code button for ASL
switch_button1 = tk.Button(frame, text="American Sign Language", command=switch_to_code1, width=30, height=2)
switch_button1.pack(side=tk.LEFT)

# Create chart button
chart_button1 = tk.Button(frame, text="ASL Chart", command=lambda: show_chart("ASL_CHART.png"), width=10, height=2)
chart_button1.pack(side=tk.LEFT)

# Create code label
code_label = tk.Label(window, text="Sign Language: American Sign Language", font=("Arial", 14), fg="white", bg="black")
code_label.pack()

# Define video loop function
def video_loop():
    global use_code1
    success, img = cap.read()
    imgOutput = img.copy()

    if use_code1:
        # Process image for ASL detection
        hands, img = detector.findHands(img)
        try:
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgWhite[:, (imgSize - wCal) // 2:(imgSize - wCal) // 2 + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgWhite[(imgSize - hCal) // 2:(imgSize - hCal) // 2 + hCal, :] = imgResize
                    
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        except Exception as e:
            print(f"Error: {e}")

    # Convert the OpenCV image to a PIL image
    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgOutput)

    # Convert the PIL image to a Tkinter image
    imgTk = ImageTk.PhotoImage(image=imgPIL)

    # Update the label with the new image
    label.config(image=imgTk)
    label.image = imgTk

    # Call the video_loop function after 1ms
    window.after(1, video_loop)

# Start the video loop
video_loop()

# Start the Tkinter event loop
window.mainloop()
