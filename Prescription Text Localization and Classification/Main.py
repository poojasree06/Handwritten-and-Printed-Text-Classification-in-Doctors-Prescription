from tkinter import *
import tkinter as tk
import os
import hashlib
from tkinter import filedialog
import cv2
import numpy as np
from joblib import load
global hubba


def tester(clf,mpred):
	clf = load("data.joblib")
	dj = clf.predict(mpred)
	print(dj[0])
	return dj[0]

def checker(img):	
	arr=[]
	dj=[]
	rows = img.shape[0]
	cols = img.shape[1]
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	arr.append(rows)
	arr.append(cols)
	arr.append(rows/cols)
	retval,bwMask =cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	mycnt=0
	myavg=0
	for xx in range (0,cols):
		mycnt=0
		for yy in range (0,rows):
			if bwMask[yy,xx]==0:
				mycnt=mycnt+1
				
		myavg=myavg+(mycnt*1.0)/rows
	myavg=myavg/cols
	arr.append(myavg)
	change=0
	for xx in range (0,rows):
		mycnt=0
		for yy in range (0,cols-1):
			if bwMask[xx:yy].all()!=bwMask[xx:yy+1].all():
				mycnt=mycnt+1
		change=change+(mycnt*1.0)/cols
	change=change/(rows)
	arr.append(change)
	dj.append(arr)
	return dj

def Classify(filename,segments_directory):
    img = cv2.imread(filename)
    hgt=img.shape[0]
    wdt=img.shape[1]
    hBw=hgt/float(wdt)
    dim = (576, int(576 * hBw))
    fram = img.copy()
    img=cv2.resize(img,dim)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    linek = np.zeros((11,11),dtype=np.uint8)
    linek[5,...]=1
    x=cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek ,iterations=1)
    gray-=x
    kernel = np.ones((5,5), np.uint8)
    ret2,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gray = cv2.dilate(gray, kernel, iterations=1) 
    contours2, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    x=0
    clf = load("data.joblib")
    
    contours2 = sorted(contours2, key=lambda c: cv2.boundingRect(c)[1])  # Sort by y-coordinate, need to use custom comparator ?
    while x<len(contours2):
        (start_x,start_y,width,height)= cv2.boundingRect(contours2[x])
        segment = img[start_y:start_y+height, start_x:start_x+width]

        segment_name = f'segment_{x:02d}.jpg'
        
        # Save the segment in the output directory
        segment_path = os.path.join(segments_directory, segment_name)
        cv2.imwrite(segment_path, segment)
        x=x+1
    return img


def generate_identifier(file_content):
    # Calculate a unique 64-hexadecimal identifier for the uploaded file
    identifier = hashlib.sha256(file_content).hexdigest()[:64]
    return identifier

def upload_prescription():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if file_path:
        with open(file_path, 'rb') as file:
            identifier = generate_identifier(file.read())
            original_filename = os.path.basename(file_path)
            # Create the directory structure
            main_directory = os.path.join('DATABASE', identifier)
            os.makedirs(main_directory, exist_ok=True)
             # Create the INPUT directory
            input_directory = os.path.join(main_directory, 'INPUT')
            os.makedirs(input_directory, exist_ok=True)
            # Save the prescription file to the INPUT directory
            prescription_filename = identifier
            prescription_dest_path = os.path.join(input_directory, prescription_filename)
            with open(file_path, 'rb') as source, open(prescription_dest_path, 'wb') as dest:
                dest.write(source.read())
            
            result_label.config(text=f"'{original_filename}' has been uploaded with identifier '{identifier}'into the database.")

            # Create the SEGMENTS directory
            segments_directory = os.path.join(main_directory, 'SEGMENTS')
            os.makedirs(segments_directory, exist_ok=True)
            
            saveImage=Classify(prescription_dest_path,segments_directory)
            
            
            segment_dest_path = os.path.join(segments_directory, "temp.png")
            cv2.imwrite(segment_dest_path,saveImage)




# Create the main application window
root = tk.Tk()
root.title("Prescription Organizer")

# Create and configure a label
info_label = tk.Label(root, text="Upload a prescription image:")
info_label.pack()

# Create and configure a button to trigger file upload
upload_button = tk.Button(root, text="Upload", command=upload_prescription)
upload_button.pack()

# Create and configure a label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the Tkinter main loop
root.mainloop()