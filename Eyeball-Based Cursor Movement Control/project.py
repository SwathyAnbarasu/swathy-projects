from imutils import face_utils
import dlib
import cv2
import pyautogui as pag
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.ndimage import gaussian_filter
def MAR(point1,point2,point3,point4,point5,point6,point7,point8):
	mar = (dst(point1,point2) + dst(point3,point4) + dst(point5,point6))/(3.0*dst(point7,point8))
	return mar
def EAR(point1,point2,point3,point4,point5,point6):
	ear = (dst(point2,point6) + dst(point3,point5))/(2*dst(point1,point4))*1.0
	return ear

def dst(point1, point2):
	distance = np.sqrt((point1[0] - point2[0])*2 + (point1[1] - point2[1])*2)
	return distance
def angle(point1):
	slope12 = (point1[1] - 250)/(point1[0] - 250)*1.0
	agle = 1.0*np.arctan(slope12)
	return agle
def nothing(x):
	pass
lclick = np.array([])
rclick = np.array([])
scroll = np.array([])
eyeopen = np.array([])
lclickarea = np.array([])
rclickarea = np.array([])
t1 = np.array([])
t2 = np.array([])
t3 = np.array([])
t4 = np.array([])
pag.PAUSE = 0
p = "shape_predictor.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
(lstart,lend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(rstart,rend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(mstart,mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
currenttime = time.time()
while(time.time() - currenttime <= 25):
	ret,image = cap.read()
	blackimage = np.zeros((480,640,3),dtype = np.uint8)
	image = cv2.flip(image,1)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
	gray = clahe.apply(gray)
	rects = detector(gray,0)
	for (i,rect) in enumerate(rects):
		shape = predictor(gray,rect)
		shape = face_utils.shape_to_np(shape)
		lefteye = EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		righteye = EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		mar = MAR(shape[50],shape[58],shape[51],shape[57],shape[52],shape[56],shape[48],shape[54]) 
		EARdiff = (lefteye - righteye)*100
		leftEye = shape[lstart:lend]
		rightEye = shape[rstart:rend]
		mouthroi = shape[mstart:mend]
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouthroi)
		learmar = lefteye/mar
		rearmar = righteye/mar
		cv2.drawContours(image,[mouthroi],-1,(0,255,0),1)
		cv2.drawContours(image,[leftEyeHull],-1,(0,255,0),1)
		cv2.drawContours(image,[rightEyeHull],-1,(0,255,0),1)
		marea = cv2.contourArea(mouthHull)
		larea = cv2.contourArea(leftEyeHull)
		rarea = cv2.contourArea(rightEyeHull)
		LAR = larea/marea
		RAR = rarea/marea
		print(learmar,rearmar)
		elapsedTime = time.time() - currenttime
		if elapsedTime < 5.0:
			cv2.putText(blackimage,'Keep Both Eyes Open',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			eyeopen = np.append(eyeopen,[EARdiff])
			t1 = np.append(t1,[elapsedTime])
		elif elapsedTime > 5.0 and elapsedTime < 10.0:
			cv2.putText(blackimage,'Close Left Eye',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			if elapsedTime > 7.0 and elapsedTime < 10.0:
				lclick = np.append(lclick,[EARdiff])
				lclickarea = np.append(lclickarea,[larea])
				t2 = np.append(t2,[elapsedTime])
		elif elapsedTime > 12.0 and elapsedTime < 17.0:
			cv2.putText(blackimage,'Open Left eye and close Right Eye',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			if elapsedTime > 14.0 and elapsedTime < 17.0:
				rclick = np.append(rclick,[EARdiff])
				rclickarea = np.append(rclickarea,[rarea])
				t3 = np.append(t3,[elapsedTime])
		elif elapsedTime > 19.0 and elapsedTime < 24.0:
			cv2.putText(blackimage,'Open Your Mouth',(0,100),font,1,(255,255,255),2,cv2.LINE_AA)
			if elapsedTime > 21.0 and elapsedTime < 24.0:
				scroll = np.append(scroll,[mar])
				t4 = np.append(t4,[elapsedTime])
		else:
			pass 
		for (x,y) in shape:
			cv2.circle(image,(x,y),2,(0,255,0),-1)
		res = np.vstack((image,blackimage))
		cv2.imshow('Calibration',res)
	if cv2.waitKey(5) & 0xff == 27: 
		break
plt.subplot(2,2,1)
eyeopen_smooth = gaussian_filter(eyeopen,sigma = 5)
plt.title('Both Eyes Open')
plt.plot(t1,eyeopen_smooth)

plt.subplot(2,2,2)
lclick_smooth = gaussian_filter(lclick,sigma = 5)
plt.title('Left click')
plt.plot(t2,lclick_smooth)
plt.subplot(2,2,3)
rclick_smooth = gaussian_filter(rclick,sigma = 5)
plt.title('Right click')
plt.plot(t3,rclick_smooth)
plt.subplot(2,2,4)
scroll_smooth = gaussian_filter(scroll,sigma = 5)
plt.title('Scroll Mode')
plt.plot(t4,scroll_smooth)
plt.show()
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
MARlist = np.array([])
scroll_status = 0
eyeopen = np.sort(eyeopen)
lclick = np.sort(lclick)
rclick = np.sort(rclick)
scroll = np.sort(scroll)
lclickarea = np.sort(lclickarea)
rclickarea = np.sort(rclickarea)
openeyes = np.median(eyeopen)
leftclick = np.median(lclick) - 1
rightclick = np.median(rclick) + 1
scrolling = np.median(scroll)
leftclickarea = np.median(lclickarea)
rightclickarea = np.median(rclickarea)
print("Standard Deviation = "+str(np.std(lclick)))
print("Standard Deviation = "+str(np.std(rclick)))
print("Left click value = "+str(leftclick))
print("Right click value = "+str(rightclick))
ll = 0
while(True):
	try: 
		frameTimeInitial = time.time()
		blackimage = np.zeros((480,640,3),dtype = np.uint8)

		_, image = cap.read() 
		image=cv2.flip(image,1)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
		gray = clahe.apply(gray)
		cv2.circle(image,(250,250),50,(0,0,255),2)

		_,image2 = cap.read()
		image2 = cv2.flip(image2,1)
		image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

		rects = detector(gray, 0)

		for (i, rect) in enumerate(rects):

			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			[h,k] = shape[33]
			lefteye = EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
			righteye = EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
			EARdiff = (lefteye - righteye)*100
			mar = MAR(shape[50],shape[58],shape[51],shape[57],shape[52],shape[56],shape[48],shape[54])

			leftEye = shape[lstart:lend]
			rightEye = shape[rstart:rend]
			mouthroi = shape[mstart:mend]
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			mouthHull = cv2.convexHull(mouthroi)
			cv2.drawContours(image,[mouthroi],-1,(0,255,0),1) 
			cv2.drawContours(image,[leftEyeHull],-1,(0,255,0),1)
			cv2.drawContours(image,[rightEyeHull],-1,(0,255,0),1)
			larea = cv2.contourArea(leftEyeHull)
			rarea = cv2.contourArea(rightEyeHull)
			marea = cv2.contourArea(mouthHull)
			if EARdiff < leftclick and larea < leftclickarea:
				pag.click(button = 'left') 
				cv2.putText(blackimage,"Left Click",(0,300),font,1,(255,255,255),2,cv2.LINE_AA)
				lclick = np.array([])
			elif EARdiff > rightclick and rarea < rightclickarea: # Right click will be initiated if the EARdiff is more than the rightclick calculated during calibration 
				pag.click(button = 'right') 
				cv2.putText(blackimage,"Right Click",(0,300),font,1,(255,255,255),2,cv2.LINE_AA)
				lclick = np.array([])

		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
		MARlist = np.append(MARlist,[mar])
		if len(MARlist) == 30:
			mar_avg = np.mean(MARlist)
			MARlist = np.array([])
			if int(mar_avg*100) > int(scrolling*100):
				if scroll_status == 0:
					scroll_status = 1
				else:
					scroll_status = 0

		if scroll_status == 0:
			if ((h-250)*2 + (k-250)*2 - 50*2 > 0):


				a = angle(shape[33])
				if h > 250:
					time.sleep(0.03)
					pag.moveTo(pag.position()[0]+(10*np.cos(1.0*a)),pag.position()[1]+(10*np.sin(1.0*a)),duration = 0.01)
					cv2.putText(blackimage,"Moving",(0,250),font,1,(255,255,255),2,cv2.LINE_AA)
				else:
					time.sleep(0.03)
					pag.moveTo(pag.position()[0]-(10*np.cos(1.0*a)),pag.position()[1]-(10*np.sin(1.0*a)),duration = 0.01)
					cv2.putText(blackimage,"Moving",(0,250),font,1,(255,255,255),2,cv2.LINE_AA)
		else:
			cv2.putText(blackimage,'Scroll mode ON',(0,100),font,1,(255,255,255),2,cv2.LINE_AA)
			if k > 300:
				cv2.putText(blackimage,"Scrolling Down",(0,300),font,1,(255,255,255),2,cv2.LINE_AA) 
				pag.scroll(-1)
			elif k < 200:
				cv2.putText(blackimage,"Scrolling Up",(0,300),font,1,(255,255,255),2,cv2.LINE_AA) 
				pag.scroll(1)
			else:
				pass
		

		frameTimeFinal = time.time()
		cv2.putText(blackimage,"FPS: "+str(int(1/(frameTimeFinal - frameTimeInitial))),(0,150),font,1,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(blackimage,"Press Esc to abort",(0,200),font,1,(255,255,255),2,cv2.LINE_AA)
		res = np.vstack((image,blackimage))
		cv2.imshow('Cursor Control',res)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	except:
		blackimage = np.zeros((480,640,3),dtype = np.uint8)
		cv2.putText(blackimage,"Landmarks Lost.\nCheck the lighting or reposition your face",(0,100),font,1,(255,255,255),2,cv2.LINE_AA)
		_,image = cap.read() 
		image = cv2.flip(image,1)
		res = np.vstack((image,blackimage))
		cv2.imshow('Cursor Control',res)
		k = cv2.waitKey(5) & 0xff
		if k == 27:
			break
cv2.destroyAllWindows()
cap.release()