import cv2   #include opencv library functions in python
import pytesseract
from pytesseract import Output

url = 0 # set url=0 for default cam on pc
#Create an object to hold reference to camera video capturing
cap = cv2.VideoCapture(url)

#check if connection with camera is successfully
if cap.isOpened():
        
    # continue to display window until 'q' is pressed
    while (True):
        ret, frame = cap.read()  #capture a frame from live video

        #check whether frame is successfully captured
        if ret:
            # Converting into gray frame
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # Extracting the data from frame in form of dictionary
            frame_data = pytesseract.image_to_data(gray_frame,output_type=Output.DICT)

            # Setting the coordinates of the text scanned
            for i in range(len(frame_data['text'])):

                # x-> corrdinate from left, y-> top coordinate, 
                # w-> width of the text, h-> height of the text
                x = frame_data['left'][i]
                y = frame_data['top'][i]
                w = frame_data['width'][i]
                h = frame_data['height'][i]

                accuracy = frame_data['conf'][i]

                # Showing data only if accuracy is more than 20% 
                # You can also change the accuracy but it highly depends upon the quality of 
                # scanned frame and the data
                if int(accuracy) > 10:
                    # Setting the text 
                    text = frame_data['text'][i]
                    text = "".join([c for c in text]).strip()
        
                    # Placing the text on the frame   
                    cv2.putText(frame,text,(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)

            # Showing the frame
            cv2.imshow("Text Frame",frame)
            cv2.waitKey(1)
           
            # # Press 'q' to break out of the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            if cv2.getWindowProperty("Text Frame", cv2.WND_PROP_VISIBLE) < 1:
                break

        #print error if frame capturing was unsuccessful
        else:
            print("Error : Failed to capture frame")

# print error if the connection with camera is unsuccessful
else:
    print("Cannot open camera")

# Releasing the frame and closing frame window
cap.release()
cv2.destroyAllWindows()
    