import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to generate histograms
def get_histogram(src):

    bgr_planes = cv2.split(src)  # Split a 3-channel image into 3 single-channel images
    histSize = 1024
    histRange = (0, histSize) # The upper boundary is exclusive

    # Calculate histogram
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange)  # Calculate histogram for the Blue channel
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange)  # Calculate histogram for the Green channel
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange)  # Calculate histogram for the Red channel

    # Normalization due to the distance of the photo in the image. Example: The histogram calculation of the head to the body can change due to the distance of the taken photo
    b = b_hist / sum(b_hist)  # Normalize Blue channel
    r = r_hist / sum(r_hist)  # Normalize Red channel
    g = g_hist / sum(g_hist)  # Normalize Green channel

    # Histogram
    histogram = np.array([b, g, r]).reshape(-1, 1)  # Concatenate the 3 arrays into a single array

    return histogram, b, g, r

# Histogram of the original team images
mhot = cv2.imread('mhot.jpg')  # Load image
hmh, b1, g1, r1 = get_histogram(mhot)
# ax[0][0].set_title("More Hot")
# ax[0][0].plot(hmh)  # Plot the histogram

hot = cv2.imread('hot.jpg')  # Load image
hho, b1, g1, r1 = get_histogram(hot)
# ax[0][1].set_title("Hot")
# ax[0][1].plot(ht)  # Plot the histogram

mid = cv2.imread('mid.jpg')  # Load image
hmd, b1, g1, r1 = get_histogram(mid)
# ax[0][2].set_title("Mid")
# ax[0][2].plot(hmd)  # Plot the histogram

mlower = cv2.imread('mlower.jpg')  # Load image
hml, b1, g1, r1 = get_histogram(mlower)
# ax[1][3].set_title("Lower")
# ax[1][3].plot(hml)  # Plot the histogram


def get_tracker():
    tracker = cv2.TrackerCSRT_create()
    return tracker

cap = cv2.VideoCapture("teste1.mp4")  # Create the video capture object

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Create the video writer object

_, frame = cap.read()  # Capture the first frame
bb = []  # Create an empty list to store the coordinates of the bounding boxes

out = cv2.VideoWriter('teste_out.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))  # Set the output file name, FPS, and resolution

while True:
    roi = cv2.selectROI('Frame', frame)  # Function to select ROI
    # print(roi)
    bb.append(roi)  # Add the coordinates of the ROI box to the list

    k = cv2.waitKey(0)
    if k == ord('q'):
        break

multiTracker = cv2.MultiTracker_create()  # Create the MultiTracker object

for bbox in bb:
    multiTracker.add(get_tracker(), frame, bbox)  # Initialize the Tracker object for each selected ROI

while True:
    old_frame = frame

    ret, frame = cap.read()  # Capture a frame
    if not ret:  # Check the video status
        exit()

    _, bxs = multiTracker.update(frame)  # Update the Tracker object for the new position of each selected ROI

    for ID, box in enumerate(bxs):
        p1 = (int(box[0]), int(box[1]))  # Coordinates of the detection boxes
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

        x = int(box[1])
        y = int(box[0])
        a = int(box[2])
        b = int(box[3])
        cortada = frame[x:x+b, y:y+a]

        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)  # Draw rectangles on the detected areas

        ho, bo, go, ro = get_histogram(cortada)  # Get the histogram of the cropped image
        mh = cv2.compareHist(ho, hmh, 0)
        ht = cv2.compareHist(ho, hho, 0)
        md = cv2.compareHist(ho, hmd, 0)
        ml = cv2.compareHist(ho, hml, 0)

        if mh > ht and mh > md and mh > ml:
            # print ("Very Hot")
            cv2.putText(frame, "Very Hot", (int(box[0]-8), int(box[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)  # Text with the ID of each object
        elif ht > mh and ht > md and ht > ml:
            # print("Hot")
            cv2.putText(frame, "Hot", (int(box[0]-8), int(box[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # Text with the ID of each object
        elif md > mh and md > ht and md > ml:
            # print("Cold")
            cv2.putText(frame, "Cold", (int(box[0]-5), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)  # Text with the ID of each object
        elif ml > mh and ml > ht and ml > md:
            # print("Very Cold")
            cv2.putText(frame, "Very Cold", (int(box[0]-5), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)  # Text with the ID of each object

    cv2.imshow('Frame', frame)
    out.write(frame)

    k = cv2.waitKey(15)
    if k == ord('q'):
        out.release()  # Without this, the video does not work
        exit()

out.release()
cv2.destroyAllWindows()
