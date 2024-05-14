import os
import time
import cv2
import imutils
import shutil
import img2pdf
import glob
import argparse
import numpy as np
from roi import ROISelectorGUI
############# Define constants

OUTPUT_SLIDES_DIR = f"./output"

FRAME_RATE = 3                   # no.of frames per second that needs to be processed, fewer the count faster the speed
WARMUP = FRAME_RATE              # initial number of frames to be skipped
FGBG_HISTORY = FRAME_RATE * 15   # no.of frames in background object
VAR_THRESHOLD = 16               # Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
DETECT_SHADOWS = False            # If true, the algorithm will detect shadows and mark them.
MIN_PERCENT = 0.1                # min % of diff between foreground and background to detect if motion has stopped
MAX_PERCENT = 3                  # max % of diff between foreground and background to detect if frame is still in motion


def get_frames(video_path):
    '''A fucntion to return the frames from a video located at video_path
    this function skips frames as defined in FRAME_RATE'''
    
    
    # open a pointer to the video file initialize the width and height of the frame
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'unable to open file {video_path}')


    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_time = 0
    frame_count = 0
    print("total_frames: ", total_frames)
    print("FRAME_RATE", FRAME_RATE)

    # loop over the frames of the video
    while True:
        # grab a frame from the video

        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)    # move frame to a timestamp
        frame_time += 1/FRAME_RATE

        (_, frame) = vs.read()
        # if the frame is None, then we have reached the end of the video file
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()
 
def detect_unique_screenshots(video_path, output_folder_screenshot_path, roi):
    # Initialize fgbg a Background object with Parameters
    # history = The number of frames history that affects the background subtractor
    # varThreshold = Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
    # detectShadows = If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.

    fgbg = cv2.createBackgroundSubtractorMOG2(history=FRAME_RATE * 15, varThreshold=16, detectShadows=False)

    captured = False
    start_time = time.time()
    (W, H) = (None, None)

    screenshoots_count = 0
    for frame_count, frame_time, frame in get_frames(video_path):
        orig = frame.copy()  # clone the original frame (so we can save it later)
        frame = imutils.resize(frame, width=600)  # resize the frame

        # Apply ROI
        if roi:
            x0, y0, x1, y1 = roi
            print(f"ROI coordinates: ({x0}, {y0}), ({x1}, {y1})")
            print("Frame shape:", frame.shape)
            frame_roi = frame[y0:y1, x0:x1]
            print("Cropped frame shape:", frame_roi.shape)
        else:
            frame_roi = frame

        print( apply_edge_detection(frame) )

        mask = fgbg.apply(frame_roi)  # apply the background subtractor

        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # compute the percentage of the mask that is "foreground"
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # if p_diff less than N% then motion has stopped, thus capture the frame
        if p_diff < MIN_PERCENT and not captured:
            captured = True
            filename = f"{screenshoots_count:03}_{round(frame_time / 60, 2)}.png"

            path = os.path.join(output_folder_screenshot_path, filename)
            print("saving {}".format(path))
            cv2.imwrite(path, frame_roi)
            screenshoots_count += 1

        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p_diff >= MAX_PERCENT:
            captured = False
    print(f'{screenshoots_count} screenshots Captured!')
    print(f'Time taken {time.time() - start_time}s')
    return



def initialize_output_folder(video_path):
    '''Clean the output folder if already exists'''
    output_folder_screenshot_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}"

    if os.path.exists(output_folder_screenshot_path):
        shutil.rmtree(output_folder_screenshot_path)

    os.makedirs(output_folder_screenshot_path, exist_ok=True)
    print('initialized output folder', output_folder_screenshot_path)
    return output_folder_screenshot_path


def convert_screenshots_to_pdf(output_folder_screenshot_path):
    output_pdf_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}" + '.pdf'
    print('output_folder_screenshot_path', output_folder_screenshot_path)
    print('output_pdf_path', output_pdf_path)
    print('converting images to pdf..')
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_folder_screenshot_path}/*.png"))))
    print('Pdf Created!')
    print('pdf saved at', output_pdf_path)

def apply_edge_detection(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    # Apply contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_contour_area = 100  # Adjust this threshold as needed
    max_contour_area = 10000  # Adjust this threshold as needed
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            filtered_contours.append(contour)
    
    # Check if any contours intersect the entire frame
    frame_height, frame_width = gray.shape
    frame_region = np.array([[0, 0], [0, frame_height], [frame_width, frame_height], [frame_width, 0]])
    frame_region_mask = np.zeros_like(gray)
    cv2.fillPoly(frame_region_mask, [frame_region], 255)
    for contour in filtered_contours:
        intersection = cv2.bitwise_and(frame_region_mask, cv2.drawContours(np.zeros_like(gray), [contour], -1, 255, thickness=cv2.FILLED))
        if cv2.countNonZero(intersection) > 0:
            print("Text is cut off in this frame!")
            # You can return a boolean indicating whether text is cut off or perform further actions here
    
    return edges

if __name__ == "__main__":
    
#     video_path = "./input/Test Video 2.mp4"
#     choice = 'y'
#     output_folder_screenshot_path = initialize_output_folder(video_path)
    
    
    parser = argparse.ArgumentParser("video_path")
    parser.add_argument("video_path", help="path of video to be converted to pdf slides", type=str)
    args = parser.parse_args()
    video_path = args.video_path

    start_frame = 0  # Define your start frame here
    end_frame = 1000  # Define your end frame here

    roi_selector_gui = ROISelectorGUI(video_path)
    roi_selector_gui.run()
    roi_shape = roi_selector_gui.roi

    print('video_path', video_path)
    output_folder_screenshot_path = initialize_output_folder(video_path)
    detect_unique_screenshots(video_path, output_folder_screenshot_path, roi_shape)

    print('Please Manually verify screenshots and delete duplicates')
    while True:
        choice = input("Press y to continue and n to terminate")
        choice = choice.lower().strip()
        if choice in ['y', 'n']:
            break
        else:
            print('please enter a valid choice')

    if choice == 'y':
        convert_screenshots_to_pdf(output_folder_screenshot_path)