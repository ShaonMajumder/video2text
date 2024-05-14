# Video 2 Text
## Description
video2pdfslides.py is a Python script that converts a video into a PDF document containing slides extracted from the video frames. The script allows users to select regions of interest (ROI) within the video frames to capture specific content for each slide.

## Features
- Converts video to PDF slides.
- Allows selection of ROI within video frames.
- Extracts slides based on selected ROI.
- Supports various video formats.

## Tasks
- Done: Adding ROI
- Done: Adding Video Frame
- Image 2 Text
- Virtual Env
- Dockerize
- Adding Edge detection to ROI


## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/video2pdfslides.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the script with the following command:

```bash
python video2pdfslides.py [video_path]
```
Replace [video_path] with the path to the input video file.

Example:

```bash
python video2pdfslides.py input_video.mp4
```

## How It Works
The script loads the input video file.
It displays a GUI window allowing users to play the video, select regions of interest (ROI), and finalize the ROI selection.
After finalizing the ROI selection, the script extracts slides from the video frames based on the selected ROI.
Finally, it generates a PDF document containing the extracted slides.

## License
This project is licensed under the MIT License.

## Author
Shaon Majumder - smazoomder@gmail.com
Advance AI and UI
Initials - https://github.com/kaushikj/video2pdf