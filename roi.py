import tkinter as tk
from PIL import Image, ImageTk
import cv2

class ROISelectorGUI:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.selection_duration = 5  # Selection duration in seconds

        self.root = tk.Tk()
        self.root.title("ROI Selector")

        ###
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tk.Canvas(self.root, width=self.frame_width, height=self.frame_height)

        # self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.pack()

        self.scale_length = min(300, self.frame_count)  # Set maximum length of 300 pixels for the slider
        self.scale = tk.Scale(self.root, from_=0, to=self.frame_count - 1, orient=tk.HORIZONTAL, command=self.update_frame,
                              length=self.scale_length)
        self.scale.pack()

        self.select_button = tk.Button(self.root, text="Select ROI", command=self.select_roi)
        self.select_button.pack()

        self.finalize_button = tk.Button(self.root, text="Finalize ROI", command=self.finalize_roi, state=tk.DISABLED)
        self.finalize_button.pack()

        self.roi = None

        self.update_frame(0)

    def update_frame(self, frame_num):
        self.current_frame = int(frame_num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

    def select_roi(self):
        self.roi_selector = ROISelector(self.canvas, self.finalize_button)

    def finalize_roi(self):
        self.roi = self.roi_selector.get_roi()
        roi_width = self.roi[2] - self.roi[0]
        roi_height = self.roi[3] - self.roi[1]
        print("Finalized ROI: Shape =", (roi_width, roi_height))
        self.root.destroy()  # Destroy the root window
        return (roi_width, roi_height)


    def run(self):
        self.root.mainloop()

class ROISelector:
    def __init__(self, canvas, finalize_button):
        self.canvas = canvas
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect = None
        self.finalize_button = finalize_button
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

    def on_release(self, event):
        self.finalize_button.config(state=tk.NORMAL)

    def get_roi(self):
        x0 = min(self.start_x, self.end_x)
        y0 = min(self.start_y, self.end_y)
        x1 = max(self.start_x, self.end_x)
        y1 = max(self.start_y, self.end_y)
        return (x0, y0, x1, y1)