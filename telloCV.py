"""
tellotracker:
Allows manual operation of the drone and demo tracking mode.

Requires mplayer to record/save video.

Controls:
- tab to lift off
- WASD to move the drone
- space/shift to ascend/descent slowly
- Q/E to yaw slowly
- arrow keys to ascend, descend, or yaw quickly
- backspace to land, or P to palm-land
- enter to take a picture
- R to start recording video, R again to stop recording
  (video and photos will be saved to a timestamped file in ~/Pictures/)
- Z to toggle camera zoom state
  (zoomed-in widescreen or high FOV 4:3)
- T to toggle tracking
@author Leonie Buckley, Saksham Sinha and Jonathan Byrne
@copyright 2018 see license file for details
"""
import time
import datetime
import os
import tellopy
import numpy
import av
import cv2
from pynput import keyboard
from tracker import Tracker

#posenet
import os
import numpy as np
import sys
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
import math

import threading
import traceback

frame = None
run_recv_thread = True



def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def argmax2d(inp_3d):
    """
    Get the x,y positions of the heatmap of each part's argmax()
    """
    heatmapPositions = np.zeros(shape=(17,2))
    heatmapConf = np.zeros(shape=(17,1))
    for i in range(17):
        argmax_i =  np.unravel_index(inp_3d[:,:,i].argmax(), inp_3d[:,:,i].shape)
        max_i =  inp_3d[:,:,i].max()
        heatmapPositions[i,:] = argmax_i
        heatmapConf[i,:] = max_i
    return heatmapPositions,heatmapConf
def get_offsetVector(heatmapPositions=None,offsets=None):
    allArrays = np.zeros(shape=(17,2))
    for idx,el in enumerate(heatmapPositions):
#         print(el)
        allArrays[idx,0] = offsets[int(el[0]),int(el[1]),idx]
        allArrays[idx,1] = offsets[int(el[0]),int(el[1]),17+idx]
    return allArrays


MODEL_NAME = "pose_TFLite_model"
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
resW, resH = '952x720'.split('x')
imW, imH = int(resW), int(resH)
use_TPU = False
min_thresh = 0.7

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = width/2
input_std = width/2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
#posenet

def main():
    """ Create a tello controller and show the video feed."""
    tellotrack = TelloCV()

    # for packet in tellotrack.container.demux((tellotrack.vid_stream,)):
    #     for frame in packet.decode():
    #         start = time.time()

    #         image = tellotrack.process_frame(frame)
    #         print("image_time",time.time()-start)

    #         cv2.imshow('tello', image)
    #         _ = cv2.waitKey(1) & 0xFF

    #posenet
    try:
        threading.Thread(target=tellotrack.recv_thread).start()

        while True:
            if frame is None:
                time.sleep(0.01)
            else:
                # print("frame FOUNDD")
                image = tellotrack.process_frame(frame)
                cv2.imshow('Original', image)
                # cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                cv2.waitKey(1)
                # long delay
                # time.sleep(0.5)
                image = None

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        run_recv_thread = False
        cv2.destroyAllWindows()
    #posenet

class TelloCV(object):
    """
    TelloTracker builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self):
        self.prev_flight_data = None
        self.record = False
        self.tracking = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.speed = 50
        self.drone = tellopy.Tello()
        self.init_drone() #posenet
        self.init_controls()

        # container for processing the packets into frames
        self.container = av.open(self.drone.get_video_stream())
        self.vid_stream = self.container.streams.video[0]
        self.out_file = None
        self.out_stream = None
        self.out_name = None
        self.start_time = time.time()
        
        # tracking a color
        green_lower = (30, 50, 50)
        green_upper = (80, 255, 255)
        #red_lower = (0, 50, 50)
        # red_upper = (20, 255, 255)
        # blue_lower = (110, 50, 50)
        # upper_blue = (130, 255, 255)
        self.track_cmd = ""
        # self.tracker = Tracker(self.vid_stream.height,
        #                        self.vid_stream.width,
        #                        green_lower, green_upper) #posenet
        self.tracker = Tracker(720,
                               960,
                               green_lower, green_upper) #posenet

    #posenet
    def recv_thread(self):
        global frame
        global run_recv_thread

        print('start recv_thread()')
        # drone = tellopy.Tello()

        try:
            # self.drone.connect()
            # self.drone.wait_for_connection(60.0)
            # #posenet
            # self.drone.start_video()
            # self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
            #                     self.flight_data_handler)
            # self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
            #                     self.handle_flight_received)
            #posenet
            # container = av.open(self.drone.get_video_stream())
            frame_count = 0
            while run_recv_thread:
                for f in self.container.decode(video=0):
                    frame_count = frame_count + 1
                    # skip first 300 frames
                    if frame_count < 300:
                        continue
                    frame = f
                time.sleep(0.01)
        except Exception as ex:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print(ex)
        finally:
            self.drone.quit()
    #posenet

    def init_drone(self):
        """Connect, uneable streaming and subscribe to events"""
        # self.drone.log.set_level(2)
        self.drone.connect()
        self.drone.wait_for_connection(60.0) #posenet
        self.drone.start_video()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
                             self.handle_flight_received)


    def on_press(self, keyname):
        """handler for keyboard listener"""
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.quit()
                exit(0)
            if keyname in self.controls:
                key_handler = self.controls[keyname]
                if isinstance(key_handler, str):
                    getattr(self.drone, key_handler)(self.speed)
                else:
                    key_handler(self.speed)
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""
        self.keydown = False
        keyname = str(keyname).strip('\'')
        print('-' + keyname)
        if keyname in self.controls:
            key_handler = self.controls[keyname]
            if isinstance(key_handler, str):
                getattr(self.drone, key_handler)(0)
            else:
                key_handler(0)

    def init_controls(self):
        """Define keys and add listener"""
        self.controls = {
            'w': lambda speed: self.drone.forward(speed),#'forward',
            's': 'backward',
            'a': 'left',
            'd': 'right',
            'Key.space': 'up',
            'Key.shift': 'down',
            'Key.shift_r': 'down',
            'q': 'counter_clockwise',
            'e': 'clockwise',
            'i': lambda speed: self.drone.flip_forward(),
            'k': lambda speed: self.drone.flip_back(),
            'j': lambda speed: self.drone.flip_left(),
            'l': lambda speed: self.drone.flip_right(),
            # arrow keys for fast turns and altitude adjustments
            'Key.left': lambda speed: self.drone.counter_clockwise(speed),
            'Key.right': lambda speed: self.drone.clockwise(speed),
            'Key.up': lambda speed: self.drone.up(speed),
            'Key.down': lambda speed: self.drone.down(speed),
            'Key.tab': lambda speed: self.drone.takeoff(),
            'Key.backspace': lambda speed: self.drone.land(),
            'p': lambda speed: self.palm_land(speed),
            't': lambda speed: self.toggle_tracking(speed),
            'r': lambda speed: self.toggle_recording(speed),
            'z': lambda speed: self.toggle_zoom(speed),
            'Key.enter': lambda speed: self.take_picture(speed)
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()
        # self.key_listener.join()

    def process_frame(self, frame):
        """convert frame to cv2 image and show"""
        
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        image = cv2.cvtColor(numpy.array(
            frame.to_image()), cv2.COLOR_RGB2BGR)
        image = self.write_hud(image)
        if self.record:
            self.record_vid(frame)

        # xoff, yoff = self.tracker.track(image)
        xoff, yoff = 0,0
        xLeftWrist, yLeftWrist =0,0
        xNose, yNose =0,0
        # print("CV xoff{}, yoff {}".format(xoff, yoff))
        #posenet
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input
        
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        
        heatmapscores = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        offsets = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects

        
        # define vectorized sigmoid
        sigmoid_v = np.vectorize(sigmoid)
        # 1 sigmoid
        sigmoheatmapscores = sigmoid_v(heatmapscores)
        # 2 argmax2d
        heatmapPositions,heatmapConfidence = argmax2d(sigmoheatmapscores)
        # 3 offsetVectors
        offsetVectors = get_offsetVector(heatmapPositions,offsets)
        # 4 keypointPositions
        outputStride = 32
        keypointPositions = heatmapPositions * outputStride + offsetVectors
        # 5 draw keypoints
        for idx,el in enumerate(heatmapConfidence):
            if heatmapConfidence[idx][0] >= min_thresh:
                x = round((keypointPositions[idx][1]/width)*imW)
                y = round((keypointPositions[idx][0]/height)*imH)
                if 'right' in labels[idx]:
                    cv2.circle(image,(int(x),int(y)), 5, (0,255,0), -1)
                elif 'left' in labels[idx]:
                    cv2.circle(image,(int(x),int(y)), 5, (0,0,255), -1)
                elif 'nose' in labels[idx]:
                    xNose, yNose = int(x),int(y)
                    xoff, yoff = (x-int(960/2)),(int(720/2)-y)
                    # print("NOSE xoff{}, yoff {}".format(xoff, yoff))
                    cv2.circle(image,(int(x),int(y)), 5, (255,0,0), -1)
                if 'leftWri' in labels[idx]:
                    xLeftWrist, yLeftWrist = int(x),int(y)

        #posenet
        def draw_arrows(frame):
            """Show the direction vector output in the cv2 window"""
            #cv2.putText(frame,"Color:", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
            cv2.arrowedLine(frame, (int(960/2), int(720/2)),
                        (int(960/2 + xoff), int(720/2 - yoff)),
                        (0, 0, 255), 1)
            return frame
        # image = self.tracker.draw_arrows(image)
        image = draw_arrows(image)
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        # Draw framerate in corner of frame
        cv2.putText(image,
                'FPS: {0:.2f}'.format(frame_rate_calc),
                (imW-200,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                1,
                cv2.LINE_AA)

        distance = 150
        cmd = ""
        # print(yoff)
        # print("WRIST {}>>>> NOSE {}???? ".format(yLeftWrist,yNose),yLeftWrist>yNose)
        if self.tracking:
            # if yLeftWrist>yNose:
            #     print("RECORDING",yLeftWrist)
                # cmd = "r"
                # lambda speed: self.toggle_recording(speed)
            if xoff < -distance and xoff>-960/2:
                cmd = "counter_clockwise"
            elif xoff > distance and xoff<960/2:
                cmd = "clockwise"
            elif yoff < -distance and yoff>-720/2:
                cmd = "down"
            elif yoff > distance and yoff<720/2:
                print("UPPPPPPPPPPPPPPP",yoff)
                cmd = "up"
            else:
                if self.track_cmd is not "":
                    getattr(self.drone, self.track_cmd)(0)
                    self.track_cmd = ""

        if cmd is not self.track_cmd:
            if cmd is not "":
                print("track command:", cmd)
                getattr(self.drone, cmd)(self.speed)
                self.track_cmd = cmd

        return image

    def write_hud(self, frame):
        """Draw drone info, tracking and record on frame"""
        stats = self.prev_flight_data.split('|')
        stats.append("Tracking:" + str(self.tracking))
        if self.drone.zoom:
            stats.append("VID")
        else:
            stats.append("PIC")
        if self.record:
            diff = int(time.time() - self.start_time)
            mins, secs = divmod(diff, 60)
            stats.append("REC {:02d}:{:02d}".format(mins, secs))

        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=30)
        return frame

    def toggle_recording(self, speed):
        """Handle recording keypress, creates output stream and file"""
        if speed == 0:
            return
        self.record = not self.record

        if self.record:
            datename = [os.getenv('HOME'), datetime.datetime.now().strftime(self.date_fmt)]
            self.out_name = '{}/Pictures/tello-{}.mp4'.format(*datename)
            print("Outputting video to:", self.out_name)
            self.out_file = av.open(self.out_name, 'w')
            self.start_time = time.time()
            self.out_stream = self.out_file.add_stream(
                'mpeg4', self.vid_stream.rate)
            self.out_stream.pix_fmt = 'yuv420p'
            self.out_stream.width = self.vid_stream.width
            self.out_stream.height = self.vid_stream.height

        if not self.record:
            print("Video saved to ", self.out_name)
            self.out_file.close()
            self.out_stream = None

    def record_vid(self, frame):
        """
        convert frames to packets and write to file
        """
        new_frame = av.VideoFrame(
            width=frame.width, height=frame.height, format=frame.format.name)
        for i in range(len(frame.planes)):
            new_frame.planes[i].update(frame.planes[i])
        pkt = None
        try:
            pkt = self.out_stream.encode(new_frame)
        except IOError as err:
            print("encoding failed: {0}".format(err))
        if pkt is not None:
            try:
                self.out_file.mux(pkt)
            except IOError:
                print('mux failed: ' + str(pkt))

    def take_picture(self, speed):
        """Tell drone to take picture, image sent to file handler"""
        if speed == 0:
            return
        self.drone.take_picture()

    def palm_land(self, speed):
        """Tell drone to land"""
        if speed == 0:
            return
        self.drone.palm_land()

    def toggle_tracking(self, speed):
        """ Handle tracking keypress"""
        if speed == 0:  # handle key up event
            return
        self.tracking = not self.tracking
        print("tracking:", self.tracking)
        return

    def toggle_zoom(self, speed):
        """
        In "video" mode the self.drone sends 1280x720 frames.
        In "photo" mode it sends 2592x1936 (952x720) frames.
        The video will always be centered in the window.
        In photo mode, if we keep the window at 1280x720 that gives us ~160px on
        each side for status information, which is ample.
        Video mode is harder because then we need to abandon the 16:9 display size
        if we want to put the HUD next to the video.
        """
        if speed == 0:
            return
        self.drone.set_video_mode(not self.drone.zoom)

    def flight_data_handler(self, event, sender, data):
        """Listener to flight data from the drone."""
        text = str(data)
        if self.prev_flight_data != text:
            self.prev_flight_data = text

    def handle_flight_received(self, event, sender, data):
        """Create a file in ~/Pictures/ to receive image from the drone"""
        path = '%s/Pictures/tello-%s.jpeg' % (
            os.getenv('HOME'),
            datetime.datetime.now().strftime(self.date_fmt))
        with open(path, 'wb') as out_file:
            out_file.write(data)
        print('Saved photo to %s' % path)


if __name__ == '__main__':
    main()
