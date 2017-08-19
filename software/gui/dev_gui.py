# https://www.tutorialspoint.com/python/python_gui_programming.htm

import Tkinter as tk
from PIL import ImageTk, Image

import lcm


IMAGE_CHANNELS = "IMAGE_RAW"


class DevGUI():
    def __init__(self):
        self.setupWindow()
        self.setupLCM()

        # Display image: https://stackoverflow.com/questions/23901168/how-do-i-insert-a-jpeg-image-into-a-python-tkinter-window
        # Path to image
        path = "/home/eon-alone/projects/controls-drawer/results/line_following/test_8_10/frame_1502414920918577.png"
        import cv2
        frame = cv2.imread(path)
        #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(self.window, image = img)
        #The Pack geometry manager packs widgets in rows or columns.
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        # TODO
        # Okay, the reason I don't know how to use the GUI yet is that I need
        # to call lcmobj.handle() to get onImage to work but the mainloop is
        # running and isn't calling handle()
        self.window.mainloop()

    def setupWindow(self):
        self.window = tk.Tk()
        self.window.title("Controls Drawer Dev GUI")
        self.window.geometry("700x700")
        self.window.configure(background='grey')

    def setupLCM(self):
        self.lcmobj = lcm.LCM()
        self.lcmobj.subscribe(IMAGE_CHANNELS, self.onImage)

    def onImage(self, channel, data):
        print("Received an image on channel {}!".format(channel))
        # Decode/parse out the image
        image = lcm_msgs.auto_decode(channel, data)
        # Get actual image data that cv2 can handle
        frame = lcm_msgs.image_t_to_nparray(image)


if __name__ == "__main__":
    GUI = DevGUI()
