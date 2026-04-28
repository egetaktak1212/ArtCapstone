import cv2
import numpy as np
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen
from kivy.uix.screenmanager import ScreenManager, NoTransition
from kivy.app import Builder
from kivy.core.window import Window
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.button import ButtonBehavior
from plyer import filechooser
from edit_keypoints import EditKeypoints
import platform
import os

import sys
print(sys.executable)

Builder.load_file('main2.kv')

Window.resizable = False
Window.clearcolor = (35/255, 35/255, 35/255, 1)

#TO DO: UPLOAD SYSTEM!!!!
#CANT START UNLESS UPLOAD IMAGE
#change text showing whats uploaded
#all the rescaling shit guhhhh

loaded_ref_image_path = "keypoints\image\cameron.JPG"

#this class gets used later to make preview window clickable
class ClickableImage(ButtonBehavior, Image):
    pass

class CamWidget(BoxLayout):
    
    def scale_change(self, up):
        if up:
            self.scale += 0.1
        elif not up: self.scale -= 0.1
        
        self.scale = max(0.1, self.scale)
        self.scale = min(5, self.scale)
        self.apply_transforms()
    
    def pos_change(self, v):
        self.cursorx += int(v[0])
        self.cursory += int(v[1])
        self.apply_transforms()

        
    def angle_change(self, clock):
        if clock:
            self.angle -= 15
        elif not clock: self.angle += 15
        self.apply_transforms()
        
    def alpha_change(self, up):
        if up:
            self.alpha -= 0.1
        else:
            self.alpha += 0.1
        
        self.alpha = max(0, self.alpha)
        self.alpha = min(1, self.alpha)
    
    def pause(self):
        self.live = not self.live
    
    #find the line intersection of the two vectors. 
    def line_intersection(self, p1, p2, p3, p4):
        #this whole thing is created to use Cramer's Rule
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        determinant = a1 * b2 - a2 * b1

        if determinant == 0:
            return None
        
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        
        return (x, y)
    def order_from_ids(self, detected):
            return np.array([
                detected["tl"],
                detected["tr"],
                detected["br"],
                detected["bl"]
            ], dtype=np.float32)
            
    def performWarping(self, paint, ref, corners, homo = None): 
        
        refHeight, refWidth = ref.shape[:2]
        #currently using the boundaries of the ref image as the shit to project
        refCorners = np.array([
            (0, 0),
            (refWidth - 1, 0),
            (refWidth - 1, refHeight - 1),
            (0, refHeight - 1),
        ])
        #if this func was not called with an old homo, make a new one
        if (homo is None):
            paintCorners = corners
            homo, _ = cv2.findHomography(refCorners, paintCorners)

        #warp shit. if func was called with an old homo, it would be used here
        warped_ref = cv2.warpPerspective(ref, homo, (paint.shape[1], paint.shape[0]))

        result = paint.copy()
        #cv2 weighted thing to overlay with lover alpha/opacity
        beta = (1.0 - self.alpha)
        dst = cv2.addWeighted(result, self.alpha, warped_ref, beta, 0.0)

        #wherever img data exists in the warped reference, combine the overlayed dst version with it.
        mask = warped_ref.sum(axis=2) > 0
        result[mask] = dst[mask]
        return result, homo

    #if we have diagonal corners
    def find_corner(self, corners, gray):
        #same shit from the main loop, got lazy
        corners, ids, _ = self.detector.detectMarkers(gray)
        detected = {}
        if ids is not None:
            ids = ids.flatten()
            for i in range(len(ids)):
                marker_id = ids[i]
                if marker_id in self.marker_map:
                    c = corners[i][0]
                    label = self.marker_map[marker_id]
                    detected[label] = c
        #if we got the top left and bottom right
        if "tl" in detected and "br" in detected:
            topCorner = detected["tl"]
            bottomCorner = detected["br"]
            #find where vectors of the sides of the two markers intersect, which is ideally where the other corners would be
            bl = self.line_intersection(topCorner[0], topCorner[3], bottomCorner[2], bottomCorner[3])
            tr = self.line_intersection(topCorner[0], topCorner[1], bottomCorner[2], bottomCorner[1])
            #return an alr sorted version with our inputted corners + the predicted ones
            return np.array([
            topCorner[0],
            tr,
            bottomCorner[2],
            bl
            ], dtype=np.float32)
        #same shit but for the other corners
        elif "tr" in detected and "bl" in detected:
            topCorner = detected["tr"]
            bottomCorner = detected["bl"]
            br = self.line_intersection(topCorner[1], topCorner[2], bottomCorner[3], bottomCorner[2])
            tl = self.line_intersection(topCorner[1], topCorner[0], bottomCorner[3], bottomCorner[0])
            return np.array([
            tl,
            topCorner[1],
            br,
            bottomCorner[3]
            ], dtype=np.float32)
    
    def scaleToFit(self, image, background):
        h_ref, w_ref = image.shape[:2]

        h_canv, w_canv = background.shape[:2]

        scale_tuple = (h_canv/h_ref, w_canv/w_ref)
        scale = min(scale_tuple[0], scale_tuple[1])

        new_h_ref = int(h_ref * scale)
        new_w_ref = int(w_ref * scale)

        return cv2.resize(image, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA), scale

    def center_pad(self, image, canvas):
        H, W = canvas.shape[:2]
        h, w = image.shape[:2]

        result = np.zeros(canvas.shape, dtype=image.dtype)

        start_y = (H - h) // 2
        start_x = (W - w) // 2

        result[start_y:start_y+h, start_x:start_x+w] = image
        return result

    def place_image(self, image, canvas, center_x, center_y):
        H, W = canvas.shape[:2]
        h, w = image.shape[:2]

        #template we gonna put everything into
        result = np.zeros(canvas.shape, dtype=image.dtype)

        #top left corner of where we wanna put image
        start_x = int(center_x - w // 2)
        start_y = int(center_y - h // 2)

        #bottom right corner
        end_x = start_x + w
        end_y = start_y + h

        #what actual range on the canvas we can occupy. if our start is outside the bounds, itll just start at 0
        canvas_x1 = max(0, start_x)
        canvas_y1 = max(0, start_y)
        #bottom right
        canvas_x2 = min(W, end_x)
        canvas_y2 = min(H, end_y)


        #either start with the first pixel on the picture, or start farther right if start_x is negative (cuz we need to skip)
        img_x1 = max(0, -start_x)
        img_y1 = max(0, -start_y)
        #get the range of pixels that span the canvas incase it goes past it
        img_x2 = img_x1 + (canvas_x2 - canvas_x1)
        img_y2 = img_y1 + (canvas_y2 - canvas_y1)

        #if top left is actually top left
        if canvas_x1 < canvas_x2 and canvas_y1 < canvas_y2:
            result[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = image[img_y1:img_y2, img_x1:img_x2]

        return result

    def scale_image(self, original, scale):
        H, W = original.shape[:2]
        new_h_ref = int(H * scale) 
        new_w_ref = int(W * scale) 

        return cv2.resize(original, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA)

    def turn_into_big_box_using_diagonals(self, image):
        h, w = image.shape[:2]
        point1 = np.array([0,0])
        point2 = np.array([h,w])
        dist = int(np.linalg.norm(point1-point2))
        
        padding = 5
        
        result = np.zeros((dist + padding, dist + padding, 3), dtype=np.uint8)
        
        return self.center_pad(image, result)
        
    def rotate_diagonalized_image(self, image, theta, scale = 1.0):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, theta, scale)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image

  
    def getStarterRef(self,original):
        #canvas original size ratio
        canvash = 14
        canvasw = 11

        #make a zero matrix of the same size as the canvas. this is what i will eventually project and this is where i will put the transformed ref image
        self.canvasref = np.zeros((canvash*200, canvasw*200, 3), dtype=np.uint8)
        scaled_ref, initial_scale = self.scaleToFit(original, self.canvasref)
        
        h, w = self.canvasref.shape[:2]
        center = (w // 2, h // 2)
        
        return scaled_ref, initial_scale, center
    
    def changeOverlay(self, guidelines, sketch, og):
        frame = np.zeros_like(self.originalref)

        if (og):
            frame = self.originalref
        if sketch:
            frame = self.masking(self.sketch, frame)
        if guidelines:
            frame = self.masking(self.guideline, frame)
        return frame

    def masking(self, fg, bg):
        
        mask = np.any(fg != [0, 0, 0], axis=-1)
        result = bg.copy()
        result[mask] = fg[mask]
        
        return result

    def hideShit(self, i):
        if i == 0:
            self.ogshow = not self.ogshow
        elif i == 1:
            self.guideshow = not self.guideshow
        elif i == 2:
            self.sketchshow = not self.sketchshow
        self.activeref = self.changeOverlay(self.guideshow, self.sketchshow, self.ogshow)

    def apply_transforms(self):
        #follow bible
        
        # for every transform
        # take the original. scale it with the factor
        first_scale = self.scale_image(self.activeref, self.scale)
        # diagonalize
        second_diagonalize = self.turn_into_big_box_using_diagonals(first_scale)
        # rotate with rotation factor
        third_rotate = self.rotate_diagonalized_image(second_diagonalize, self.angle)
        # put it wherever our saved cx cy are
        four_place = self.place_image(third_rotate, self.canvasref, self.cursorx, self.cursory)
        # set the ref we are using to this final thing
        self.ref = four_place
    
    def startCamera(self):
        global loaded_ref_image_path
        self.live = True
        self.last_frame = None
        self.last_H = None
        #setup cam
        if platform.system() == "Darwin":
            self.capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        elif platform.system() == "Windows":
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.showimage = True
        self.showsketch = True
        self.showguidelines = True

        
        #these are the dirs and images for th esketch and guidelines
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sketch = cv2.imread(os.path.join(base_dir, "saved_sketch.png"))
        self.guideline = cv2.imread(os.path.join(base_dir, "saved_guidelines.png"))

        #the active ref will be the original ref overlayed w whatever you want
        self.originalref = cv2.imread(loaded_ref_image_path)

        #resize cuz for some reason mias outputs are bigger
        height, width = self.originalref.shape[:2]
        self.sketch = cv2.resize(self.sketch, (width, height), interpolation=cv2.INTER_LINEAR)
        self.guideline = cv2.resize(self.guideline, (width, height), interpolation=cv2.INTER_LINEAR)

        self.ogshow = True
        self.guideshow = True
        self.sketchshow = True
        self.activeref = self.changeOverlay(True, True, True)

        #big shot here. creating the og scaled_ref and setting params for scale, angle etc
        
        #follow the bible
        #1. Make canvas ref
        #2. scaled_ref = Scale ref to fit
        #2.5 save scale for this
        self.scaled_ref, self.scale, self.canvas_center = self.getStarterRef(self.originalref)
        self.angle = 0
        self.cursorx = self.canvas_center[0]
        self.cursory = self.canvas_center[1]
        self.alpha = 0.5
        #3. diagonalize scaled_ref
        self.diagonalized_ref = self.turn_into_big_box_using_diagonals(self.scaled_ref)
        #4. put it in the center
        self.ref = self.place_image(self.diagonalized_ref, self.canvasref, self.cursorx, self.cursory)

        self.apply_transforms()


        #im using these smaller ArUco's cuz i need better detection from far away
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

        #map for tying the IDs of the markers to which corner theyre supposed to be
        self.marker_map = {
            1: "tl",
            3: "tr",
            4: "br",
            6: "bl",
        }
        #this should start when the user picks live camera
        
        
        
        Clock.schedule_interval(self.update, 1.0 / 60.0)
    
    def stopCamera(self):
        Clock.unschedule(self.update)
        if self.capture:
            self.capture.release()
            self.capture = None
        self.last_frame = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.orientation = "vertical"
        self.camera = Image()
        self.add_widget(self.camera)
        self.move_button_amount = 50
        self.capture = None
        
        

    def frameFunction(self,frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect markers and store corners and ids in arrays
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            #makes np shit into normal array
            ids = ids.flatten()
            detected_markers = {}
            #for every marker we detected
            for i in range(len(ids)):
                marker_id = ids[i]
                #if marker is one of our accepted ones
                if marker_id in self.marker_map:
                    #this gets an array of tuples, with 0 being top left, so on and so forth
                    c = corners[i][0]
                    #if its the top left one, get c[0], etc.
                    if self.marker_map[marker_id] == "tl":
                        selected_corner = c[0]
                    elif self.marker_map[marker_id] == "tr":
                        selected_corner = c[1]
                    elif self.marker_map[marker_id] == "br":
                        selected_corner = c[2]
                    elif self.marker_map[marker_id] == "bl":
                        selected_corner = c[3]
                    #dictionary where the corner coords will be saved as "tl" or "tr" and whatever
                    detected_markers[self.marker_map[marker_id]] = selected_corner
                    
                    
            #here, i would check if we have a set referance image. if we do, update it with the users qualifications which are stored somewhere im sure
            #if the ref is there, proj it. if its not there, say no reference            
                    
                    
        
            #if we have all four corners
            if len(detected_markers) == 4:
                #order them to match the tl, tr, br, bl order for homo
                ordered = self.order_from_ids(detected_markers)
                #run the full thing
                frame, self.last_H =self. performWarping(frame, self.ref, ordered)
            elif len(detected_markers) >= 2 and self.find_corner(corners, gray) is not None:
                #if we only got two, lets see if they're opposite diagonal corners
                #if so, we estimate the other corners and warp based on it
                frame, self.last_H = self.performWarping(frame, self.ref, self.find_corner(corners, gray))
            #else, we didnt detect the fucking thing
            elif self.last_H is not None:
            #maybe future, add a timer. like, if the canvas isnt there for 1 second, say no canvas detected
                #if all of it didnt work, we use the last homo from eons past
                frame, _ = self.performWarping(frame, self.ref, None, self.last_H)
            #elif legit nothing lets say canvas not detected

            #nice debug drawing
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        return frame
        
    def update(self, dt):
        if self.live or self.last_frame is None:
            ret, frame = self.capture.read()
            self.last_frame = frame
            if not ret:
                return
        else: frame = self.last_frame

        frame = self.frameFunction(frame)

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr'
        )
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.camera.texture = texture

class MainMenu(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def cam_opener(self):
        app = App.get_running_app()
        if (app.did_we_upload):
            self.manager.current = 'cameraWindow'

    def file_picker(self):
        app = App.get_running_app()
        app.did_we_upload = True

        if platform.system() == "Darwin":
            self.file_picker_mac()
        elif platform.system() == "Windows":
            self.file_picker_win()

    def reformat_path(self, path):
        reformat_path = path[1:]

    def file_picker_win(self):
        global loaded_ref_image_path
        picked_paths = filechooser.open_file(title="Select a file", multiple = False, filters = ["*jpg", "*png", "*jpeg"])
        if picked_paths:
            path = picked_paths[0]
            if path:
                global loaded_ref_image_path
                loaded_ref_image_path = path
                #path = path[2:].replace('\\', '/')
                print(path)
                editor = self.manager.get_screen('keypointEditor')
                editor.load_image(path)
                self.manager.current = 'keypointEditor'

        

    def file_picker_mac(self):
        #other one didnt work on mac so this is my workaround, camera also deosnt work for me but,,, alas,,,
        import subprocess
        script = '''
        tell application "System Events"
            activate
        end tell
        tell application "System Events"
            set theFile to choose file with prompt "Select an image" of type {"jpg", "jpeg", "png"}
            return POSIX path of theFile
        end tell
        '''
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True
            )
            path = result.stdout.strip()
            if path:
                global loaded_ref_image_path
                loaded_ref_image_path = path
                editor = self.manager.get_screen('keypointEditor')
                editor.load_image(path)
                self.manager.current = 'keypointEditor'
        except Exception as e:
            print(f"File picker error: {e}")

     

    def on_enter(self, *args):
        Window.size = (1000, 750)

        app = App.get_running_app()
        if app.saved_sketch_path:
            self.show_saved_sketch(app)



    def show_saved_sketch(self, app):
        container = self.ids.previewContainer
        container.clear_widgets()
        path = self.makePreview(app)
        preview = ClickableImage(source=path)
        preview.reload()

        preview.size_hint = (1, 1)
        preview.fit_mode="contain"


        preview.bind(on_press=lambda instance: self.open_editor_again())

        container.add_widget(preview)

    def makePreview(self, app):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        sketch = cv2.imread(os.path.join(base_dir, "saved_sketch.png"))
        guideline = cv2.imread(os.path.join(base_dir, "saved_guidelines.png"))
        originalref = cv2.imread(app.saved_image_path)
        height, width = originalref.shape[:2]
        sketch = cv2.resize(sketch, (width, height), interpolation=cv2.INTER_LINEAR)
        guideline = cv2.resize(guideline, (width, height), interpolation=cv2.INTER_LINEAR)

        final = self.masking(sketch, originalref)
        final = self.masking(guideline, final)
        
        final_path = os.path.join(base_dir, "preview.png")

        cv2.imwrite(final_path, final)
        return final_path



    def masking(self, fg, bg):
        
        mask = np.any(fg != [0, 0, 0], axis=-1)
        result = bg.copy()
        result[mask] = fg[mask]
        
        return result



    def open_editor_again(self):
        app = App.get_running_app()
        if (app.did_we_upload):
            editor = self.manager.get_screen('keypointEditor')

            editor.load_existing(
                app.saved_image_path,
                app.saved_points
            )

            self.manager.current = 'keypointEditor'
            


class CamWindow(Screen):
        
    def on_enter(self, *args):
        # img = CoreImage("./assets/campage/page.png")
        # Window.size = img.size
        Clock.schedule_once(lambda dt: self.ids.camwidget.startCamera(), 1)

    
    def on_leave(self, *args):
        self.ids.camwidget.stopCamera()

class refProjector(App):
    #will contain paths to image and sketch
    saved_sketch_path = None
    saved_image_path = None
    did_we_upload = False
    
    def build(self):
        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(MainMenu(name='mainMenu'))
        sm.add_widget(CamWindow(name='cameraWindow'))
        sm.add_widget(EditKeypoints(name='keypointEditor'))
    

        return sm
    
refproject = refProjector()
refproject.run()