import math
import threading

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Rectangle, Fbo, ClearColor, ClearBuffers, Scale, Translate
from kivy.core.image import Image as CoreImage
from kivy.clock import mainthread
from kivy.core.window import Window
from plyer import filechooser
from kivy.app import App
from PIL import Image as PILImage




import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'keypoints'))

from keypoints.run_tests import run_everything

CONNECTIONS = [
    [[1, 2], [2, 3]],
    [[4, 5], [5, 6]],
    [[7, 8], [8, 9]],
    [[10, 11], [11, 12]],
    [[14, 15], [15, 16], [14, 16], [22, 14], [22, 16]],
    [[18, 19], [19, 20]],
    # [[23, 24],[24, 26],[25, 26],[23, 25]]
]

GUIDELINES = [
    [21, 22], #vertical line
    [8,11], # across eyes
    [14,16], #across nose
    [18,20] #across mouth
]
_connection_ids = {p for feature in CONNECTIONS for pair in feature for p in pair}

# only block movement for guideline points that aren't also sketch points (21 and 22)
GUIDELINE_POINT_IDS = {p for pair in GUIDELINES for p in pair} - _connection_ids

def _dist_sq(ax, ay, bx, by):
    return (ax - bx) ** 2 + (ay - by) ** 2

def _point_near_line(px, py, x1, y1, x2, y2, threshold=10):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return _dist_sq(px, py, x1, y1) < threshold ** 2
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y) < threshold

def _get_connected_points(start_points, connections):
    connected = set(start_points)
    changed = True
    while changed:
        changed = False
        for feature in connections:
            for p1, p2 in feature:
                if p1 in connected or p2 in connected:
                    if p1 not in connected or p2 not in connected:
                        connected.add(p1)
                        connected.add(p2)
                        changed = True
    return connected

# drawing
class FaceMeshWidget(Widget):

    DOT_RADIUS = 6
    HIT_RADIUS = 12
    LINE_HIT   = 10
    LINE_WIDTH = 2
    DOT_COLOR  = (1, 0, 0, 1)
    LINE_COLOR = (1, 0, 0, 1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # empty state until load() is called

        self.texture     = None
        self.img_w       = 1
        self.img_h       = 1
        self.connections = []
        self._img_pts    = {}
        self._drag_point = None
        self._drag_group = None
        self._last_touch = None

        self._bbox = None
        

        self.show_background = True
        self.show_guidelines = True
        self.show_sketch     = True
        self.show_bbox = True
        self.guideline_color = (0, 0, 1, 0.7)
        self.bind(size=self._on_layout, pos=self._on_layout)

    def load(self, image_path, points_dict, connections):
        core_img         = CoreImage(image_path)
        self.texture     = core_img.texture
        self.img_w       = self.texture.width
        self.img_h       = self.texture.height
        self.connections = connections
        self._img_pts    = {int(k): (float(v[0]), float(v[1]))
                            for k, v in points_dict.items()}
        self._drag_point = None
        self._drag_group = None
        self._last_touch = None
        self._redraw()

#export the drawing as an image so that we can scale it around with eges stuff
    def export_as_image(self, filename):

        scale, ox, oy = self._scale_and_offset()
        width  = int(self.img_w * scale)
        height = int(self.img_h * scale)

        fbo = Fbo(size=(width, height), with_stencilbuffer=False)


        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()

            # convert image coords to fbo coords
            # origin is bottom-left, image origin is top-left, so flip y
            def convert(ix, iy):
                x = ix * scale
                y = (self.img_h - iy) * scale  # flip y
                return x, y

            Color(1, 0, 0, 1)
            # lines
            for feature in self.connections:
                for p1, p2 in feature:
                    if p1 in self._img_pts and p2 in self._img_pts:
                        x1, y1 = convert(*self._img_pts[p1])
                        x2, y2 = convert(*self._img_pts[p2])
                        Line(points=[x1, y1, x2, y2], width=2)

            # dots
            r = self.DOT_RADIUS
            for ix, iy in self._img_pts.values():
                x, y = convert(ix, iy)
                Ellipse(pos=(x - r, y - r), size=(r * 2, r * 2))

        fbo.draw()

        pixels = fbo.texture.pixels
        img = PILImage.frombytes("RGBA", (width, height), pixels)
        img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
        img.save(filename)
        # print(f"Sketch path: {filename}")

    def export_guidelines_as_image(self, filename):
        scale, ox, oy = self._scale_and_offset()
        width  = int(self.img_w * scale)
        height = int(self.img_h * scale)

        fbo = Fbo(size=(width, height), with_stencilbuffer=False)

        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()

            def convert(ix, iy):
                x = ix * scale
                y = (self.img_h - iy) * scale
                return x, y

            # bbox
            if self._bbox is not None:
                bx, by, bw, bh = self._bbox
                fx = bx * scale
                fy = (self.img_h - (by + bh)) * scale
                Color(0, 0.5, 1, 1)
                Line(rectangle=(fx, fy, bw * scale, bh * scale), width=2)

            # guidelines
            Color(0, 0, 1, 1)
            wpts_img = self._img_pts 

            img_left   = 0
            img_right  = width
            img_top    = height
            img_bottom = 0

            for pair in GUIDELINES:
                p1, p2 = pair
                if p1 not in wpts_img or p2 not in wpts_img:
                    continue
                x1, y1 = convert(*wpts_img[p1])
                x2, y2 = convert(*wpts_img[p2])

                if abs(x2 - x1) < abs(y2 - y1):
                    avg_x = (x1 + x2) / 2
                    Line(points=[avg_x, img_bottom, avg_x, img_top],
                        width=1.2, dash_offset=5, dash_length=8)
                else:
                    avg_y = (y1 + y2) / 2
                    Line(points=[img_left, avg_y, img_right, avg_y],
                        width=1.2, dash_offset=5, dash_length=8)

        fbo.draw()
        #bruhhhhhhhhhhhhhhh
        pixels = fbo.texture.pixels
        img = PILImage.frombytes("RGBA", (width, height), pixels)
        img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
        img.save(filename)
        # print(f"Guidelines path: {filename}")

    def _scale_and_offset(self):
        ww, wh = self.width, self.height
        if ww <= 0 or wh <= 0:
            return 1.0, 0.0, 0.0
        scale = min(ww / self.img_w, wh / self.img_h)
        ox = self.x + (ww - self.img_w * scale) / 2.0
        oy = self.y + (wh - self.img_h * scale) / 2.0
        return scale, ox, oy

    def _img_to_widget(self, ix, iy):
        scale, ox, oy = self._scale_and_offset()
        return ox + ix * scale, oy + (self.img_h - iy) * scale

    def _widget_pts(self):
        return {k: self._img_to_widget(ix, iy)
                for k, (ix, iy) in self._img_pts.items()}

    def _on_layout(self, *_):
        self._redraw()

    def set_bbox(self, bbox):
        self._bbox = bbox  # expects (bx, by, bw, bh) which gets passed from run_everything()
        self._redraw()

    def _redraw(self):
        self.canvas.clear()
        if self.texture is None:
            return
        with self.canvas:
            # background image
            scale, ox, oy = self._scale_and_offset()

            if self.show_background:
                Color(1, 1, 1, 1)
                Rectangle(texture=self.texture, pos=(ox, oy), size=(self.img_w * scale, self.img_h * scale))
                
            
            
            wpts = self._widget_pts()

            if self.show_guidelines:
                self._draw_guidelines(wpts)
            if self._bbox is not None and self.show_bbox:
                bx, by, bw, bh = self._bbox
                wx = ox + bx * scale
                wy = oy + (self.img_h - (by + bh)) * scale
                Color(0, 0.5, 1, 1)
                Line(rectangle=(wx, wy, bw * scale, bh * scale), width=2)

            Color(*self.LINE_COLOR)


        
            # red lines and dots
            if self.show_sketch:
                Color(*self.LINE_COLOR)
                for feature in self.connections:
                    for p1, p2 in feature:
                        if p1 in wpts and p2 in wpts:
                            x1, y1 = wpts[p1]
                            x2, y2 = wpts[p2]
                            Line(points=[x1, y1, x2, y2], width=self.LINE_WIDTH)
                Color(*self.DOT_COLOR)
                r = self.DOT_RADIUS
                for wx, wy in wpts.values():
                    Ellipse(pos=(wx - r, wy - r), size=(r * 2, r * 2))

                

    def _draw_guidelines(self, wpts):
        Color(*self.guideline_color) 
        scale, ox, oy = self._scale_and_offset()
        img_left= ox
        img_right= ox + self.img_w * scale
        img_top = oy + self.img_h * scale
        img_bottom = oy

        for pair in GUIDELINES:
            p1, p2 = pair
            if p1 not in wpts or p2 not in wpts:
                continue
            x1, y1 = wpts[p1]
            x2, y2 = wpts[p2]

            # vertical guideline, use average x of both points, extend top to bottom
            if abs(x2 - x1) < abs(y2 - y1):
                avg_x = (x1 + x2) / 2
                Line(points=[avg_x, img_bottom, avg_x, img_top], width=1.2, dash_offset=5, dash_length=8)
            # horizontal guideline, use average y of both points, extend left to right
            else:
                avg_y = (y1 + y2) / 2
                Line(points=[img_left, avg_y, img_right, avg_y], width=1.2, dash_offset=5, dash_length=8)

    # interact
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        touch.grab(self)
        mx, my = touch.pos
        wpts = self._widget_pts()

        for k, (wx, wy) in wpts.items():
            if k in GUIDELINE_POINT_IDS:
                continue  
            if _dist_sq(mx, my, wx, wy) < self.HIT_RADIUS ** 2:
                self._drag_point = k
                self._drag_group = None
                self._last_touch = (mx, my)
                return True

        for feature in self.connections:
            for p1, p2 in feature:
                if p1 not in wpts or p2 not in wpts:
                    continue
                x1, y1 = wpts[p1]
                x2, y2 = wpts[p2]
                if _point_near_line(mx, my, x1, y1, x2, y2, self.LINE_HIT):
                    self._drag_group = _get_connected_points({p1, p2}, self.connections)
                    self._drag_point = None
                    self._last_touch = (mx, my)
                    return True
        return False

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return False
        if self._drag_point is None and self._drag_group is None:
            return False
        mx, my = touch.pos
        lx, ly = self._last_touch
        scale, _, _ = self._scale_and_offset()
        dix =  (mx - lx) / scale
        diy = -(my - ly) / scale

        if self._drag_point is not None:
            ix, iy = self._img_pts[self._drag_point]
            self._img_pts[self._drag_point] = (ix + dix, iy + diy)
        elif self._drag_group is not None:
            for k in self._drag_group:
                if k in self._img_pts:
                    ix, iy = self._img_pts[k]
                    self._img_pts[k] = (ix + dix, iy + diy)

        self._last_touch = (mx, my)
        self._redraw()
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        touch.ungrab(self)
        self._drag_point = None
        self._drag_group = None
        self._last_touch = None
  
        return True

    def get_points(self):
        return dict(self._img_pts)

# the Screen
class EditKeypoints(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._show_bg = True
        self._show_guidelines = True
        self._show_sketch = True

        root = BoxLayout(orientation='vertical')

        # buttons
        toolbar = BoxLayout(size_hint=(1, None), height=70, spacing=50, padding=10)

        back_btn = Button(text='Save & Exit', size_hint=(None, 1), width=180) #for save and exit
        guide_btn = Button(text='Hide Guidelines', size_hint=(None, 1), width=240)#for hiding guidelines
        sketch_btn = Button(text='Hide Sketch', size_hint=(None, 1), width=200)#for hiding sketch
        toggle_btn = Button(text='Hide Image', size_hint=(None, 1), width=200)#toggle image on/off button

        back_btn.bind(on_press=self.save_and_go_back) #changed: instead of just leaving call the save and leave function
        toggle_btn.bind(on_press=lambda _: self._toggle('bg', toggle_btn))
        guide_btn.bind(on_press=lambda _: self._toggle('guidelines', guide_btn))
        sketch_btn.bind(on_press=lambda _: self._toggle('sketch', sketch_btn))

        self._status = Label(text='Upload an image to begin.',
                             color=(0.7, 0.7, 0.7, 1))
        toolbar.add_widget(back_btn)
        toolbar.add_widget(self._status)
        toolbar.add_widget(guide_btn)
        toolbar.add_widget(sketch_btn)
        
        
        toolbar.add_widget(toggle_btn)

        # the drawing widget
        self._mesh = FaceMeshWidget(size_hint=(1, 1))

        root.add_widget(toolbar)
        root.add_widget(self._mesh)
        self.add_widget(root)

    def load_image(self, path):
        print("A")
        print(path)
        #called in mainmenu when you pick file
        self._image_path = path
        self._status.text = 'Running CNN, please wait' #since it can take a few seconds
        threading.Thread(target=self._run_cnn, args=(path,), daemon=True).start()

    def load_existing(self, path, points_dict):

        #dont want to re-run CNN every time image is clicked, just save the points
        self._image_path = path
        self._status.text = "Loaded saved sketch"

        self._mesh.load(path, points_dict, CONNECTIONS)

    def _run_cnn(self, path):
        try:
            print("C")
            points_dict, bbox = run_everything(path)
            print("AAAAAAAA")
            self._cnn_done(path, points_dict, bbox)
        except Exception as e:
            print("D")
            self._cnn_error(str(e))

    @mainthread
    def _cnn_done(self, path, points_dict, bbox):
        self._status.text = f'Loaded: {path.split("/")[-1]} drag dots or lines to adjust'
        self._mesh.load(path, points_dict, CONNECTIONS)
        self._mesh.set_bbox(bbox)

    @mainthread
    def _cnn_error(self, msg):
        self._status.text = f'Error: {msg}'

    def get_points(self):
        return self._mesh.get_points()
    
    def _toggle(self, target, btn):
        if target == 'bg':
            self._show_bg = not self._show_bg
            self._mesh.show_background = self._show_bg
            btn.text = 'Show Image' if not self._show_bg else 'Hide Image'
        elif target == 'guidelines':
            self._show_guidelines = not self._show_guidelines
            self._mesh.show_guidelines = self._show_guidelines
            self._mesh.show_bbox = self._show_guidelines
            btn.text = 'Show Guidelines' if not self._show_guidelines else 'Hide Guidelines'
        elif target == 'sketch':
            self._show_sketch = not self._show_sketch
            self._mesh.show_sketch = self._show_sketch
            btn.text = 'Show Sketch' if not self._show_sketch else 'Hide Sketch'
        self._mesh._redraw()
    
    # def toggle_background(self, *args):
    #     self._mesh.show_background = not self._mesh.show_background
    #     self._mesh._redraw()

    def save_and_go_back(self, *args):
        app = App.get_running_app()

        filename = "saved_sketch.png"
        guide_filename = 'saved_guidelines.png'

        self._mesh.export_as_image(filename)
        self._mesh.export_guidelines_as_image(guide_filename)
        
        app.saved_sketch_path = filename
        app.saved_image_path = self._image_path
        app.saved_points = self._mesh.get_points() #save the CURRENT points


        self.manager.current = 'mainMenu'