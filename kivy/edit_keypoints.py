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
]

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

        self.show_background = True
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
    def export_as_image(self, filename, include_background=True):

        scale, ox, oy = self._scale_and_offset()

        width = int(self.img_w * scale)
        height = int(self.img_h * scale)

        fbo = Fbo(size=(width, height))

        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            Scale(1, -1, 1) #flip vertically
            # optional background
            if include_background:
                Color(1, 1, 1, 1)
                Rectangle(texture=self.texture, pos=(0, 0), size=(width, height))

            # draw lines + points using current position even if user has moved them
            Color(1, 0, 0, 1)

            # convert points to export space
            def convert(ix, iy):
                x = ix * scale
                y = (self.img_h - iy) * scale
                return x, y

            # lines
            for feature in self.connections:
                for p1, p2 in feature:
                    if p1 in self._img_pts and p2 in self._img_pts:
                        x1, y1 = convert(*self._img_pts[p1])
                        x2, y2 = convert(*self._img_pts[p2])
                        Line(points=[x1, y1, x2, y2], width=2)

            # dots
            r = 4
            for ix, iy in self._img_pts.values():
                x, y = convert(ix, iy)
                Ellipse(pos=(x - r, y - r), size=(r * 2, r * 2))

        fbo.draw()
        fbo.texture.save(filename)

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

    def _redraw(self):
        self.canvas.clear()
        if self.texture is None:
            return
        with self.canvas:
            # background image
            scale, ox, oy = self._scale_and_offset()

            if self.show_background:
                Color(1, 1, 1, 1)
                Rectangle(texture=self.texture,
                        pos=(ox, oy),
                        size=(self.img_w * scale, self.img_h * scale))
                


            Color(*self.LINE_COLOR)
            wpts = self._widget_pts()
            for feature in self.connections:
                for p1, p2 in feature:
                    if p1 in wpts and p2 in wpts:
                        x1, y1 = wpts[p1]
                        x2, y2 = wpts[p2]
                        Line(points=[x1, y1, x2, y2], width=self.LINE_WIDTH)
            # dots
            Color(*self.DOT_COLOR)
            r = self.DOT_RADIUS
            for wx, wy in wpts.values():
                Ellipse(pos=(wx - r, wy - r), size=(r * 2, r * 2))

    # interact
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        touch.grab(self)
        mx, my = touch.pos
        wpts = self._widget_pts()

        for k, (wx, wy) in wpts.items():
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

        root = BoxLayout(orientation='vertical')

        # back button 
        toolbar = BoxLayout(size_hint=(1, None), height=50, spacing=10, padding=8)
        back_btn = Button(text='Save & Exit', size_hint=(None, 1), width=180)
        back_btn.bind(on_press=self.save_and_go_back) #changed: instead of just leaving call the save and leave function
        self._status = Label(text='Upload an image to begin.',
                             color=(0.7, 0.7, 0.7, 1))
        toolbar.add_widget(back_btn)
        toolbar.add_widget(self._status)
        #toggle image on/off button
        toggle_btn = Button(text='Toggle Background', size_hint=(None, 1), width=220)
        toggle_btn.bind(on_press=self.toggle_background)
        toolbar.add_widget(toggle_btn)

        # the drawing widget
        self._mesh = FaceMeshWidget(size_hint=(1, 1))

        root.add_widget(toolbar)
        root.add_widget(self._mesh)
        self.add_widget(root)

    def load_image(self, path):
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
            points_dict, bbox = run_everything(path)
            self._cnn_done(path, points_dict)
        except Exception as e:
            self._cnn_error(str(e))

    @mainthread
    def _cnn_done(self, path, points_dict):
        self._status.text = f'Loaded: {path.split("/")[-1]} drag dots or lines to adjust'
        self._mesh.load(path, points_dict, CONNECTIONS)

    @mainthread
    def _cnn_error(self, msg):
        self._status.text = f'Error: {msg}'

    def get_points(self):
        return self._mesh.get_points()
    
    def toggle_background(self, *args):
        self._mesh.show_background = not self._mesh.show_background
        self._mesh._redraw()

    def save_and_go_back(self, *args):
        app = App.get_running_app()

        filename = "saved_sketch.png"

        self._mesh.export_as_image(filename, include_background=False)

        app.saved_sketch_path = filename
        app.saved_image_path = self._image_path
        app.saved_points = self._mesh.get_points() #save the CURRENT points

        self.manager.current = 'mainMenu'