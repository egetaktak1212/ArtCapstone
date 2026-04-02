import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/egeta/Downloads/Screenshot_2026-02-06-01-36-55-95_965bbf4d18d205f782c6b8409c5773a42.jpg"

ref = cv2.imread(path)

#canvas original size ratio
canvash = 14
canvasw = 11

#make a zero matrix of the same size as the canvas. this is what i will eventually project and this is where i will put the transformed ref image
canvasref = np.zeros((canvash*200, canvasw*200, 3), dtype=np.uint8)

canvas_center = (canvasref.shape[0]//2, canvasref.shape[1]//2)

#find scale and scale down the ref image to fit in the canvas initially
def scaleToFit(image, background):
    h_ref, w_ref = image.shape[:2]

    h_canv, w_canv = background.shape[:2]

    scale_tuple = (h_canv/h_ref, w_canv/w_ref)
    scale = min(scale_tuple[0], scale_tuple[1])

    new_h_ref = int(h_ref * scale)
    new_w_ref = int(w_ref * scale)

    return cv2.resize(image, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA), scale

def center_pad(image, canvas):
    H, W = canvas.shape[:2]
    h, w = image.shape[:2]

    result = np.zeros(canvas.shape, dtype=image.dtype)

    start_y = (H - h) // 2
    start_x = (W - w) // 2

    result[start_y:start_y+h, start_x:start_x+w] = image
    return result

scaled_ref, scale = scaleToFit(ref, canvasref)

padded_ref = center_pad(scaled_ref, canvasref)



def place_image(image, canvas, center_x, center_y):
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

shifted_ref = place_image(scaled_ref, canvasref, canvas_center[1], canvas_center[0])
plt.imshow(shifted_ref)
plt.show()
#the algorithm will put scaled_ref on the canvas. so this function will change scaled ref but use the original so that detail is not lost.
# def scale_image(image, original, amount):
#     H, W = original.shape[:2]
#     h, w = image.shape[:2]
    
#     scale = h/H
#     scale += amount
#     scale = max(0.00001, scale)
    
    
#     new_h_ref = int(H * scale) + 1
#     new_w_ref = int(W * scale) + 1

#     return cv2.resize(original, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA)

def scale_image(original, scale):
    H, W = original.shape[:2]
    new_h_ref = int(H * scale) 
    new_w_ref = int(W * scale) 

    return cv2.resize(original, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA)

def turn_into_big_box_using_diagonals(image):
    h, w = image.shape[:2]
    point1 = np.array([0,0])
    point2 = np.array([h,w])
    dist = int(np.linalg.norm(point1-point2))
    
    padding = 5
    
    result = np.zeros((dist + padding, dist + padding, 3), dtype=np.uint8)
    
    return center_pad(image, result)
    
def rotate_diagonalized_image(image, theta, scale = 1.0):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, theta, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image


scaled_ref = scale_image(ref, scale)

shifted_ref = place_image(scaled_ref, canvasref, canvas_center[1], canvas_center[0])

plt.imshow(shifted_ref)
plt.show()

#1. Make canvas ref
#2. scaled_ref = Scale ref to fit
#2.5 save scale for this
#3. diagonalize scaled_ref
#4. put it in the center
#


# for every transform
# take the original. scale it with the factor
# diagonalize
# rotate with rotation factor
# put it wherever our saved cx cy are
