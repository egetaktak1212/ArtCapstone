import cv2
import mediapipe as mp

def isolate_face(image_path):
    mp_face = mp.solutions.face_detection
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(img_rgb)

    if not results.detections:
        print(f"No face detected in: {image_path}")
        return None

    if len(results.detections) > 1:
        print(f"Multiple faces found, using most confident")

    # Get the highest confidence detection
    best = max(results.detections, key=lambda d: d.score[0])
    box  = best.location_data.relative_bounding_box

    # MediaPipe returns relative coords [0,1] — convert to pixels
    x = int(box.xmin * img_w)
    y = int(box.ymin * img_h)
    w = int(box.width  * img_w)
    h = int(box.height * img_h)

    # Clamp to image bounds
    x = max(0, x);  y = max(0, y)
    w = min(w, img_w - x);  h = min(h, img_h - y)

    return (x, y, w, h)
