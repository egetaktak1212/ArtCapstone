import cv2
import numpy as np

def everything():

    ref = cv2.imread("./mictures/cameronwinterrightsize.png")

    #im using these smaller ArUco's cuz i need better detection from far away
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    #map for tying the IDs of the markers to which corner theyre supposed to be
    marker_map = {
        1: "tl",
        3: "tr",
        4: "br",
        6: "bl",
    }

    #sort corners nicely for homo
    def order_from_ids(detected):
        return np.array([
            detected["tl"],
            detected["tr"],
            detected["br"],
            detected["bl"]
        ], dtype=np.float32)

    #stuffed the whole warping algo into this function from last week
    def performWarping(paint, ref, corners, homo = None): 
        
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
        alpha = 0.5
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(result, alpha, warped_ref, beta, 0.0)

        #wherever img data exists in the warped reference, combine the overlayed dst version with it.
        mask = warped_ref.sum(axis=2) > 0
        result[mask] = dst[mask]
        return result, homo

    #if we have diagonal corners
    def find_corner(corners):
        #same shit from the main loop, got lazy
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            ids = ids.flatten()
            detected = {}
            
            for i in range(len(ids)):
                marker_id = ids[i]
                if marker_id in marker_map:
                    c = corners[i][0]
                    label = marker_map[marker_id]
                    detected[label] = c
        #if we got the top left and bottom right
        if "tl" in detected and "br" in detected:
            topCorner = detected["tl"]
            bottomCorner = detected["br"]
            #find where vectors of the sides of the two markers intersect, which is ideally where the other corners would be
            bl = line_intersection(topCorner[0], topCorner[3], bottomCorner[2], bottomCorner[3])
            tr = line_intersection(topCorner[0], topCorner[1], bottomCorner[2], bottomCorner[1])
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
            br = line_intersection(topCorner[1], topCorner[2], bottomCorner[3], bottomCorner[2])
            tl = line_intersection(topCorner[1], topCorner[0], bottomCorner[3], bottomCorner[0])
            return np.array([
            tl,
            topCorner[1],
            br,
            bottomCorner[3]
            ], dtype=np.float32)
        

    import numpy as np
    #find the line intersection of the two vectors. 
    def line_intersection(p1, p2, p3, p4):
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



    #load camera

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080/2)


    #make a var for last homo if we fuck it up and need a backup
    last_H = None

    while True:
        #get frame from cam
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect markers and store corners and ids in arrays
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            #makes np shit into normal array
            ids = ids.flatten()
            detected_markers = {}
            #for every marker we detected
            for i in range(len(ids)):
                marker_id = ids[i]
                #if marker is one of our accepted ones
                if marker_id in marker_map:
                    #this gets an array of tuples, with 0 being top left, so on and so forth
                    c = corners[i][0]
                    #if its the top left one, get c[0], etc.
                    if marker_map[marker_id] == "tl":
                        selected_corner = c[0]
                    elif marker_map[marker_id] == "tr":
                        selected_corner = c[1]
                    elif marker_map[marker_id] == "br":
                        selected_corner = c[2]
                    elif marker_map[marker_id] == "bl":
                        selected_corner = c[3]
                    #dictionary where the corner coords will be saved as "tl" or "tr" and whatever
                    detected_markers[marker_map[marker_id]] = selected_corner
            #if we have all four corners
            if len(detected_markers) == 4:
                #order them to match the tl, tr, br, bl order for homo
                ordered = order_from_ids(detected_markers)
                #run the full thing
                frame, last_H = performWarping(frame, ref, ordered)
            elif len(detected_markers) >= 2:
                #if we only got two, lets see if they're opposite diagonal corners
                if find_corner(corners) is not None:
                    #if so, we estimate the other corners and warp based on it
                    frame, last_H = performWarping(frame, ref, find_corner(corners))
            elif last_H is not None:
                #if all of it didnt work, we use the last homo from eons past
                frame, _ = performWarping(frame, ref, None, last_H)

            #nice debug drawing
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        #small box to show
        cv2.imshow("Live Projection", frame)
        #idk what the fuck this does, it just came w the template from opencv
        if cv2.waitKey(1) & 0xFF == 27:
            break
    #donezo
    cap.release()
    cv2.destroyAllWindows()



# c[0] top left
# c[1] top right
# c[2] bottom right
# c[3] bottom left
