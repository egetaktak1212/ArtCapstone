# Feature Detection and Projection for Artists

### By Mia McCarthy and Ege Taktak

Programming Languages Used:

* Python  
* Kivy

Libraries Used:

* torch                 2.11.0
* torchvision           0.26.0
* tqdm                  4.67.3
* scipy                 1.17.1
* pillow                12.2.0
* numpy                 1.26.4
* opencv-contrib-python 4.11.0.86
* opencv-python         4.13.0.92
* matplotlib            3.10.8
* mediapipe             0.10.21
* python                3.12.0
* Kivy                  2.3.1

Important note: mediapipe 0.10.21 is required and is only available on python 3.12.0 and below.

## Setting Up Canvas

1. Print out markers #1, #3, #4, and #6 from ArtCapstone\markers

2. Tape or glue to the corners of your canvas
   * #1 in the top left
   * #3 in the top right
   * #4 in the bottom right
   * #6 in the bottom left

## App Instructions

1. Run ArtCapstone\kivy\main2.py
   * this is the file that contains our UI, projection code, and its connections to the CNN
   * don't confuse with main.py, that is vestigial code I'm too sentimental to delete

2. Upload an image
   * Image must be a jpeg (jpg) or png

3. Adjust CNN results before pressing save and exit

4. Open Camera
   * Make sure to keep all four markers in frame (or two diagonally opposite at the very least)

5. Use on-screen buttons to make any transformations.

If open camera crashes, adjustments to the camera index OpenCV has access to might be necessary. For all machines, the default camera is 0. For you, if it's different, that's your fault.

For any testing information and documentation, refer to our submitted project report.

In case you want the link to our slideshow, here it is:
https://docs.google.com/presentation/d/1kEs2nwT0aqS8rImiCY0EcYJPvQftZmiUmAhzzugZ4zY/edit?usp=sharing
