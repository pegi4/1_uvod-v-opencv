import cv2 as cv
import numpy as np

def resize_image(image, width, height):
    '''Resize the image to the specified width x height.'''
    pass

def process_image_with_boxes(image, box_width, box_height, skin_color) -> list:
    '''Iterate through the image in box-sized sections (box_width x box_height) and calculate the number of skin-colored pixels in each box.
    Boxes must not overlap!
    Returns a list of boxes, each containing the count of skin-colored pixels.
    Example: If the image has 25 boxes with 5 boxes per row, the list should be structured as
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]].
      Here, the first box has 1 skin pixel, the second 0, the third 0, the fourth 1, and the fifth 1.'''
    pass

def count_skin_colored_pixels(image, skin_color) -> int:
    '''Count the number of skin-colored pixels in the box.'''
    pass

def determine_skin_color(image, top_left, bottom_right) -> tuple:
    '''This function is called only once on the first image from the camera.
    Returns the skin color in the region defined by the bounding box (top_left, bottom_right).
    The calculation method is left to your imagination.'''

    print("Selecting color...")

    x1, y1 = top_left
    x2, y2 = bottom_right

    fild = image[y1:y2, x1:x2]
    mean = np.mean(fild, axis=(0, 1))
    print(f"Mean: {mean}")
    std = np.std(fild, axis=(0, 1))
    print(f"Std: {std}")
    k = 1.0

    lower_bound = np.clip(mean - k * std, 0, 255)
    upper_bound = np.clip(mean + k * std, 0, 255)

    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    return (lower_bound, upper_bound)

    pass

if __name__ == '__main__':

    camera = cv.VideoCapture(1)
    if not camera.isOpened():
        print('Camera does not work.')
        exit()

    skin_color = None

    while True:
        # Read the image from the camera
        ret, image = camera.read()
        
        if not ret:
            print('Error reading from camera.')
            camera.release()
            exit()

        image = cv.flip(image, 1)
        cv.imshow('Camera', image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            roi = cv.selectROI("Select the fild", image, fromCenter=False, showCrosshair=True)
            cv.destroyWindow("Select the fild")
            # Cordinates from roi: (x, y, width, height)
            x, y, w, h = roi
            upper_left = (x, y)
            down_right = (x + w, y + h)
            print(f"Top left corner: {upper_left}")
            print(f"Lower right corner: {down_right}")
            skin_color = determine_skin_color(image, upper_left, down_right)
            print(f"Skin color {skin_color}")
        elif key == ord('q'):
            camera.release()
            cv.destroyAllWindows()
            print('Camera closed.')
            exit(0)
            break

    # Zapremo okno
    camera.release()
    cv.destroyAllWindows()

    #Izračunamo barvo kože na prvi sliki

    #Zajemaj slike iz kamere in jih obdeluj     
    
    #Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
        #Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
        #Vprašanje 2: Kako prešteti število ljudi?

        #Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
        #in ne pozabite, da ni nujno da je škatla kvadratna.
    pass