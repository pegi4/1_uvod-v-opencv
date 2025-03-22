import cv2 as cv
import numpy as np

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
    k = 0.5

    lower_bound = np.clip(mean - k * std, 0, 255).reshape(1, 3)
    upper_bound = np.clip(mean + k * std, 0, 255).reshape(1, 3)

    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    return (lower_bound, upper_bound)

def resize_image(image, width, height):
    '''Resize the image to the specified width x height.'''
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def process_image_with_boxes(image, box_width, box_height, skin_color) -> list:
    '''Iterate through the image in box-sized sections (box_width x box_height) and calculate the number of skin-colored pixels in each box.
    Boxes must not overlap!
    Returns a list of boxes, each containing the count of skin-colored pixels.
    Example: If the image has 25 boxes with 5 boxes per row, the list should be structured as
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]].
      Here, the first box has 1 skin pixel, the second 0, the third 0, the fourth 1, and the fifth 1.'''
    
    h, w = image.shape[:2]
    num_boxes_y = h // box_height
    num_boxes_x = w // box_width
    
    # Ustvari matriko koordinat za zgornje leve kote škatel
    y_coords = np.arange(0, num_boxes_y * box_height, box_height)
    x_coords = np.arange(0, num_boxes_x * box_width, box_width)
    coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)  # Koordinate [x, y]
    
    # Preštej piksle za vsako škatlo
    counts = []
    for x, y in coords:
        podslika = image[y:y + box_height, x:x + box_width]
        st_pikslov = count_skin_colored_pixels(podslika, skin_color)
        counts.append(st_pikslov)
    
    # Preoblikuj v matriko
    return np.array(counts).reshape(num_boxes_y, num_boxes_x).tolist()

def count_skin_colored_pixels(image, skin_color) -> int:
    '''Count the number of skin-colored pixels in the box.'''
    lower_bound, upper_bound = skin_color
    mask = cv.inRange(image, lower_bound, upper_bound)
    return cv.countNonZero(mask)

if __name__ == '__main__':

    camera = cv.VideoCapture(1)
    if not camera.isOpened():
        print('Camera does not work.')
        exit()

    skin_color = None
    target_width, target_height = 220, 340
    box_width, box_height = 20, 20

    while True:
        # Read the image from the camera
        ret, image = camera.read()
        
        if not ret:
            print('Error reading from camera.')
            camera.release()
            exit()

        image = cv.flip(image, 1)

        if skin_color is None:
            cv.imshow('Camera', image)
        else:
            resized_image = resize_image(image, target_width, target_height)
            boxes = process_image_with_boxes(resized_image, box_width, box_height, skin_color)
            print(f"Boxes matrix: {boxes}")
            # Prikaz škatel na zmanjšani sliki
            for y, vrstica in enumerate(boxes):
                for x, st_pikslov in enumerate(vrstica):
                    if st_pikslov > 200:  # Prag za "kožo"
                        x1 = x * box_width
                        y1 = y * box_height
                        x2 = x1 + box_width
                        y2 = y1 + box_height
                        cv.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv.imshow('Camera', resized_image)

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
            break

    camera.release()
    cv.destroyAllWindows()
    print('Camera closed.')
    
    #Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
        #Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
        #Vprašanje 2: Kako prešteti število ljudi?

        #Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
        #in ne pozabite, da ni nujno da je škatla kvadratna.
    pass
