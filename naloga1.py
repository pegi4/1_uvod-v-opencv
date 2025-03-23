import cv2 as cv
import numpy as np
import time

def determine_skin_color(image, top_left, bottom_right) -> tuple:
    print("Selecting color...")

    x1, y1 = top_left
    x2, y2 = bottom_right

    fild = image[y1:y2, x1:x2]
    mean = np.mean(fild, axis=(0, 1))
    print(f"Mean: {mean}")
    std = np.std(fild, axis=(0, 1))
    print(f"Std: {std}")
    k = 1

    lower_bound = np.clip(mean - k * std, 0, 255).reshape(1, 3)
    upper_bound = np.clip(mean + k * std, 0, 255).reshape(1, 3)

    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    return (lower_bound, upper_bound)

def resize_image(image, width, height):
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def process_image_with_boxes(image, box_width, box_height, skin_color) -> list:
    h, w = image.shape[:2]
    num_boxes_y = h // box_height  # 17 for 340/20
    num_boxes_x = w // box_width   # 11 for 220/20
    
    y_coords = np.arange(0, num_boxes_y * box_height, box_height)
    x_coords = np.arange(0, num_boxes_x * box_width, box_width)
    coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    
    candidates = []
    counts = np.zeros((num_boxes_y, num_boxes_x), dtype=int)
    for idx, (x, y) in enumerate(coords):
        podslika = image[y:y + box_height, x:x + box_width]
        st_pikslov = count_skin_colored_pixels(podslika, skin_color)
        counts[idx // num_boxes_x, idx % num_boxes_x] = st_pikslov
        if st_pikslov > 200:  # Threshold for skin
            candidates.append((x, y, x + box_width, y + box_height))
    
    def merge_boxes(candidates):
        if not candidates:
            return []
        
        merged_boxes = []
        while candidates:
            current = list(candidates.pop(0))  # [x1, y1, x2, y2]
            merged = True
            while merged:
                merged = False
                i = 0
                while i < len(candidates):
                    box = candidates[i]

                    x_overlap = current[0] <= box[2] + box_width and current[2] >= box[0] - box_width
                    y_overlap = current[1] <= box[3] + box_height and current[3] >= box[1] - box_height
                    if x_overlap and y_overlap:
                        current[0] = min(current[0], box[0])
                        current[1] = min(current[1], box[1])
                        current[2] = max(current[2], box[2])
                        current[3] = max(current[3], box[3])
                        candidates.pop(i)
                        merged = True
                    else:
                        i += 1
            merged_boxes.append(((current[0], current[1]), (current[2], current[3])))
        
        return merged_boxes
    
    faces = merge_boxes(candidates)
    return faces

def count_skin_colored_pixels(image, skin_color) -> int:
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
    color_thickness = 2

    while True:

        start_time = time.time()

        ret, captured_image = camera.read()
        if not ret:
            print('Error reading from camera.')
            break

        captured_image = cv.flip(captured_image, 1)
        image = resize_image(captured_image, target_width, target_height)

        if skin_color is None:
            cv.imshow('Camera', image)
        else:
            faces = process_image_with_boxes(image, box_width, box_height, skin_color)
            #print(f"Detected faces: {len(faces)}")
            for (x1, y1), (x2, y2) in faces:
                cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), color_thickness)

            fps = 1.0 / (time.time() - start_time)
            cv.putText(image, f"FPS: {int(fps)}", (10, 20), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), color_thickness)
            
            cv.putText(image, f"Detected faces: {len(faces)}", (10, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), color_thickness)
            cv.imshow('Camera', image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            roi = cv.selectROI("Select the field", image, fromCenter=False, showCrosshair=True)
            cv.destroyWindow("Select the field")

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