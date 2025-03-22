import cv2 as cv
import numpy as np

def zmanjsaj_sliko(slika, sirina, visina):
    '''Zmanjšaj sliko na velikost sirina x visina.'''
    pass

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    '''Sprehodi se skozi sliko v velikosti škatle (sirina_skatle x visina_skatle) in izračunaj število pikslov kože v vsaki škatli.
    Škatle se ne smejo prekrivati!
    Vrne seznam škatel, s številom pikslov kože.
    Primer: Če je v sliki 25 škatel, kjer je v vsaki vrstici 5 škatel, naj bo seznam oblike
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]]. 
      V tem primeru je v prvi škatli 1 piksel kože, v drugi 0, v tretji 0, v četrti 1 in v peti 1.'''
    pass

def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    '''Prestej število pikslov z barvo kože v škatli.'''
    pass

def doloci_barvo_koze(slika,levo_zgoraj,desno_spodaj) -> tuple:
    '''Ta funkcija se kliče zgolj 1x na prvi sliki iz kamere. 
    Vrne barvo kože v območju ki ga definira oklepajoča škatla (levo_zgoraj, desno_spodaj).
      Način izračuna je prepuščen vaši domišljiji.'''
    pass

if __name__ == '__main__':

    camera = cv.VideoCapture(1)

    if not camera.isOpened():
        print('Camera does not work.')
        exit()

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
        elif key == ord('q'):
            camera.realise()
            cv.destroyAllWindows()
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