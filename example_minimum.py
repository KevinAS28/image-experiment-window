import copy
import os

from imgwinfw import *

img_dir = 'imgs/'
all_images = [cv2.imread(img_dir+i) for i in os.listdir(img_dir)]

w0 = ImageExperimentWindow(all_images)
w0.add_trackbar('image index', (0, len(all_images)-1), 'ii')
w0.add_button('Click me', lambda: print('Button customed!'))

def onchange(all_images, all_trackbars, from_main):
    img = from_main[1]
    print('changed!')

def process(self, all_images, all_trackbars):
    
    img = cv2.resize(all_images[all_trackbars['ii']], (600, 250))
    ori = copy.deepcopy(img)
    
    img = cv2.putText(img, f'IMG', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    img1 = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
    return ori, img1, [ori, img1]

w0.start(process, onchange, mode=IMGWIN_MODE.SHOW)