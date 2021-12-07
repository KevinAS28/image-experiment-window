# from genericpath import exists
import cv2
# import sys
import numpy as np
# from threading import Thread
import os
import traceback
import copy

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
# from skimage.io import imread, imsave

class ImageExperimentWindow:
    exist_window_names = dict()
    previous_values = dict()
    trackbars = dict()
    global_vars = dict()
    def __init__(self, image_list, window_name='control output', auto_terminate=True, wait_time = 33):
        if window_name in self.exist_window_names:
            if auto_terminate:
                self.exist_window_names[window_name].close_window()
            else:
                raise ValueError(f'Window name {window_name} is still exist')

        self.exist_window_names[window_name] = self
        self.window_name = window_name
        self.wait_time = wait_time
        self.image_list = image_list

        cv2.namedWindow(window_name)

    def get_window_name(self):
        return self.window_name

    def change_value(self, name, val):
        p_val = cv2.getTrackbarPos(name)
        self.previous_values[name] = p_val
        cv2.setTrackbarPos(name, self.window_name, val)


    def process(self, all_images, all_trackbars):
        return self.image_list[0], []

    def process_internal(self, process_fun, all_trackbars):
        try:
            ori, output, vars = process_fun(self, self.image_list, all_trackbars)
            self.global_vars['from_main'] = vars
            
            cv2.imshow('image output', output)
            cv2.imshow('image original', ori)
        except Exception as e:
            print('='*20)
            print('Error: ' + str(e))
            print(traceback.format_exc())
            print('='*20)

    def start(self, process_fun=None, onchange=lambda self, all_images, all_trackbars, from_main: print('Value changed')):
        process_fun = self.process if process_fun is None else process_fun
        all_trackbars = {self.trackbars[name]: cv2.getTrackbarPos(name, self.window_name) for name in self.trackbars}
        self.process_internal(process_fun, all_trackbars)
        while 1:
            all_trackbars = {self.trackbars[name]: cv2.getTrackbarPos(name, self.window_name) for name in self.trackbars}
            process_fun = self.process if process_fun is None else process_fun

            for name, key in self.trackbars.items():
                if all_trackbars[key]!=self.previous_values[name]:
                    onchange(self.image_list, all_trackbars, self.global_vars['from_main'])
                    for name, key in self.trackbars.items():
                        value = all_trackbars[key]
                        print(f'{name}: {value}', end=' | ')
                        self.previous_values[name] = value

                    self.process_internal(process_fun, all_trackbars)
    
                    break

       
            if cv2.waitKey(self.wait_time) & 0xFF == ord('q'):
                self.close_window()
                break

    def close_window(self):
        cv2.destroyWindow(self.window_name)

    def add_trackbar(self, name, range, track_key=None, default_pos=0, function=lambda x: None):
        if not (name in self.trackbars):
            cv2.createTrackbar(name, self.window_name , *range, function)
            cv2.setTrackbarPos(name, self.window_name, default_pos)
            key = track_key if not (track_key is None) else name[0]+name[-1]
            self.trackbars[name] = key
            self.previous_values[name] = default_pos
            return self.trackbars[name]
        else:
            raise ValueError(f'Duplicate trackbar name: {name}')


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def get_color(image, vmin, smax):
    image = cv2.resize(cv2.bitwise_not(image), (1000, 600))

    lower = np.array([0, 0, vmin])
    upper = np.array([111, smax, 255])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)

    h, w = image.shape[:2]
    crop_img = image[ int(h/10):int(h-int(h/10)),  int(w/10):int(w-(int(w/10)*3))]
    br0 = str(cv2.mean(cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV))[2])
    cr0 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).std()

    h, w = image.shape[:2]
    crop_img = output[ int(h/10):int(h-int(h/10)),  int(w/10):int(w-(int(w/10)*3))]
    br1 = str(cv2.mean(cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV))[2])
    cr1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).std()

    output = cv2.bitwise_not(output)

    return output, br0, cr0, br1, cr1


def get_contrast(image):
    return image.std()
    
def get_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.mean(cv2.split(image)[2])[0]


def nothing(x):
    pass





# Load in image

# def process(trackbars):


all_images = [cv2.resize((cv2.imread('ktp_data_new/'+i)), (1000, 600)) for i in os.listdir('ktp_data_new')]

w0 = ImageExperimentWindow(all_images)
w0.add_trackbar('image index', (0, len(all_images)-1), 'ii')
# w0.add_trackbar('threshold percent', (1, 101), 'tp')
# w0.add_trackbar('vmin', (0, 255), 'vmin', 168)
# w0.add_trackbar('smax', (0, 255), 'smax', 172)
# w0.add_trackbar('alpha', (0, 300), 'a', 0)
# w0.add_trackbar('beta', (0, 100), 'b', 0)
# w0.add_trackbar('blur', (0, 10), 'bl', 0)
# w0.add_trackbar('C', (0, 30), 'c', 0)
w0.add_trackbar('block size', (0, 100), 'block', 40)
w0.add_trackbar('delta', (0, 100), 'delta', 0)
w0.add_trackbar('brightness', (0, 200), 'br', 100)
w0.add_trackbar('start d:b', (50, 300), 'srdb', 70)
w0.add_trackbar('step d:b', (0, 100), 'stdb', 1)
w0.add_trackbar('Nothing | Auto | Blur | Sharp', (0, 3), 'nabs', 1)
w0.add_trackbar('blur00', (0, 25), 'b00', 5)
w0.add_trackbar('blur01', (0, 25), 'b01', 5)
w0.add_trackbar('umks0', (0, 25), 'umks0', 5)
w0.add_trackbar('umks1', (0, 25), 'umks1', 5)
w0.add_trackbar('ums', (0, 100), 'ums', 10)
w0.add_trackbar('uma', (0, 100), 'uma', 10)
# w0.add_trackbar('denoising0', (0, 100), 'deno0', 10)
# w0.add_trackbar('denoising1', (0, 100), 'deno1', 10)
# w0.add_trackbar('denoising2', (0, 100), 'deno2', 7)
# w0.add_trackbar('denoising3', (0, 100), 'deno3', 21)
# w0.add_trackbar('delta_b', (0, 100), 'delta', 25)

def ktp_parse_button():
    pass

cv2.createButton('Parse KTP', ktp_parse_button, None , cv2.QT_PUSH_BUTTON, 0)

def process(a_i, a_t):
    image = a_i[a_t['ii']]
    vmin, smax = a_t['vmin'], a_t['smax']
    blur = a_t['bl']
    if (blur%2)==0:
        blur+=1
    

    image1, alpha, beta = automatic_brightness_and_contrast(image, clip_hist_percent=a_t['tp'])
    image1 = cv2.GaussianBlur(image1, (blur, blur), 0)
    image1, br0, cr0, br1, cr1 = get_color(image1, vmin, smax)
    return image1, [image, image1]

def process1(a_i, a_t):
    image = a_i[a_t['ii']]
    blur = a_t['bl']
    if (blur%2)==0:
        blur+=1
    
    vmin, smax = a_t['vmin'], a_t['smax']
    alpha, beta = a_t['a']/100, a_t['b']
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    image = cv2.GaussianBlur(image, (blur, blur), 0)
    image, br0, cr0, br1, cr1 = get_color(image, vmin, smax)
    # image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, image.shape[0] if (image.shape[0]%2)==1 else image.shape[0]-1, a_t['c'])
    return image, [a_i[a_t['ii']], image]

def onchange1(a_i, a_t, fm):
    output = fm[0]
    ori = fm[1]
    print('Image writed')
    cv2.imwrite('auto.jpeg', output)

def onchange(a_i, a_t, fm):
    i0 = fm[0]
    i1 = fm[1]

    hsv0 = cv2.cvtColor(i0, cv2.COLOR_RGB2HSV)
    gray0 = cv2.cvtColor(i0, cv2.COLOR_RGB2GRAY)
    hsv1 = cv2.cvtColor(i1, cv2.COLOR_RGB2HSV)
    gray1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)

    br0 = get_brightness(hsv0) 
    cr0 = get_contrast(gray0)
    br1 = get_brightness(hsv1) 
    cr1 = get_contrast(gray1)

    print(f'br0: {br0} | cr0: {cr0} | br1: {br1} | cr1: {cr1}\n')
    cv2.imwrite('auto.jpeg', i1)

def process2(a_i, a_t):
    img = a_i[a_t['ii']]
    yen_threshold = threshold_yen(img)
    bright = rescale_intensity(img, (0, yen_threshold), (0, a_t['y']))

    return bright, [img, bright]


def adjust_gamma(image, gamma=1.2):
    
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    
    return cv2.LUT(image, table)








def preprocess(image):
    image = cv2.medianBlur(image, 3)
    return 255 - image




def postprocess(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def get_block_index(image_shape, yx, block_size): 
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)













def adaptive_median_threshold(img_in, delta):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < delta] = 255
    kernel = np.ones((3,3),np.uint8)
    img_out = 255 - cv2.dilate(255 - img_out,kernel,iterations = 2)
    return img_out




def block_image_process(image, block_size, delta):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx], delta)
    return out_image


def process_image(img, block_size, delta):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, block_size, delta)
    image_out = postprocess(image_out)
    return image_out


def sigmoid(x, orig, rad):
    k = np.exp((x - orig) * 5 / rad)
    return k / (k + 1.)



def combine_block(img_in, mask):
    
    
    img_out = np.zeros_like(img_in)
    img_out[mask == 255] = 255
    fimg_in = img_in.astype(np.float32)

    
    
    
    idx = np.where(mask == 0)
    if idx[0].shape[0] == 0:
        img_out[idx] = img_in[idx]
        return img_out

    
    
    lo = fimg_in[idx].min()
    hi = fimg_in[idx].max()
    v = fimg_in[idx] - lo
    r = hi - lo

    
    
    img_in_idx = img_in[idx]
    ret3,th3 = cv2.threshold(img_in[idx],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    
    bound_value = np.min(img_in_idx[th3[:, 0] == 255])
    bound_value = (bound_value - lo) / (r + 1e-5)
    f = (v / (r + 1e-5))
    f = sigmoid(f, bound_value + 0.05, 0.2)

    
    img_out[idx] = (255. * f).astype(np.uint8)
    return img_out



def combine_block_image_process(image, mask, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = combine_block(
                image[block_idx], mask[block_idx])
    return out_image




def combine_postprocess(image):
    return image


def combine_process(img, mask):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_out = combine_block_image_process(image_in, mask, 20)
    image_out = combine_postprocess(image_out)
    return image_out

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    
    mn = np.min(v[v <= lim])
    mx = np.max(v[v <= lim])
    if mn+value < 0:
        value = -mn
        
    if mx+value > 255:
        value = 255-mx
    

    value = np.uint8(value)
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



def onchange_write(a_i, a_t, fm):
    img = fm[1]
    cv2.imwrite('auto.jpeg', img)

    img = fm[0]
    h, w = img.shape[:2]
    crop_img = img[ int(h/10):int(h-int(h/10)),  int(w/10):int(w-(int(w/10)*3))]
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    br0 = get_brightness(hsv) 
    cr0 = get_contrast(gray)

    srdb = a_t['srdb']
    stdb = a_t['stdb']
    dist_b = br0-srdb
    delta = (stdb*dist_b)/2

    print(f'br0 = {br0}, cr0 = {cr0}, srdb = {srdb}, stdb = {stdb}, delta = {delta}')

w0.start(process4, onchange_write)
