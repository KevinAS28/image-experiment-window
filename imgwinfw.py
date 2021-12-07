import cv2
import traceback

class IMGWIN_MODE:
    SHOW = 1
    COMPARE = 2

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

        self.mode = IMGWIN_MODE.COMPARE
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
            
            if self.mode == IMGWIN_MODE.COMPARE:
                cv2.imshow('image output', output)
                cv2.imshow('image original', ori)
            elif self.mode == IMGWIN_MODE.SHOW:
                cv2.imshow('image output', output)
            else:
                raise ValueError(f'MODE {self.mode} is not a valid mode')

        except Exception as e:
            print('='*20)
            print('Error: ' + str(e))
            print(traceback.format_exc())
            print('='*20)

    def start(self, process_fun=None, onchange=lambda self, all_images, all_trackbars, from_main: print('Value changed'), mode=IMGWIN_MODE.COMPARE):
        self.mode = mode
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

    def add_button(self, text, onclick=lambda: print('Button clicked!')):
        cv2.createButton(text, onclick, None , cv2.QT_PUSH_BUTTON, 0)