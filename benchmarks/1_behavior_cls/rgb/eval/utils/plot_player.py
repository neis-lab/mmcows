import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets

# Source link: https://stackoverflow.com/questions/46325447/animated-interactive-plot-using-matplotlib

class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.01, 0.92), run_status=True, interval_ms=1000, **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = run_status
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), interval=interval_ms,
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, 
                                           cache_frame_data=False, 
                                           **kwargs )    

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        print('start')
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        print('stop')
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        print('run forward')
        self.forwards = True
        self.start()
    def backward(self, event=None):
        print('run backward')
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        print('one forward')
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        print('one backward')
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.84, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="800%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '', 
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)
        
        
    # def set_pos(self, i, *args):
    #     self.i = int(self.slider.val)
    #     self.func(self.i, *args)

    # def update(self, i, *args):
    #     self.set_pos(i, *args)

# # ===============================================
# """ Main program from here """
# if __name__ == '__main__':
#     fig, ax = plt.subplots()
#     x = np.linspace(0,6*np.pi, num=100)
#     y = np.sin(x)

#     ax.plot(x,y)
#     point, = ax.plot([],[], marker="o", color="crimson", ms=15)

#     def update(i):
#         point.set_data(x[i],y[i])

#     ani = Player(fig, update, maxi=len(y)-1)

#     plt.show()
        
# ===============================================
""" Main program from here """
if __name__ == '__main__':
    date = '0725'

    from pen_model import *
    from projection import *
    from handle_xml import *

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    project_dir = os.path.dirname(current_dir)   # Get the parent directory (one level up)

    proj_data_dir = project_dir + '/data/cam_cal_mat'

    Proj_cam_list = read_projection_matrices(proj_data_dir, date)

    cam_coord = []
    for idx in range(4):
        proj_mat = Proj_cam_list[idx]
        cam_coord.append(cal_cam_coord(proj_mat))
        print(f"Cam {idx+1} loc: {cal_cam_coord(proj_mat).astype(int)}")
    cam_coord = np.asarray(cam_coord).reshape((4,3))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.clear()
        draw_pen(ax, cam_coord, legend=False)

    ani = Player(fig, update, maxi=1000) # "ani = " is a must

    plt.show()