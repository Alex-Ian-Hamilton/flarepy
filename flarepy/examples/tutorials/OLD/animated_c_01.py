import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons



def make_fig():
    pass




def update(val):
    amp = sld_width_upper.val
    freq = sld_width_lower.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()


def reset(event):
    sld_width_lower.reset()
    sld_width_upper.reset()


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()





if __name__ == '__main__':
    # The data we need for plotting
    points = 1000
    widths = np.arange(100)
    data = np.random.rand(points) +0.000001#np.arange(100)
    cwt_image=np.random.random((points,len(widths)))
    peaks=np.random.randint(2,points, size=10)
    ridge_lines=[[np.array([1, 2, 3, 4, 5, 6, 7]), np.array([100, 101, 99, 100, 101, 102, 106])]]
    filtered_ridge_lines=[[np.array([1, 2, 3, 4, 5, 6, 7]), np.array([200, 201, 199, 200, 201, 202, 206])]]


    """
    fig = plot_cwt_components(show=['linear','log','image','ridges','image/ridges'],
                            data=ser_xrsb_raw_int_60S_box5,
                            cwt_image=cwt_dat,
                            peaks=df_peaks_cwt,
                            ridge_lines=ridge_lines,
                            filtered_ridge_lines=filtered,
                            title='CWT Peak Detection Steps - GOES - '+str_start[0:10],
                            savepath='CWT Peak Detection Steps - GOES - '+str_start[0:10]+'.png')
    """



    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    s = a0*np.sin(2*np.pi*f0*t)
    l, = plt.plot(t, s, lw=2, color='red')
    plt.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    #axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    #axamp = plt.axes([0.25, 0.15, 0.65, 0.03])

    # Axes (positions) for the sliders
    ax_width_lower = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_width_upper = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_width_steps = plt.axes([0.25, 0.20, 0.65, 0.03])

    # Make parameter sliders
    sld_width_lower = Slider(ax_width_lower, 'width lower', 1, 100, valinit=50)
    sld_width_upper = Slider(ax_width_upper, 'width upper', 1, 101, valinit=100.0)
    sld_width_steps = Slider(ax_width_steps, 'width steps', 1, 100, valinit=0.0, valfmt='%1.0f')

    # Dettect slider click
    sld_width_lower.on_changed(update)
    sld_width_upper.on_changed(update)
    sld_width_steps.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15])
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    radio.on_clicked(colorfunc)


    plt.show()
