from pathlib import Path

EVAL_POINTS_DIR = Path("/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/EVAL_POINTS")


plotting_parameters = {
    'font_scale': 3,
    'fig_width': 14,
    'fig_height': 14,
    'linewidth': 8,
    'markersize': 14,
    'blue_bgr': (0.7058823529411765*255, 0.4666666666666667*255, 0.12156862745098039*255),
    'orange_bgr': (0.054901960784313725*255, 0.4980392156862745*255, 1.0*255),
    'myblue': (0.415686275, 0.698039216, 0.831372549),
    'myorange': (1, 0.6, 0.2),
    'myyellow': (1, 0.706, 0),
    'mygreen': (0.443137255, 0.749019608, 0.431372549),
    'myred': (0.91372549, 0.282352941, 0.28627451),
    'mypurple': (0.678431373, 0.443137255, 0.709803922),
    'mygrey': (0.6, 0.6, 0.6),
    'mypink': (0.8902, 0.4667, 0.7608)
}

BENCHMARK = Path("/media/beverley/beverley_t7/SANGOHENKA-BENCHMARK")
LOGS_ROOT = Path('/home/beverley/Repos/sangohenka/logs')

TARGET_RESOLUTION = [640, 480]
FREQUENCY_GPS_HZ = 10.0

N_SUPERPOINT_KPS = 10

THRESHOLD_CORAL_PERCENTAGE = 95.0
THRESHOLD_NUM_OBSERVATIONS = 5

VISUALIZE = False
VERBOSE = False

OVERLEAF_PATH = Path("/home/beverley/Downloads") #Path("/home/beverley/Dropbox/Apps/Overleaf/[RSS26] GORRY BEVERLEY (1)/figures")