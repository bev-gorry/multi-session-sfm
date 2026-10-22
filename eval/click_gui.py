"""
Torch-free pieces of the annotation GUI (numpy, OpenCV, matplotlib only), shared by eval/select_eval_points.py (full
annotation, Linux) and eval/reclick_eval_points.py (re-clicking fixed query points, also on a Mac): the zoomed
correspondence refiner, the epipolar line clipped to the view, and the homography / epipolar transfer of a point.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]

blue   = [c / 255.0 for c in BLUE]
yellow = [c / 255.0 for c in YELLOW]
pink   = [c / 255.0 for c in PINK]
grey   = [c / 255.0 for c in GREY]

def epipolar_line_in_window(F, u0, v0, x_min, x_max, y_min, y_max):
    """Epipolar line of (u0,v0) clipped to a viewing window. Returns two endpoints or None."""
    line = cv2.computeCorrespondEpilines(
        np.array([[[u0, v0]]], dtype=np.float32), 1, F)[0, 0]
    a, b, c = line
    pts = []
    if abs(b) > 1e-9:  # y = -(a x + c)/b at window x-edges
        for x in (x_min, x_max):
            y = -(a * x + c) / b
            if y_min <= y <= y_max:
                pts.append((x, y))
    if abs(a) > 1e-9:  # x = -(b y + c)/a at window y-edges
        for y in (y_min, y_max):
            x = -(b * y + c) / a
            if x_min <= x <= x_max:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    # take the two most distant intersection points (handles corner duplicates)
    pts = np.array(pts)
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return pts[i], pts[j]


class ZoomedPointRefiner:
    """
    Interactive refinement of one correspondence.
    Left: zoomed image0 around the clicked point (reference).
    Right: zoomed image1 around the H/F estimate, with epipolar line.
      - left click: set corrected point
      - scroll: zoom in/out (both panels)
      - enter: accept current point
      - escape: skip this point
      - q, or closing the window: quit (self.quit = True; eval/reclick_eval_points.py saves and stops)
    """
    def __init__(self, image0, image1, u0, v0, H, F, zoom=200, title=""):
        self.u0, self.v0 = u0, v0
        self.F = F
        self.zoom = zoom
        self.h1, self.w1 = image1.shape[:2]

        # H estimate + F refinement as starting point
        self.u_h, self.v_h = transfer_point_with_homography(H, u0, v0)
        if F is not None:
            self.u_gt, self.v_gt = transfer_H_refined_by_F(H, F, u0, v0)
        else:
            self.u_gt, self.v_gt = self.u_h, self.v_h
        self.accepted = None
        self.decided = False   # enter or escape pressed
        self.quit = False      # q pressed or window closed without a decision

        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 8))
        self.axs[0].imshow(image0)
        self.axs[1].imshow(image1)
        self.axs[0].plot(u0, v0, 'o', ms=10, mfc='none', mec='green', mew=2)
        self.axs[0].plot(u0, v0, '+', ms=14, color='green')
        self.axs[0].set_title("image0 — clicked point")

        # H estimate (cyan) and current/refined point (red)
        self.axs[1].plot(self.u_h, self.v_h, 'x', ms=10, color='blue',
                         label='H estimate')
        (self.marker,) = self.axs[1].plot(self.u_gt, self.v_gt, 'o', ms=10,
                                          mfc='none', mec='red', mew=2,
                                          label='current')
        (self.cross,) = self.axs[1].plot(self.u_gt, self.v_gt, '+', ms=14,
                                         color='red')
        self.epi_artist = None
        self.axs[1].legend(loc='upper right')
        self.axs[1].set_title("image1 — click to correct | scroll zoom | enter accept | esc skip")
        if title:
            self.fig.suptitle(title)

        self._set_views()
        self._draw_epiline()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show()

    def _set_views(self):
        z = self.zoom
        self.axs[0].set_xlim(self.u0 - z, self.u0 + z)
        self.axs[0].set_ylim(self.v0 + z, self.v0 - z)   # inverted y for images
        self.axs[1].set_xlim(self.u_gt - z, self.u_gt + z)
        self.axs[1].set_ylim(self.v_gt + z, self.v_gt - z)

    def _draw_epiline(self):
        if self.F is None:
            return
        if self.epi_artist is not None:
            self.epi_artist.remove()
            self.epi_artist = None
        x_min, x_max = sorted(self.axs[1].get_xlim())
        y_min, y_max = sorted(self.axs[1].get_ylim())
        seg = epipolar_line_in_window(self.F, self.u0, self.v0,
                                      x_min, x_max, y_min, y_max)
        if seg is not None:
            (x1, y1), (x2, y2) = seg
            (self.epi_artist,) = self.axs[1].plot(
                [x1, x2], [y1, y2], '--', color=yellow, lw=1.5, alpha=0.8)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.axs[1] or event.button != 1:
            return
        self.u_gt, self.v_gt = event.xdata, event.ydata
        self.marker.set_data([self.u_gt], [self.v_gt])
        self.cross.set_data([self.u_gt], [self.v_gt])
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        self.zoom *= 0.8 if event.button == 'up' else 1.25
        self.zoom = float(np.clip(self.zoom, 30, max(self.h1, self.w1)))
        self._set_views()
        self._draw_epiline()

    def on_key(self, event):
        if event.key == 'enter':
            self.accepted = (float(self.u_gt), float(self.v_gt))
            self.decided = True
            plt.close(self.fig)
        elif event.key == 'escape':
            self.accepted = None
            self.decided = True
            plt.close(self.fig)
        elif event.key == 'q':
            self.quit = True
            plt.close(self.fig)

    def on_close(self, event):
        if not self.decided:
            self.quit = True

def transfer_H_refined_by_F(H, F, u, v):
    u_h, v_h = transfer_point_with_homography(H, u, v)
    line = cv2.computeCorrespondEpilines(
        np.array([[[u, v]]], dtype=np.float32), 1, F)[0, 0]
    a, b, c = line
    t = (a * u_h + b * v_h + c) / (a * a + b * b)
    return u_h - a * t, v_h - b * t

def transfer_point_with_homography(H, u_clicked, v_clicked):
    pt0 = np.array([[[u_clicked, v_clicked]]], dtype=np.float32)
    pt1 = cv2.perspectiveTransform(pt0, H)[0, 0]
    u_gt, v_gt = pt1[0], pt1[1]
    return u_gt, v_gt
