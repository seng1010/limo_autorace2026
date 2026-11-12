"""Read-only live lane inspector, rendered in a separate desktop process.

No ROS publishers, camera drivers, recording, or vehicle controls live here.
Snapshots come directly from the running lane pipeline, never a second detector.
"""
import math
import multiprocessing as mp
import queue
import time

import cv2
import numpy as np


VIEW_NAMES = {
    'raw': 'Original camera', 'overlay': 'Camera + ROI / path',
    'bev': 'BEV + path', 'windows': 'Sliding windows',
    'binary': 'Combined mask', 'color': 'Color mask', 'gradient': 'Gradient mask',
}
DEFAULT_VIEWS = ('raw', 'overlay', 'bev', 'windows')


def _draw_polyline(image, points, color, dashed=False):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    # Reject singular projections. Clip each segment before passing to OpenCV.
    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        if dashed and (index // 3) % 2:
            continue
        if not np.all(np.isfinite([a, b])) or np.max(np.abs([a, b])) > 1e6:
            continue
        visible, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                         tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)))
        if visible:
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)


def path_layers(packet, camera=False):
    state = packet['state']
    if not state.get('valid') or packet.get('left_fit') is None or packet.get('right_fit') is None:
        return {}
    height = packet['bev'].shape[0]
    ys = np.linspace(0, height - 1, 100)
    left = np.column_stack((np.polyval(packet['left_fit'], ys), ys))
    right = np.column_stack((np.polyval(packet['right_fit'], ys), ys))
    layers = dict(left=left, right=right, center=(left + right) * .5)
    if camera:
        layers = {name: cv2.perspectiveTransform(points.astype(np.float32)[None],
                  packet['inverse_matrix'])[0] for name, points in layers.items()}
    return layers


def render_view(packet, kind, options):
    camera = kind == 'overlay'
    source = packet.get('camera' if camera else kind)
    if source is None:
        return None
    result = source.copy()
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if kind not in ('overlay', 'bev'):
        return result
    if camera and options.get('roi', True):
        cv2.polylines(result, [np.rint(packet['roi']).astype(np.int32)], True, (0, 220, 220), 2)
    layers = path_layers(packet, camera)
    recovered = packet['state'].get('reference', 'both')
    for name, color in (('left', (255, 120, 0)), ('right', (0, 120, 255)), ('center', (0, 255, 0))):
        if name in layers and options.get(name, True):
            inferred = (name == 'left' and recovered == 'right') or (name == 'right' and recovered == 'left')
            _draw_polyline(result, layers[name], color, dashed=inferred)
    target = packet.get('target_bev_xy')
    if layers and target is not None and options.get('target', True):
        point = np.float32([[target]])
        if camera:
            point = cv2.perspectiveTransform(point, packet['inverse_matrix'])
        x, y = point[0, 0]
        if np.isfinite(x + y) and 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            cv2.circle(result, (round(float(x)), round(float(y))), 7, (255, 0, 255), -1)
    return result


def tile(image, label, width, height):
    output = np.zeros((height, width, 3), np.uint8)
    if image is not None:
        ratio = min(width / image.shape[1], max(1, height - 28) / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * ratio)), max(1, int(image.shape[0] * ratio))))
        x, y = (width - resized.shape[1]) // 2, 28 + (height - 28 - resized.shape[0]) // 2
        output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    else:
        cv2.putText(output, 'No image', (10, 60), 0, .6, (100, 100, 255), 1)
    cv2.putText(output, label, (10, 20), 0, .50, (240, 240, 240), 1, cv2.LINE_AA)
    return output


def render_grid(packet, selected, options, width=1000, height=700, columns=0):
    width, height = max(320, width), max(240, height)
    selected = [kind for kind in selected if kind in VIEW_NAMES]
    if not selected:
        return tile(None, 'Select a view on the right', width, height)
    columns = min(len(selected), columns or (1 if len(selected) == 1 else 2))
    rows = math.ceil(len(selected) / columns)
    tile_w, tile_h = width // columns, max(60, height // rows)
    grid = np.zeros((tile_h * rows, tile_w * columns, 3), np.uint8)
    for i, kind in enumerate(selected):
        y, x = (i // columns) * tile_h, (i % columns) * tile_w
        grid[y:y + tile_h, x:x + tile_w] = tile(render_view(packet, kind, options), VIEW_NAMES[kind], tile_w, tile_h)
    return grid


def status_text(packet):
    s = packet['state']
    return '\n'.join([
        'COMMAND OUTPUT: ' + ('ENABLED' if s['commands_enabled'] else 'OFF (manual inspection)'),
        f"mode={s['mode']} reference={s['reference']}",
        f"valid={s['valid']} usable={s['usable']}",
        f"confidence={s['confidence']:.2f}",
        f"L: {s['left_type']}", f"R: {s['right_type']}",
        f"lateral error: {s['lateral']:+.3f}",
        f"heading: {math.degrees(s['heading_rad']):+.1f} deg",
        f"bend span: {s['shape_span']:.3f} rad",
        f"curve weight: {s['curve_weight']:.2f}",
        f"curve control: {s['curve_control']}",
        f"calculated steer: {s['angle_cmd_deg']:+.1f} deg",
        f"calculated v: {s['lane_v']:.3f} m/s",
        f"calculated omega: {s['lane_omega']:+.3f} rad/s",
        f"/cmd_vel: {s.get('actual_cmd')}",
        f"cmd age at snapshot: {s.get('actual_cmd_age_sec')} s",
        f"lookahead: {s['lookahead']}",
        f"reason: {s['reason']}",
        '', 'Green = calculated center path', 'Blue/orange = left/right boundary',
        'Dashed boundary = inferred', 'Yellow = ROI; magenta = target',
        '', 'Steering is a command, NOT a sensor reading.',
        'Freeze/close affects display only.', 'It does NOT stop the vehicle.',
    ])


def run_window(frames, feedback, stop):
    """Tk is imported only in the spawned child (Ubuntu: python3-tk)."""
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        root.title('LIMO Live Lane Inspector — display only')
        root.geometry(f'{min(1420, root.winfo_screenwidth() - 60)}x{min(900, root.winfo_screenheight() - 100)}')
        root.minsize(960, 600)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        banner = tk.Label(root, text='Waiting for camera…', anchor='w', bg='#442b13', fg='white')
        banner.grid(row=0, column=0, columnspan=2, sticky='ew')
        image_label = tk.Canvas(root, bg='#111111', highlightthickness=0)
        image_label.grid(row=1, column=0, sticky='nsew')
        image_item = image_label.create_image(0, 0, anchor='nw')
        side_container = ttk.Frame(root)
        side_container.grid(row=1, column=1, sticky='ns')
        side_canvas = tk.Canvas(side_container, width=350, highlightthickness=0)
        side_scroll = ttk.Scrollbar(side_container, orient='vertical', command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side='right', fill='y')
        side_canvas.pack(side='left', fill='both', expand=True)
        sidebar = ttk.Frame(side_canvas, padding=10)
        side_canvas.create_window(0, 0, anchor='nw', window=sidebar, width=350)
        sidebar.bind('<Configure>', lambda event: side_canvas.configure(scrollregion=side_canvas.bbox('all')))
        selection = {key: tk.BooleanVar(value=key in DEFAULT_VIEWS) for key in VIEW_NAMES}
        options = {key: tk.BooleanVar(value=True) for key in ('roi', 'left', 'right', 'center', 'target')}
        frozen = tk.BooleanVar(value=False)
        column_choice = tk.StringVar(value='Auto')
        current, latest = None, None

        def preset(names):
            for name, value in selection.items():
                value.set(name in names)

        ttk.Label(sidebar, text='Views (select several)').pack(anchor='w')
        buttons = ttk.Frame(sidebar)
        buttons.pack(fill='x')
        for title, names in [('4 views', DEFAULT_VIEWS), ('Overlay', ('overlay',)), ('BEV', ('bev',))]:
            ttk.Button(buttons, text=title, command=lambda n=names: preset(n)).pack(side='left')
        for key, label in VIEW_NAMES.items():
            ttk.Checkbutton(sidebar, text=label, variable=selection[key]).pack(anchor='w')
        ttk.Label(sidebar, text='Layout columns').pack(anchor='w')
        ttk.Combobox(sidebar, values=('Auto', '1', '2', '3'), textvariable=column_choice,
                     state='readonly', width=10).pack(anchor='w')
        ttk.Separator(sidebar).pack(fill='x', pady=5)
        ttk.Label(sidebar, text='Overlay layers (camera / BEV)').pack(anchor='w')
        for key in options:
            ttk.Checkbutton(sidebar, text=key.upper(), variable=options[key]).pack(anchor='w')
        ttk.Checkbutton(sidebar, text='Freeze display (Space)', variable=frozen).pack(anchor='w', pady=5)
        status = ttk.Label(sidebar, text='Waiting for frames…', justify='left', wraplength=330)
        status.pack(anchor='w', fill='x')
        root.bind('<space>', lambda event: frozen.set(not frozen.get()))
        root.bind('1', lambda event: preset(('overlay',)))
        root.bind('2', lambda event: preset(('bev',)))
        root.bind('3', lambda event: preset(('windows',)))
        root.bind('4', lambda event: preset(DEFAULT_VIEWS))

        def close():
            stop.set()
            root.destroy()

        def callback_error(exc_type, exc, traceback):
            try:
                feedback.put_nowait(f'Live debugger unavailable: {exc_type.__name__}: {exc}')
            except queue.Full:
                pass
            close()

        root.report_callback_exception = callback_error

        def tick():
            nonlocal current, latest
            if stop.is_set():
                root.destroy()
                return
            # Read a bounded number; never spend forever draining a fast sender.
            for _ in range(2):
                try:
                    latest = frames.get_nowait()
                except queue.Empty:
                    break
            if not frozen.get():
                current = latest
            live_age = time.monotonic() - latest['state']['received_mono'] if latest else None
            if frozen.get():
                text, color = 'FROZEN DISPLAY — vehicle is NOT stopped', '#8a3e00'
            elif live_age is None or live_age > 1.0:
                text, color = 'STALE / WAITING — no recent camera result', '#8b2222'
            else:
                text, color = 'LIVE — display only', '#176136'
            age_text = f' | latest result age={live_age:.2f}s' if live_age is not None else ''
            banner.configure(text=text + age_text, bg=color)
            if current:
                grid = render_grid(current, [key for key, value in selection.items() if value.get()],
                                   {key: value.get() for key, value in options.items()},
                                   image_label.winfo_width(), image_label.winfo_height(),
                                   0 if column_choice.get() == 'Auto' else int(column_choice.get()))
                rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                ppm = f'P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n'.encode() + rgb.tobytes()
                photo = tk.PhotoImage(data=ppm, format='PPM')
                image_label.itemconfigure(image_item, image=photo)
                image_label.image = photo
                status.configure(text=status_text(current))
            root.after(100, tick)

        root.protocol('WM_DELETE_WINDOW', close)
        feedback.put_nowait('Live debugger window opened')
        root.after(100, tick)
        root.mainloop()
    except Exception as exc:
        try:
            feedback.put_nowait(f'Live debugger unavailable: {type(exc).__name__}: {exc}. '
                                'Use the NoMachine desktop and install python3-tk.')
        except queue.Full:
            pass
    finally:
        stop.set()


class LiveViewer:
    def __init__(self, fps=5.0):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('live_debug_fps must be finite and positive')
        self.fps = min(15., fps)
        self.next_frame = 0.
        context = mp.get_context('spawn')
        self.frames = context.Queue(maxsize=2)
        self.feedback = context.Queue(maxsize=4)
        self.stop = context.Event()
        self.process = context.Process(target=run_window, args=(self.frames, self.feedback, self.stop), daemon=True)
        self.process.start()

    def wants_frame(self):
        now = time.monotonic()
        if self.stop.is_set() or not self.process.is_alive() or now < self.next_frame:
            return False
        self.next_frame = now + 1.0 / self.fps
        return True

    def submit(self, packet):
        # Never wait for a GUI, disk, pipe, or queue slot in the driving callback.
        try:
            self.frames.put_nowait(packet)
        except queue.Full:
            pass

    def poll_message(self):
        try:
            return self.feedback.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        for channel in (self.frames, self.feedback):
            channel.cancel_join_thread()
            channel.close()
