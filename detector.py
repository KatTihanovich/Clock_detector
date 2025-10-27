import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np


class ClockFaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор циферблата и стрелок")

        self.img_path = None
        self.cv_img = None
        self.result_img = None
        self.display_img = None
        self.tk_img = None
        self.last_detection = None

        # --- UI ---
        frame_left_container = tk.Frame(root)
        frame_left_container.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Canvas + Scrollbar
        self.scroll_canvas = tk.Canvas(frame_left_container, width=300)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        scrollbar = tk.Scrollbar(frame_left_container, orient="vertical", command=self.scroll_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.bind('<Configure>',
                                lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.frame_left = tk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.frame_left, anchor='nw')

        frame_right = tk.Frame(root)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Button(self.frame_left, text="Open image", command=self.open_image).pack(fill=tk.X, pady=4)
        tk.Button(self.frame_left, text="Detect Face", command=self.detect).pack(fill=tk.X, pady=4)
        tk.Button(self.frame_left, text="Detect Hands", command=self.detect_hands).pack(fill=tk.X, pady=4)
        tk.Button(self.frame_left, text="Save result", command=self.save_result).pack(fill=tk.X, pady=4)

        tk.Label(self.frame_left, text="HoughCircles params").pack(pady=(10, 0))
        self.hough_dp = tk.DoubleVar(value=1.2)
        tk.Scale(self.frame_left, variable=self.hough_dp, from_=1.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
                 label="dp").pack(fill=tk.X)
        self.hough_minDist = tk.IntVar(value=100)
        tk.Scale(self.frame_left, variable=self.hough_minDist, from_=10, to=1000, orient=tk.HORIZONTAL,
                 label="minDist").pack(fill=tk.X)
        self.hough_param1 = tk.IntVar(value=100)
        tk.Scale(self.frame_left, variable=self.hough_param1, from_=10, to=300, orient=tk.HORIZONTAL,
                 label="param1").pack(fill=tk.X)
        self.hough_param2 = tk.IntVar(value=30)
        tk.Scale(self.frame_left, variable=self.hough_param2, from_=5, to=150, orient=tk.HORIZONTAL,
                 label="param2").pack(fill=tk.X)
        self.hough_minRadius = tk.IntVar(value=0)
        tk.Scale(self.frame_left, variable=self.hough_minRadius, from_=0, to=2000, orient=tk.HORIZONTAL,
                 label="minRadius").pack(fill=tk.X)
        self.hough_maxRadius = tk.IntVar(value=0)
        tk.Scale(self.frame_left, variable=self.hough_maxRadius, from_=0, to=2000, orient=tk.HORIZONTAL,
                 label="maxRadius").pack(fill=tk.X)

        tk.Label(self.frame_left, text="Contour/Ellipse params").pack(pady=(10, 0))
        self.min_contour_area = tk.IntVar(value=2000)
        tk.Scale(self.frame_left, variable=self.min_contour_area, from_=100, to=50000, orient=tk.HORIZONTAL,
                 label="min contour area").pack(fill=tk.X)

        tk.Label(self.frame_left, text="Центр допуск (+- пикс)").pack(pady=(10, 0))
        self.center_tolerance = tk.IntVar(value=100)
        tk.Scale(self.frame_left, variable=self.center_tolerance, from_=10, to=1000, orient=tk.HORIZONTAL,
                 label="tolerance").pack(fill=tk.X)

        # Параметры для детекции стрелок
        tk.Label(self.frame_left, text="Hand Detection params").pack(pady=(10, 0))

        # Режим детекции: тёмные или светлые стрелки
        self.hand_mode = tk.StringVar(value="auto")
        tk.Label(self.frame_left, text="Hand mode").pack()
        frame_mode = tk.Frame(self.frame_left)
        frame_mode.pack(fill=tk.X)
        tk.Radiobutton(frame_mode, text="Auto", variable=self.hand_mode, value="auto").pack(side=tk.LEFT)
        tk.Radiobutton(frame_mode, text="Dark", variable=self.hand_mode, value="dark").pack(side=tk.LEFT)
        tk.Radiobutton(frame_mode, text="Light", variable=self.hand_mode, value="light").pack(side=tk.LEFT)

        self.canny_low = tk.IntVar(value=50)
        tk.Scale(self.frame_left, variable=self.canny_low, from_=10, to=200, orient=tk.HORIZONTAL,
                 label="Canny low").pack(fill=tk.X)

        self.canny_high = tk.IntVar(value=150)
        tk.Scale(self.frame_left, variable=self.canny_high, from_=50, to=300, orient=tk.HORIZONTAL,
                 label="Canny high").pack(fill=tk.X)

        self.hough_threshold = tk.IntVar(value=50)
        tk.Scale(self.frame_left, variable=self.hough_threshold, from_=10, to=200, orient=tk.HORIZONTAL,
                 label="Line threshold").pack(fill=tk.X)

        self.min_line_length = tk.IntVar(value=30)
        tk.Scale(self.frame_left, variable=self.min_line_length, from_=10, to=200, orient=tk.HORIZONTAL,
                 label="Min line length").pack(fill=tk.X)

        self.canvas = tk.Canvas(frame_right, bg='grey')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self._on_resize)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff')])
        if not path:
            return
        self.img_path = path
        self.cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self._update_display_from_cv(self.cv_img)
        self.last_detection = None

    def _update_display_from_cv(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.display_img = pil
        self.result_img = pil.copy()
        self._show_on_canvas(self.display_img)

    def _show_on_canvas(self, pil_img):
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img, anchor=tk.CENTER)

    def _on_resize(self, event):
        if self.display_img:
            self._show_on_canvas(self.result_img)

    def save_result(self):
        if self.result_img is None:
            messagebox.showinfo('Info', 'Нет результата для сохранения')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            self.result_img.save(path)
            messagebox.showinfo('Saved', f'Сохранено: {path}')

    # Поиск окружности циферблата
    def detect(self):
        if self.cv_img is None:
            messagebox.showinfo('Info', 'Сначала откройте изображение')
            return

        img = self.cv_img.copy()
        h, w = img.shape[:2]
        center_img = np.array([w / 2.0, h / 2.0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

        out_img = img.copy()
        detections = []

        # --- Контуры ---
        th = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 9)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(500, self.min_contour_area.get()):
                continue
            if len(cnt) < 5:
                continue

            x, y, wc, hc = cv2.boundingRect(cnt)
            if x < 5 or y < 5 or (x + wc) > (w - 5) or (y + hc) > (h - 5):
                continue

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (MA, ma), angle = ellipse
            ratio = MA / ma if ma > 0 else 0
            if ratio > 1.3:
                continue

            dist = np.linalg.norm(center_img - np.array([cx, cy]))
            e_area = np.pi * (MA / 2.0) * (ma / 2.0)
            score = e_area / (1.0 + dist)
            detections.append({'type': 'ellipse', 'score': score, 'params': (cx, cy, MA, ma, angle)})

        # --- HoughCircles ---
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT,
                                   dp=max(1.0, float(self.hough_dp.get())),
                                   minDist=max(10, int(self.hough_minDist.get())),
                                   param1=int(self.hough_param1.get()),
                                   param2=int(self.hough_param2.get()),
                                   minRadius=int(self.hough_minRadius.get()),
                                   maxRadius=int(self.hough_maxRadius.get()) or 0)

        if circles is not None:
            circles = np.uint16(np.around(circles[0, :, :]))
            for (x, y, r) in circles:
                dist = np.linalg.norm(center_img - np.array([x, y]))
                score = (np.pi * (r ** 2)) / (1.0 + dist * 2.0)
                detections.append({'type': 'circle', 'score': score, 'params': (x, y, r)})

        if not detections:
            messagebox.showinfo('Result', 'Не найдено циферблата.')
            return

        best = max(detections, key=lambda d: d['score'])

        if best['type'] == 'circle':
            x, y, r = best['params']
            cv2.circle(out_img, (int(x), int(y)), int(r), (0, 255, 0), 3)
            cv2.circle(out_img, (int(x), int(y)), 3, (0, 0, 255), -1)
        else:
            cx, cy, MA, ma, angle = best['params']
            cv2.ellipse(out_img, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (255, 0, 0), 3)
            cv2.circle(out_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        self.last_detection = best
        self.result_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        self._show_on_canvas(self.result_img)
        messagebox.showinfo("Result", "Циферблат найден! Теперь можно нажать 'Detect Hands'.")

    #Поиск линий стрелок и вычисление времени
    def detect_hands(self):
        if self.cv_img is None or self.last_detection is None:
            messagebox.showinfo('Info', 'Сначала выполните детекцию циферблата.')
            return

        img = self.cv_img.copy()
        h, w = img.shape[:2]

        # Извлекаем параметры циферблата
        det = self.last_detection
        if det['type'] == 'circle':
            cx, cy, radius = map(int, det['params'])
        else:
            cx_f, cy_f, MA, ma, angle = det['params']
            cx, cy = int(round(cx_f)), int(round(cy_f))
            radius = int(min(MA, ma) / 2)

        # Создаём маску круга
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(radius * 0.95), 255, -1)

        # Исключаем центральный круг (втулка)
        hub_r = max(8, int(radius * 0.08))
        cv2.circle(mask, (cx, cy), hub_r, 0, -1)

        # --- Определяем режим (светлые или тёмные стрелки) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mode = self.hand_mode.get()
        if mode == "auto":
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(face_mask, (cx, cy), int(radius * 0.9), 255, -1)
            cv2.circle(face_mask, (cx, cy), int(radius * 0.3), 0, -1)

            face_brightness = cv2.mean(gray, mask=face_mask)[0]
            mode = "dark" if face_brightness > 127 else "light"
            print(f"Auto-detected mode: {mode} (face brightness: {face_brightness:.1f})")

        # --- Предобработка в зависимости от режима ---
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        if mode == "dark":
            gray_processed = cv2.bitwise_not(gray_masked)
        else:
            gray_processed = gray_masked.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_processed = clahe.apply(gray_processed)
        gray_blur = cv2.GaussianBlur(gray_processed, (5, 5), 0)

        # --- Детекция границ ---
        edges = cv2.Canny(gray_blur,
                          int(self.canny_low.get()),
                          int(self.canny_high.get()),
                          apertureSize=3)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # --- Детекция линий с помощью HoughLinesP ---
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(self.hough_threshold.get()),
            minLineLength=int(self.min_line_length.get()),
            maxLineGap=10
        )

        if lines is None:
            debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(debug_img, (cx, cy), radius, (0, 255, 0), 2)
            res_pil = Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            self.result_img = res_pil
            self._show_on_canvas(self.result_img)
            messagebox.showinfo("Result",
                                f"Линии не найдены.\nРежим: {mode}\n"
                                "Попробуйте изменить параметры Canny или Line threshold.\n"
                                "Показана карта границ.")
            return

        # --- Группировка и фильтрация линий ---
        hand_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1 = np.hypot(x1 - cx, y1 - cy)
            d2 = np.hypot(x2 - cx, y2 - cy)
            min_dist = min(d1, d2)
            max_dist = max(d1, d2)
            if min_dist > hub_r * 2.0:
                continue
            if max_dist < radius * 0.25:
                continue
            if d1 < d2:
                base = (x1, y1)
                tip = (x2, y2)
                length = d2
            else:
                base = (x2, y2)
                tip = (x1, y1)
                length = d1

            # Правильный расчет угла с учетом системы координат изображения
            dx = tip[0] - cx
            dy = -(tip[1] - cy)  # Инвертируем Y
            angle_rad = math.atan2(dy, dx)
            angle_deg = (90 - math.degrees(angle_rad)) % 360

            hand_candidates.append({
                'base': base,
                'tip': tip,
                'length': length,
                'angle': angle_deg,
                'line': (x1, y1, x2, y2)
            })

        if not hand_candidates:
            messagebox.showinfo("Result", f"Стрелки не найдены после фильтрации.\nРежим: {mode}")
            return

        # --- Группировка похожих линий ---
        grouped_hands = []
        hand_candidates = sorted(hand_candidates, key=lambda x: x['length'], reverse=True)
        for hand in hand_candidates:
            is_duplicate = False
            for group in grouped_hands:
                angle_diff = abs(hand['angle'] - group['angle'])
                angle_diff = min(angle_diff, 360 - angle_diff)
                if angle_diff < 15:
                    if hand['length'] > group['length']:
                        grouped_hands.remove(group)
                        grouped_hands.append(hand)
                    is_duplicate = True
                    break
            if not is_duplicate:
                grouped_hands.append(hand)

        grouped_hands = sorted(grouped_hands, key=lambda x: x['length'], reverse=True)[:2]

        if len(grouped_hands) < 2:
            messagebox.showinfo("Result", f"Найдена только одна стрелка.\nРежим: {mode}")
            return

        # --- УЛУЧШЕННОЕ определение минутной и часовой стрелки ---
        hand1, hand2 = grouped_hands[0], grouped_hands[1]

        # Вычисляем ожидаемое время для обоих вариантов
        minutes_v1 = (hand1['angle'] / 6.0) % 60
        hours_v1 = (hand2['angle'] / 30.0) % 12

        minutes_v2 = (hand2['angle'] / 6.0) % 60
        hours_v2 = (hand1['angle'] / 30.0) % 12

        # Вариант 1: hand1 - минутная, hand2 - часовая
        expected_hour_angle_v1 = ((hours_v1 + minutes_v1 / 60.0) * 30.0) % 360
        error_v1 = min(abs(expected_hour_angle_v1 - hand2['angle']),
                       360 - abs(expected_hour_angle_v1 - hand2['angle']))

        # Вариант 2: hand2 - минутная, hand1 - часовая
        expected_hour_angle_v2 = ((hours_v2 + minutes_v2 / 60.0) * 30.0) % 360
        error_v2 = min(abs(expected_hour_angle_v2 - hand1['angle']),
                       360 - abs(expected_hour_angle_v2 - hand1['angle']))

        # Проверяем разницу длин
        length_diff_ratio = abs(hand1['length'] - hand2['length']) / max(hand1['length'], hand2['length'])

        # Логика выбора:
        # 1. Если согласованность углов явно лучше в одном варианте (разница > 10°), выбираем его
        # 2. Иначе, если длины различаются > 5%, выбираем длинную как минутную
        # 3. Иначе полагаемся на меньшую ошибку

        if abs(error_v1 - error_v2) > 10:  # Явная разница в согласованности
            if error_v1 < error_v2:
                minute_hand = hand1
                hour_hand = hand2
                minutes = minutes_v1
                hours = hours_v1
                debug_info = f"По согласованности: v1 error={error_v1:.1f}° < v2 error={error_v2:.1f}°"
            else:
                minute_hand = hand2
                hour_hand = hand1
                minutes = minutes_v2
                hours = hours_v2
                debug_info = f"По согласованности: v2 error={error_v2:.1f}° < v1 error={error_v1:.1f}°"
        elif length_diff_ratio > 0.05:  # Длины различаются
            if hand1['length'] > hand2['length']:
                minute_hand = hand1
                hour_hand = hand2
                minutes = minutes_v1
                hours = hours_v1
                debug_info = f"По длине: hand1({hand1['length']:.0f}px) > hand2({hand2['length']:.0f}px)"
            else:
                minute_hand = hand2
                hour_hand = hand1
                minutes = minutes_v2
                hours = hours_v2
                debug_info = f"По длине: hand2({hand2['length']:.0f}px) > hand1({hand1['length']:.0f}px)"
        else:  # Почти одинаковые длины - только по углам
            if error_v1 < error_v2:
                minute_hand = hand1
                hour_hand = hand2
                minutes = minutes_v1
                hours = hours_v1
                debug_info = f"Одинаковые длины, по ошибке: v1={error_v1:.1f}° < v2={error_v2:.1f}°"
            else:
                minute_hand = hand2
                hour_hand = hand1
                minutes = minutes_v2
                hours = hours_v2
                debug_info = f"Одинаковые длины, по ошибке: v2={error_v2:.1f}° < v1={error_v1:.1f}°"

        # --- Рисование результата ---
        out = img.copy()
        cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
        cv2.circle(out, (cx, cy), hub_r, (0, 255, 0), 2)

        info_lines = [
            f"Найдено стрелок: 2",
            f"Режим: {mode}",
            f"Разница длин: {length_diff_ratio * 100:.1f}%"
        ]

        # Минутная стрелка - красная
        x1, y1, x2, y2 = minute_hand['line']
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.circle(out, minute_hand['tip'], 6, (0, 0, 255), -1)
        label_pos = (minute_hand['tip'][0] + 15, minute_hand['tip'][1] - 10)
        cv2.putText(out, "Minutes", label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        info_lines.append(f"Минутная: {minute_hand['angle']:.1f}° ({minute_hand['length']:.0f}px) → ~{minutes:.1f} мин")

        # Часовая стрелка - голубая
        x1, y1, x2, y2 = hour_hand['line']
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 4)
        cv2.circle(out, hour_hand['tip'], 6, (255, 255, 0), -1)
        label_pos = (hour_hand['tip'][0] + 15, hour_hand['tip'][1] - 10)
        cv2.putText(out, "Hours", label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2, cv2.LINE_AA)
        info_lines.append(f"Часовая: {hour_hand['angle']:.1f}° ({hour_hand['length']:.0f}px) → ~{hours:.1f} час")

        # Финальное время
        final_hours = int(hours) if hours != 0 else 12  # 0 часов = 12
        info_lines.append(f"Определённое время: {final_hours}:{int(minutes):02d}")
        info_lines.append(f"Debug: {debug_info}")

        res_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(res_pil)
        # Определённое время для надписи на изображении
        time_text = f"{final_hours}:{int(minutes):02d}"

        # Определяем размер изображения, чтобы разместить текст внизу
        img_w, img_h = res_pil.size
        text_x = img_w // 2 - 30
        text_y = img_h - 40

        # Рисуем время чёрным шрифтом
        draw.text((text_x, text_y), time_text, fill=(0, 0, 0))

        # Отображаем итог на канвасе
        self.result_img = res_pil
        self._show_on_canvas(self.result_img)

        # --- Информационное окно с подробностями ---
        messagebox.showinfo("Result", "\n".join(info_lines))


if __name__ == '__main__':
    root = tk.Tk()
    app = ClockFaceDetectorApp(root)
    root.geometry('1100x700')
    root.mainloop()
