import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import colorchooser, messagebox
from PIL import Image, ImageTk
import os
import time

class AirDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🖐 Air Draw - Hand Gesture App")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e2f")

        self.color = (0, 255, 0)
        self.thickness = 5
        self.prev_point = None
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.display_frame = None

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        tk.Label(root, text="Air Draw using Hand Gestures 🎨",
                 fg="#00ffcc", bg="#1e1e2f", font=("Segoe UI", 18, "bold")).pack(pady=10)

        self.video_label = tk.Label(root, bg="#1e1e2f")
        self.video_label.pack()

        control_frame = tk.Frame(root, bg="#1e1e2f")
        control_frame.pack(pady=15)

        tk.Button(control_frame, text="🎨 Pick Color", command=self.choose_color,
                  bg="#00ffcc", fg="black", font=("Segoe UI", 10, "bold"), width=12).grid(row=0, column=0, padx=10)
        tk.Button(control_frame, text="🧹 Clear Screen", command=self.clear_canvas,
                  bg="#00ffcc", fg="black", font=("Segoe UI", 10, "bold"), width=12).grid(row=0, column=1, padx=10)
        tk.Button(control_frame, text="💾 Save Artwork", command=self.save_canvas,
                  bg="#00ffcc", fg="black", font=("Segoe UI", 10, "bold"), width=12).grid(row=0, column=2, padx=10)
        tk.Button(control_frame, text="❌ Exit", command=self.close_app,
                  bg="red", fg="white", font=("Segoe UI", 10, "bold"), width=12).grid(row=0, column=3, padx=10)

        self.update_frame()

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Pick Drawing Color")
        if color_code[0]:
            self.color = tuple(map(int, color_code[0]))

    def clear_canvas(self):
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    def save_canvas(self):
        if self.display_frame is None:
            messagebox.showwarning("Error", "No frame available to save yet!")
            return

        save_dir = r"D:\MCA\SEM1\project\hand_gesture\drawings"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"art_{int(time.time())}.png"
        path = os.path.join(save_dir, filename)

        img_bgr = cv2.cvtColor(self.display_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
        messagebox.showinfo("Saved", f"Your artwork has been saved at:\n{path}")

    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            self.root.after(10, self.update_frame)
            return

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Mild background blur
        blurred = cv2.GaussianBlur(img, (9, 9), 0)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                if lm_list:
                    x1, y1 = lm_list[8]   # Index tip
                    x2, y2 = lm_list[12]  # Middle tip
                    dist = np.hypot(x2 - x1, y2 - y1)

                    if dist > 40:
                        if self.prev_point:
                            cv2.line(self.canvas, self.prev_point, (x1, y1), self.color, self.thickness)
                        self.prev_point = (x1, y1)
                    else:
                        self.prev_point = None

                    self.mp_draw.draw_landmarks(blurred, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Blend drawing with blurred background
        mask = self.canvas.astype(bool)
        blurred = cv2.GaussianBlur(blurred, (35, 35), 15)
        blurred[mask] = cv2.addWeighted(blurred, 0.5, self.canvas, 2.0, 0)[mask]

        self.display_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.display_frame)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = AirDrawApp(root)
    root.mainloop()
