"""Yuz kalite uygulamasi icin basit Tkinter arayuzu."""

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config
from application_settings import ApplicationSettings
from camera_sources import (
    build_droidcam_url,
    discover_camera_sources,
    parse_camera_source,
)
from camera_stream import LatestFrameCamera


class FaceQualityGui:
    BACKGROUND = "#111827"
    PANEL = "#1f2937"
    PANEL_LIGHT = "#273449"
    TEXT = "#f9fafb"
    MUTED = "#9ca3af"
    ACCENT = "#38bdf8"
    SUCCESS = "#34d399"
    WARNING = "#fbbf24"
    ERROR = "#fb7185"

    PRE_CONTROL_MODE = "pre_control"
    MODEL_MODE = "model"

    def __init__(self, root):
        self.root = root
        self.application = None
        self.active_mode = None
        self.settings = ApplicationSettings()
        self.camera = None
        self.camera_sources = {}
        self.preview_photo = None
        self.frame_job = None
        self.last_frame_number = 0

        self.quality_value = tk.StringVar(value="Bekleniyor")
        self.alignment_value = tk.StringVar(value="Bekleniyor")
        self.metrics_value = tk.StringVar(value="Yüz henüz analiz edilmedi")
        self.pre_control_value = tk.StringVar(value="FFT: Bekleniyor")
        self.connection_value = tk.StringVar(value="Kamera kapalı")
        self.camera_value = tk.StringVar()
        self.droidcam_address_value = tk.StringVar(
            value=self.settings.get_droidcam_address()
        )
        self.mode_value = tk.StringVar(value=self.PRE_CONTROL_MODE)
        self.mode_buttons = []

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self.refresh_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-q>", lambda _event: self.close())
        self.root.bind(
            "<KeyPress-s>",
            lambda _event: self.save_current_fft_sample(),
        )

    def _configure_window(self):
        self.root.title("Yüz Kalitesi Kontrolü")
        self.root.geometry("1180x980")
        self.root.minsize(900, 820)
        self.root.configure(bg=self.BACKGROUND)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Camera.TCombobox",
            fieldbackground=self.PANEL_LIGHT,
            background=self.PANEL_LIGHT,
            foreground=self.TEXT,
            arrowcolor=self.TEXT,
            bordercolor=self.PANEL_LIGHT,
            padding=8,
        )
        style.configure(
            "Controls.Vertical.TScrollbar",
            background=self.PANEL_LIGHT,
            troughcolor=self.BACKGROUND,
            bordercolor=self.BACKGROUND,
            arrowcolor=self.TEXT,
        )

    def _build_layout(self):
        header = tk.Frame(self.root, bg=self.BACKGROUND)
        header.pack(fill="x", padx=28, pady=(22, 14))

        title_group = tk.Frame(header, bg=self.BACKGROUND)
        title_group.pack(side="left")
        tk.Label(
            title_group,
            text="Yüz Kalitesi Kontrolü",
            bg=self.BACKGROUND,
            fg=self.TEXT,
            font=("Arial", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_group,
            text="Kamerayı seç, yüzünü hizala ve sonucu kaydet.",
            bg=self.BACKGROUND,
            fg=self.MUTED,
            font=("Arial", 10),
        ).pack(anchor="w", pady=(3, 0))

        tk.Button(
            header,
            text="Telefon kamerası nasıl kullanılır?",
            command=self.show_phone_help,
            bg=self.PANEL,
            fg=self.ACCENT,
            activebackground=self.PANEL_LIGHT,
            activeforeground=self.TEXT,
            relief="flat",
            padx=14,
            pady=9,
            cursor="hand2",
        ).pack(side="right")

        content = tk.Frame(self.root, bg=self.BACKGROUND)
        content.pack(fill="both", expand=True, padx=28, pady=(0, 26))
        content.grid_columnconfigure(0, weight=4)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        self._build_preview(content)
        self._build_controls(content)

    def _build_preview(self, parent):
        preview_panel = tk.Frame(parent, bg="#050b14")
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        preview_panel.grid_rowconfigure(0, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_panel,
            text="Kamera önizlemesi",
            bg="#050b14",
            fg=self.MUTED,
            font=("Arial", 16),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        connection_bar = tk.Label(
            preview_panel,
            textvariable=self.connection_value,
            bg=self.PANEL,
            fg=self.MUTED,
            anchor="w",
            padx=14,
            pady=9,
            font=("Arial", 9),
        )
        connection_bar.grid(row=1, column=0, sticky="ew")

    def _build_controls(self, parent):
        controls_container = tk.Frame(parent, bg=self.BACKGROUND)
        controls_container.grid(row=0, column=1, sticky="nsew")
        controls_container.grid_rowconfigure(0, weight=1)
        controls_container.grid_columnconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(
            controls_container,
            bg=self.PANEL,
            highlightthickness=0,
            borderwidth=0,
            yscrollincrement=20,
        )
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")

        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=self.controls_canvas.yview,
            style="Controls.Vertical.TScrollbar",
        )
        controls_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(7, 0),
        )
        self.controls_canvas.configure(
            yscrollcommand=controls_scrollbar.set
        )

        controls = tk.Frame(
            self.controls_canvas,
            bg=self.PANEL,
            padx=20,
            pady=20,
        )
        self.controls_window = self.controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )
        controls.bind(
            "<Configure>",
            self._update_controls_scroll_region,
        )
        self.controls_canvas.bind(
            "<Configure>",
            self._resize_controls_window,
        )
        controls.grid_columnconfigure(0, weight=1)

        self._section_title(controls, "Çalışma Modu").grid(
            row=0, column=0, sticky="w"
        )

        mode_frame = tk.Frame(controls, bg=self.PANEL)
        mode_frame.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        for text, value in (
            ("Model-free Pre-Control", self.PRE_CONTROL_MODE),
            ("Model Analizi", self.MODEL_MODE),
        ):
            button = tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_value,
                value=value,
                command=self._mode_changed,
                bg=self.PANEL,
                fg=self.TEXT,
                selectcolor=self.PANEL_LIGHT,
                activebackground=self.PANEL,
                activeforeground=self.ACCENT,
                font=("Arial", 10, "bold"),
                anchor="w",
            )
            button.pack(fill="x", pady=2)
            self.mode_buttons.append(button)

        self.mode_description = tk.Label(
            controls,
            bg=self.PANEL,
            fg=self.MUTED,
            justify="left",
            anchor="w",
            wraplength=270,
            font=("Arial", 9),
        )
        self.mode_description.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        self._separator(controls).grid(
            row=3, column=0, sticky="ew", pady=14
        )
        self._section_title(controls, "Kamera").grid(
            row=4, column=0, sticky="w"
        )

        camera_row = tk.Frame(controls, bg=self.PANEL)
        camera_row.grid(row=5, column=0, sticky="ew", pady=(8, 10))
        camera_row.grid_columnconfigure(0, weight=1)

        self.camera_combobox = ttk.Combobox(
            camera_row,
            textvariable=self.camera_value,
            style="Camera.TCombobox",
        )
        self.camera_combobox.grid(row=0, column=0, sticky="ew")
        self.camera_combobox.bind(
            "<<ComboboxSelected>>", self._camera_selection_changed
        )

        self._button(
            camera_row,
            "↻",
            self.refresh_cameras,
            self.PANEL_LIGHT,
            width=3,
        ).grid(row=0, column=1, padx=(8, 0))

        self.toggle_button = self._button(
            controls,
            "Kamerayı Başlat",
            self.toggle_camera,
            self.ACCENT,
            foreground="#082f49",
        )
        self.toggle_button.grid(row=6, column=0, sticky="ew")

        self._separator(controls).grid(
            row=7, column=0, sticky="ew", pady=14
        )
        self._section_title(controls, "DroidCam • Wi-Fi").grid(
            row=8, column=0, sticky="w"
        )
        tk.Label(
            controls,
            text="Telefondaki IP adresi • otomatik kaydedilir",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Arial", 9),
        ).grid(row=9, column=0, sticky="w", pady=(7, 4))
        self.droidcam_entry = tk.Entry(
            controls,
            textvariable=self.droidcam_address_value,
            bg=self.PANEL_LIGHT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            font=("Arial", 10),
        )
        self.droidcam_entry.grid(
            row=10, column=0, sticky="ew", ipady=8
        )
        self.droidcam_entry.bind(
            "<Return>", lambda _event: self.connect_droidcam()
        )
        self._button(
            controls,
            "Telefon Kamerasına Bağlan",
            self.connect_droidcam,
            self.SUCCESS,
            foreground="#052e16",
        ).grid(row=11, column=0, sticky="ew", pady=(8, 0))

        self._separator(controls).grid(
            row=12, column=0, sticky="ew", pady=14
        )
        self._section_title(controls, "Canlı Sonuç").grid(
            row=13, column=0, sticky="w"
        )

        self.quality_label = self._status_card(
            controls, "Görüntü kalitesi", self.quality_value
        )
        self.quality_label.master.grid(
            row=14, column=0, sticky="ew", pady=(10, 8)
        )
        self.alignment_label = self._status_card(
            controls, "Yüz hizalama", self.alignment_value
        )
        self.alignment_label.master.grid(row=15, column=0, sticky="ew")

        self.pre_control_label = self._status_card(
            controls,
            "Ön kontrol analizleri",
            self.pre_control_value,
            value_font_size=10,
        )
        self.pre_control_label.master.grid(
            row=16, column=0, sticky="ew", pady=(8, 0)
        )

        self.fft_capture_button = self._button(
            controls,
            "FFT Örneğini Kaydet (1 Kare)",
            self.save_current_fft_sample,
            self.ACCENT,
            foreground="#082f49",
        )
        self.fft_capture_button.grid(
            row=17, column=0, sticky="ew", pady=(8, 0)
        )

        tk.Label(
            controls,
            textvariable=self.metrics_value,
            bg=self.PANEL,
            fg=self.MUTED,
            justify="left",
            anchor="w",
            font=("Arial", 9),
        ).grid(row=18, column=0, sticky="ew", pady=(12, 0))

        controls.grid_rowconfigure(19, weight=1)
        self._separator(controls).grid(
            row=20, column=0, sticky="ew", pady=14
        )

        self.save_face_button = self._button(
            controls,
            "Yüz Fotoğrafını Kaydet",
            self.save_face,
            self.PANEL_LIGHT,
        )
        self.save_face_button.grid(
            row=21, column=0, sticky="ew", pady=(0, 8)
        )
        self.save_json_button = self._button(
            controls,
            "JSON Sonucunu Kaydet",
            self.save_json,
            self.PANEL_LIGHT,
        )
        self.save_json_button.grid(row=22, column=0, sticky="ew")
        self._bind_controls_mousewheel(controls)
        self._mode_changed()

    def _update_controls_scroll_region(self, _event=None):
        self.controls_canvas.configure(
            scrollregion=self.controls_canvas.bbox("all")
        )

    def _resize_controls_window(self, event):
        self.controls_canvas.itemconfigure(
            self.controls_window,
            width=event.width,
        )

    def _bind_controls_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._scroll_controls, add="+")
        widget.bind("<Button-4>", self._scroll_controls, add="+")
        widget.bind("<Button-5>", self._scroll_controls, add="+")
        for child in widget.winfo_children():
            self._bind_controls_mousewheel(child)

    def _scroll_controls(self, event):
        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            direction = -1 if event.delta > 0 else 1

        self.controls_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _section_title(self, parent, text):
        return tk.Label(
            parent,
            text=text,
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Arial", 12, "bold"),
        )

    def _button(
        self,
        parent,
        text,
        command,
        background,
        foreground=None,
        width=None,
    ):
        options = {
            "text": text,
            "command": command,
            "bg": background,
            "fg": foreground or self.TEXT,
            "activebackground": self.ACCENT,
            "activeforeground": "#082f49",
            "relief": "flat",
            "padx": 12,
            "pady": 10,
            "font": ("Arial", 10, "bold"),
            "cursor": "hand2",
        }
        if width is not None:
            options["width"] = width

        return tk.Button(
            parent,
            **options,
        )

    def _status_card(
        self,
        parent,
        title,
        value_variable,
        value_font_size=13,
    ):
        card = tk.Frame(parent, bg=self.PANEL_LIGHT, padx=13, pady=11)
        tk.Label(
            card,
            text=title,
            bg=self.PANEL_LIGHT,
            fg=self.MUTED,
            font=("Arial", 9),
        ).pack(anchor="w")
        value_label = tk.Label(
            card,
            textvariable=value_variable,
            bg=self.PANEL_LIGHT,
            fg=self.MUTED,
            font=("Arial", value_font_size, "bold"),
            justify="left",
        )
        value_label.pack(anchor="w", pady=(3, 0))
        return value_label

    def _separator(self, parent):
        return tk.Frame(parent, bg="#374151", height=1)

    def refresh_cameras(self):
        selected_source = self.camera_value.get()
        discovered = discover_camera_sources()
        self.camera_sources = dict(discovered)
        values = list(self.camera_sources)
        self.camera_combobox["values"] = values

        if selected_source in values:
            self.camera_value.set(selected_source)
        elif values:
            self.camera_value.set(values[0])
        else:
            self.camera_value.set(str(config.CAMERA_INDEX))
            self.connection_value.set(
                "Kamera bulunamadı; indeks veya yayın adresi yazabilirsiniz."
            )

    def toggle_camera(self):
        if self.camera is None:
            self.start_camera()
        else:
            self.stop_camera()

    def _mode_changed(self):
        selected_mode = self.mode_value.get()

        if self.camera is not None:
            return

        if (
            self.application is not None
            and self.active_mode != selected_mode
        ):
            self.application.shutdown()
            self.application = None
            self.active_mode = None

        if selected_mode == self.PRE_CONTROL_MODE:
            self.mode_description.configure(
                text=(
                    "Yalnızca sabit yüz kılavuzundaki FFT analizi çalışır. "
                    "MediaPipe ve model dosyası yüklenmez."
                )
            )
            self.quality_value.set("Bu modda kapalı")
            self.alignment_value.set("Bu modda kapalı")
            self.metrics_value.set(
                "Yüzünü ekrandaki sabit kılavuzun içine yerleştir."
            )
            self.pre_control_value.set("FFT: Bekleniyor")
            save_state = "disabled"
            fft_capture_state = "normal"
        else:
            self.mode_description.configure(
                text=(
                    "MediaPipe ile yüz kalitesi ve hizalama çalışır. "
                    "FFT pre-control çalışmaz."
                )
            )
            self.quality_value.set("Bekleniyor")
            self.alignment_value.set("Bekleniyor")
            self.metrics_value.set("Yüz henüz analiz edilmedi")
            self.pre_control_value.set("Bu modda kapalı")
            save_state = "normal"
            fft_capture_state = "disabled"

        self.quality_label.configure(fg=self.MUTED)
        self.alignment_label.configure(fg=self.MUTED)
        self.pre_control_label.configure(fg=self.MUTED)
        self.save_face_button.configure(state=save_state)
        self.save_json_button.configure(state=save_state)
        self.fft_capture_button.configure(state=fft_capture_state)

    def _create_application_for_selected_mode(self):
        selected_mode = self.mode_value.get()
        if selected_mode == self.PRE_CONTROL_MODE:
            from model_free_pre_control_application import (
                ModelFreePreControlApplication,
            )

            application = ModelFreePreControlApplication()
        else:
            from face_quality_application import FaceQualityApplication

            application = FaceQualityApplication()

        self.active_mode = selected_mode
        return application

    def connect_droidcam(self):
        try:
            source = build_droidcam_url(
                self.droidcam_address_value.get()
            )
        except ValueError as error:
            messagebox.showwarning("DroidCam", str(error))
            return

        if self.camera is not None:
            self.stop_camera()

        phone_address = self.droidcam_address_value.get().strip()
        label = "DroidCam — " + phone_address
        self.camera_sources[label] = source
        self.camera_combobox["values"] = list(self.camera_sources)
        self.camera_value.set(label)
        if self.start_camera():
            address_was_saved = self.settings.save_droidcam_address(
                phone_address
            )
            if not address_was_saved:
                messagebox.showwarning(
                    "DroidCam ayarı",
                    "Kamera bağlandı ancak IP adresi kaydedilemedi.",
                )

    def start_camera(self):
        selected_value = self.camera_value.get()
        source = parse_camera_source(selected_value, self.camera_sources)
        if source == "":
            messagebox.showwarning("Kamera", "Lütfen bir kamera seçin.")
            return False

        camera = LatestFrameCamera(source)
        if not camera.is_opened():
            camera.release()
            self.connection_value.set("Kamera açılamadı")
            messagebox.showerror(
                "Kamera açılamadı",
                "Seçilen kamera kullanılamıyor. Başka bir kamera seçin "
                "veya başka bir uygulamanın kamerayı kullanmadığından emin olun.",
            )
            return False

        if self.application is None:
            try:
                self.application = self._create_application_for_selected_mode()
            except Exception as error:
                camera.release()
                messagebox.showerror(
                    "Mod başlatılamadı",
                    "Seçilen analiz modu başlatılamadı:\n" + str(error),
                )
                return False

        self.camera = camera
        self.last_frame_number = 0
        self.camera.start()
        self.connection_value.set("Kamera bağlı • Canlı analiz çalışıyor")
        self.toggle_button.configure(
            text="Kamerayı Durdur",
            bg=self.ERROR,
            fg="#4c0519",
        )
        self.camera_combobox.configure(state="disabled")
        for mode_button in self.mode_buttons:
            mode_button.configure(state="disabled")
        self._schedule_next_frame()
        return True

    def stop_camera(self):
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
            self.frame_job = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.preview_label.configure(image="", text="Kamera önizlemesi")
        self.preview_photo = None
        self.connection_value.set("Kamera kapalı")
        self.toggle_button.configure(
            text="Kamerayı Başlat",
            bg=self.ACCENT,
            fg="#082f49",
        )
        self.camera_combobox.configure(state="normal")
        for mode_button in self.mode_buttons:
            mode_button.configure(state="normal")

    def _schedule_next_frame(self):
        self.frame_job = self.root.after(5, self._update_frame)

    def _update_frame(self):
        self.frame_job = None
        if self.camera is None:
            return

        frame_was_read, camera_frame, frame_number = (
            self.camera.read_latest(self.last_frame_number)
        )
        if not frame_was_read or camera_frame is None:
            if self.camera.has_failed():
                self.stop_camera()
                messagebox.showerror(
                    "Kamera bağlantısı kesildi",
                    "Kameradan görüntü alınamıyor. Bağlantıyı kontrol edin.",
                )
                return
            self._schedule_next_frame()
            return

        self.last_frame_number = frame_number
        try:
            frame_result = self.application.process_frame(camera_frame)
        except Exception as error:
            self.stop_camera()
            messagebox.showerror(
                "Analiz hatası",
                "Kamera karesi işlenemedi:\n" + str(error),
            )
            return
        self.application.latest_face_image = frame_result.face_image
        self.application.latest_analysis_result = frame_result.analysis_result

        self._show_frame(frame_result.display_frame)
        if frame_result.analysis_result is not None:
            self._show_result(frame_result.analysis_result)
        self._show_pre_control_results()
        self._schedule_next_frame()

    def save_current_fft_sample(self):
        if (
            self.mode_value.get() != self.PRE_CONTROL_MODE
            or self.application is None
        ):
            print("FFT ornegi yalnizca model-free pre-control modunda kaydedilir.")
            return False
        sample_was_saved = self.application.save_fft_sample()
        if sample_was_saved:
            self.connection_value.set(
                "FFT örneği kaydedildi • Canlı analiz çalışıyor"
            )
        else:
            self.connection_value.set(
                "Kaydedilemedi • Geçerli yüz crop'u ve FFT sonucu yok"
            )
        return sample_was_saved

    def _show_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        available_width = max(self.preview_label.winfo_width(), 320)
        available_height = max(self.preview_label.winfo_height(), 240)
        image.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def _show_result(self, result):
        quality_text, quality_color = self._translate_status(
            result.quality_status
        )
        alignment_text, alignment_color = self._translate_status(
            result.alignment_status
        )
        self.quality_value.set(quality_text)
        self.quality_label.configure(fg=quality_color)
        self.alignment_value.set(alignment_text)
        self.alignment_label.configure(fg=alignment_color)

        if result.alignment is None:
            self.metrics_value.set("Yüz henüz analiz edilemedi")
            return

        alignment = result.alignment
        self.metrics_value.set(
            "Yaw: %.1f°   Pitch: %.1f°   Roll: %.1f°\n"
            "Ağız açıklığı: %.2f"
            % (
                alignment.yaw_degrees,
                alignment.pitch_degrees,
                alignment.roll_degrees,
                alignment.mouth_open_score,
            )
        )

    def _translate_status(self, status):
        translations = {
            "ACCEPTABLE": ("Uygun", self.SUCCESS),
            "UNACCEPTABLE": ("Uygun değil", self.WARNING),
            "NO_FACE": ("Yüz bulunamadı", self.ERROR),
            "MULTIPLE_FACES": ("Birden fazla yüz", self.ERROR),
        }
        return translations.get(status, (status, self.MUTED))

    def _show_pre_control_results(self):
        if self.application is None:
            return

        results = self.application.latest_pre_control_results
        if not results:
            if self.mode_value.get() == self.MODEL_MODE:
                self.pre_control_value.set("Bu modda kapalı")
            else:
                self.pre_control_value.set("Henüz sonuç yok")
            self.pre_control_label.configure(fg=self.MUTED)
            return

        lines = []
        overall_color = self.SUCCESS
        for result_key, result in results.items():
            display_name = getattr(
                result,
                "display_name",
                result_key,
            )
            status = getattr(result, "status", "Bilinmiyor")
            status_text, status_color = self._translate_pre_control_status(
                status
            )
            score_text = ""
            score = getattr(result, "score", None)
            if score is not None:
                score_text = " • %d/100" % round(score)

            line = "%s: %s%s" % (
                display_name,
                status_text,
                score_text,
            )
            if getattr(result, "warning", ""):
                attack_type = getattr(result, "attack_type", "Bilinmiyor")
                line += "\nOlası saldırı: " + attack_type
            lines.append(line)

            if status_color == self.ERROR:
                overall_color = self.ERROR
            elif status_color == self.WARNING and overall_color != self.ERROR:
                overall_color = self.WARNING

        self.pre_control_value.set("\n".join(lines))
        self.pre_control_label.configure(fg=overall_color)

    def _translate_pre_control_status(self, status):
        translations = {
            "Normal": ("Normal", self.SUCCESS),
            "Suspicious": ("Şüpheli", self.ERROR),
            "Possible Screen Replay": (
                "Olası ekran tekrar saldırısı",
                self.ERROR,
            ),
            "Analysis Uncertain": ("Analiz belirsiz", self.WARNING),
            "Unavailable": ("Kullanılamıyor", self.WARNING),
            "Normal frequency structure": ("Normal", self.SUCCESS),
            "Suspicious frequency pattern": ("Şüpheli", self.ERROR),
            "Possible replay attack": ("Olası ekran saldırısı", self.ERROR),
            "Possible printed-photo attack": (
                "Olası basılı fotoğraf saldırısı",
                self.ERROR,
            ),
            "Analysis uncertain": ("Belirsiz", self.WARNING),
        }
        return translations.get(status, (status, self.MUTED))

    def save_face(self):
        if (
            self.application is None
            or self.application.latest_face_image is None
        ):
            messagebox.showwarning(
                "Yüz kaydedilemedi",
                "Önce kamerada tek bir yüzün net biçimde göründüğünden emin olun.",
            )
            return
        self.application.save_face_image()
        messagebox.showinfo(
            "Kaydedildi",
            "Yüz fotoğrafı outputs klasörüne kaydedildi.",
        )

    def save_json(self):
        if (
            self.application is None
            or self.application.latest_analysis_result is None
        ):
            messagebox.showwarning(
                "Sonuç yok", "Kaydetmek için önce kamerayı çalıştırın."
            )
            return
        self.application.save_json_result()
        messagebox.showinfo(
            "Kaydedildi",
            "Analiz sonucu outputs klasörüne kaydedildi.",
        )

    def _camera_selection_changed(self, _event):
        if self.camera is not None:
            self.stop_camera()

    def show_phone_help(self):
        messagebox.showinfo(
            "Telefon kamerası",
            "Android: Telefonunuz destekliyorsa USB bağlantı seçeneklerinden "
            "‘Webcam’ modunu açın. Ardından bu ekranda yenile düğmesine basın.\n\n"
            "DroidCam: Telefon ve bilgisayarı aynı Wi-Fi ağına bağlayın. "
            "Telefondaki DroidCam uygulamasında görünen WiFi IP adresini "
            "DroidCam alanına girip ‘Telefon Kamerasına Bağlan’ düğmesine basın.\n\n"
            "iPhone: Linux'ta yalnızca kabloyu bağlamak genellikle yeterli "
            "değildir. Telefonu bir sanal kameraya dönüştüren bir uygulama/"
            "sürücü kurduktan sonra oluşan kamerayı bu listeden seçebilirsiniz.\n\n"
            "Ağ kamerası uygulaması kullanıyorsanız açılır kutuya uygulamanın "
            "verdiği video adresini de yazabilirsiniz.",
        )

    def close(self):
        self.stop_camera()
        if self.application is not None:
            self.application.shutdown()
        cv2.destroyAllWindows()
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    FaceQualityGui(root)
    root.mainloop()
