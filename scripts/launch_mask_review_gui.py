"""Launch a Cellpose-like desktop GUI for model comparison and mask correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from astroseg.gui import (
    MaskReviewDataset,
    load_instance_mask,
    paint_instance_disk,
    render_instance_overlay,
    save_corrected_instances,
)
from astroseg.io import load_ome_tiff
from astroseg.metrics import instance_segmentation_metrics


def _model_specification(value: str) -> tuple[str, Path]:
    """Parse ``DISPLAY_NAME=MASK_DIRECTORY`` without constraining spaces in paths."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model masks must use NAME=DIRECTORY")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Model masks must use non-empty NAME=DIRECTORY")
    return name.strip(), Path(raw_path.strip())


class MaskReviewApplication:
    """Tk-based instance viewer with Cellpose-style overlays and basic editing."""

    def __init__(
        self,
        root: object,
        dataset: MaskReviewDataset,
        correction_directory: Path,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.dataset = dataset
        self.correction_directory = correction_directory
        self.current_image_id = ""
        self.current_model = dataset.model_names[0]
        self.microscopy = None
        self.labels: np.ndarray | None = None
        self.selected_label = 0
        self.dirty = False
        self.undo_stack: list[np.ndarray] = []
        self.stroke_active = False
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_anchor: tuple[int, int] | None = None
        self.photo = None
        self.display_scale = 1.0
        self.display_origin = (0.0, 0.0)
        self.render_pending = False

        root.title("AstroSeg Mask Review")
        root.geometry("1480x920")
        root.minsize(1040, 680)
        root.configure(bg="#17191d")
        root.protocol("WM_DELETE_WINDOW", self._close)

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.TFrame", background="#202329")
        style.configure("Dark.TLabel", background="#202329", foreground="#f2f3f5")
        style.configure("Title.TLabel", background="#202329", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("Dark.TCheckbutton", background="#202329", foreground="#f2f3f5")
        style.configure("Dark.TRadiobutton", background="#202329", foreground="#f2f3f5")

        self.model_var = tk.StringVar(value=self.current_model)
        self.channel_var = tk.StringVar()
        self.alpha_var = tk.DoubleVar(value=0.58)
        self.outline_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value="select")
        self.brush_var = tk.IntVar(value=12)
        self.selected_var = tk.StringVar(value="Selected cell: none")
        self.status_var = tk.StringVar(value="Loading…")
        self.metrics_var = tk.StringVar(value="Ground-truth metrics: unavailable")

        outer = ttk.Frame(root, style="Dark.TFrame")
        outer.pack(fill="both", expand=True)

        sidebar = ttk.Frame(outer, style="Dark.TFrame", width=285)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Images", style="Title.TLabel").pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        self.image_list = tk.Listbox(
            sidebar,
            bg="#111318",
            fg="#e9eaec",
            selectbackground="#315f93",
            selectforeground="#ffffff",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#343841",
            font=("Segoe UI", 10),
        )
        self.image_list.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        for image_id in dataset.image_ids:
            self.image_list.insert("end", image_id)
        self.image_list.bind("<<ListboxSelect>>", self._select_image_from_list)

        navigation = ttk.Frame(sidebar, style="Dark.TFrame")
        navigation.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(navigation, text="◀ Previous", command=lambda: self._navigate(-1)).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(navigation, text="Next ▶", command=lambda: self._navigate(1)).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )

        center = ttk.Frame(outer, style="Dark.TFrame")
        center.pack(side="left", fill="both", expand=True)
        toolbar = ttk.Frame(center, style="Dark.TFrame")
        toolbar.pack(fill="x", padx=8, pady=8)
        ttk.Label(toolbar, text="Model", style="Dark.TLabel").pack(side="left", padx=(4, 4))
        self.model_combo = ttk.Combobox(
            toolbar,
            textvariable=self.model_var,
            values=dataset.model_names,
            state="readonly",
            width=18,
        )
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self._change_model)
        ttk.Label(toolbar, text="Channel", style="Dark.TLabel").pack(side="left", padx=(14, 4))
        self.channel_combo = ttk.Combobox(
            toolbar, textvariable=self.channel_var, state="readonly", width=16
        )
        self.channel_combo.pack(side="left")
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_render())
        ttk.Label(toolbar, text="Overlay", style="Dark.TLabel").pack(side="left", padx=(14, 4))
        ttk.Scale(
            toolbar,
            from_=0.0,
            to=1.0,
            variable=self.alpha_var,
            command=lambda _value: self._schedule_render(),
            length=120,
        ).pack(side="left")
        ttk.Checkbutton(
            toolbar,
            text="Outlines",
            variable=self.outline_var,
            command=self._schedule_render,
            style="Dark.TCheckbutton",
        ).pack(side="left", padx=10)
        ttk.Button(toolbar, text="Reset view", command=self._reset_view).pack(side="right")
        ttk.Button(toolbar, text="Compare models", command=self._show_comparison).pack(
            side="right", padx=6
        )

        self.canvas = tk.Canvas(
            center,
            bg="#090a0d",
            highlightthickness=1,
            highlightbackground="#343841",
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self._schedule_render())
        self.canvas.bind("<ButtonPress-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", lambda _event: setattr(self, "pan_anchor", None))
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_by(1.15, event))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_by(1 / 1.15, event))

        status = ttk.Frame(center, style="Dark.TFrame")
        status.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(status, textvariable=self.status_var, style="Dark.TLabel").pack(side="left")
        ttk.Label(status, textvariable=self.metrics_var, style="Dark.TLabel").pack(side="right")

        editor = ttk.Frame(outer, style="Dark.TFrame", width=260)
        editor.pack(side="right", fill="y")
        editor.pack_propagate(False)
        ttk.Label(editor, text="Mask editor", style="Title.TLabel").pack(
            anchor="w", padx=14, pady=(14, 8)
        )
        for text, value in (
            ("Select cell", "select"),
            ("Paint selected cell", "paint"),
            ("Erase", "erase"),
            ("Merge clicked → selected", "merge"),
        ):
            ttk.Radiobutton(
                editor,
                text=text,
                value=value,
                variable=self.mode_var,
                style="Dark.TRadiobutton",
            ).pack(anchor="w", padx=14, pady=2)
        ttk.Separator(editor).pack(fill="x", padx=12, pady=12)
        ttk.Label(editor, textvariable=self.selected_var, style="Dark.TLabel").pack(
            anchor="w", padx=14
        )
        ttk.Label(editor, text="Brush radius", style="Dark.TLabel").pack(
            anchor="w", padx=14, pady=(12, 0)
        )
        ttk.Scale(
            editor,
            from_=1,
            to=60,
            variable=self.brush_var,
            length=220,
        ).pack(padx=14)
        ttk.Button(editor, text="New cell", command=self._new_cell).pack(
            fill="x", padx=14, pady=(14, 4)
        )
        ttk.Button(editor, text="Delete selected cell", command=self._delete_selected).pack(
            fill="x", padx=14, pady=4
        )
        ttk.Button(editor, text="Undo", command=self._undo).pack(
            fill="x", padx=14, pady=4
        )
        ttk.Separator(editor).pack(fill="x", padx=12, pady=12)
        ttk.Button(editor, text="Save corrected mask", command=self._save).pack(
            fill="x", padx=14, pady=4
        )
        ttk.Button(editor, text="Keyboard help", command=self._show_help).pack(
            fill="x", padx=14, pady=4
        )
        ttk.Label(
            editor,
            text=(
                "Corrections are saved separately\nas TIFF and Cellpose _seg.npy.\n"
                "Predictions are never overwritten."
            ),
            style="Dark.TLabel",
            justify="left",
        ).pack(anchor="w", padx=14, pady=14)

        root.bind("<Left>", lambda _event: self._navigate(-1))
        root.bind("<Right>", lambda _event: self._navigate(1))
        root.bind("<Control-z>", lambda _event: self._undo())
        root.bind("<Control-s>", lambda _event: self._save())
        root.bind("p", lambda _event: self.mode_var.set("paint"))
        root.bind("e", lambda _event: self.mode_var.set("erase"))
        root.bind("s", lambda _event: self.mode_var.set("select"))
        root.bind("n", lambda _event: self._new_cell())

        self.image_list.selection_set(0)
        self.image_list.activate(0)
        self._load_image(dataset.image_ids[0])

    def _confirm_discard(self) -> bool:
        from tkinter import messagebox

        if not self.dirty:
            return True
        return bool(
            messagebox.askyesno(
                "Unsaved corrections",
                "Discard the unsaved corrections for this mask?",
                parent=self.root,
            )
        )

    def _select_image_from_list(self, _event: object = None) -> None:
        selection = self.image_list.curselection()
        if not selection:
            return
        image_id = self.dataset.image_ids[int(selection[0])]
        if image_id == self.current_image_id:
            return
        if not self._confirm_discard():
            current_index = self.dataset.image_ids.index(self.current_image_id)
            self.image_list.selection_clear(0, "end")
            self.image_list.selection_set(current_index)
            return
        self._load_image(image_id)

    def _navigate(self, delta: int) -> None:
        if not self.current_image_id:
            return
        current = self.dataset.image_ids.index(self.current_image_id)
        target = max(0, min(len(self.dataset.image_ids) - 1, current + delta))
        if target == current:
            return
        self.image_list.selection_clear(0, "end")
        self.image_list.selection_set(target)
        self.image_list.activate(target)
        self.image_list.see(target)
        self._select_image_from_list()

    def _load_image(self, image_id: str) -> None:
        from tkinter import messagebox

        self.current_image_id = image_id
        try:
            self.microscopy = load_ome_tiff(self.dataset.images[image_id])
            names = [name or f"Channel {index + 1}" for index, name in enumerate(self.microscopy.channel_names)]
            self.channel_combo.configure(values=names)
            preferred = next(
                (
                    name
                    for name in names
                    if any(token in name.casefold() for token in ("cy5", "gfap", "far red"))
                ),
                names[0],
            )
            self.channel_var.set(preferred)
            self._load_model_mask(self.current_model)
            self._reset_view()
        except Exception as exception:
            messagebox.showerror("Cannot load image", str(exception), parent=self.root)

    def _load_model_mask(self, model_name: str) -> None:
        from tkinter import messagebox

        path = self.dataset.model_masks[model_name].get(self.current_image_id)
        if path is None:
            self.labels = None
            self.status_var.set(f"{self.current_image_id} · no {model_name} mask")
            self.metrics_var.set("Ground-truth metrics: unavailable")
            self._schedule_render()
            return
        try:
            labels = load_instance_mask(path).copy()
            assert self.microscopy is not None
            if labels.shape != self.microscopy.image.shape[-2:]:
                raise ValueError(
                    f"Mask shape {labels.shape} does not match image shape "
                    f"{self.microscopy.image.shape[-2:]}"
                )
            self.labels = labels
            self.current_model = model_name
            self.model_var.set(model_name)
            self.selected_label = 0
            self.undo_stack.clear()
            self.dirty = False
            self._update_selected_label()
            self._update_status_and_metrics()
            self._schedule_render()
        except Exception as exception:
            messagebox.showerror("Cannot load mask", str(exception), parent=self.root)

    def _change_model(self, _event: object = None) -> None:
        requested = self.model_var.get()
        if requested == self.current_model:
            return
        if not self._confirm_discard():
            self.model_var.set(self.current_model)
            return
        self._load_model_mask(requested)

    def _channel_image(self) -> np.ndarray:
        if self.microscopy is None:
            raise RuntimeError("No microscopy image is loaded")
        names = list(self.channel_combo["values"])
        try:
            index = names.index(self.channel_var.get())
        except ValueError:
            index = 0
        return self.microscopy.image[index]

    def _schedule_render(self) -> None:
        if self.render_pending:
            return
        self.render_pending = True
        self.root.after_idle(self._render)

    def _render(self) -> None:
        from PIL import Image, ImageTk

        self.render_pending = False
        if self.microscopy is None:
            return
        rgb = render_instance_overlay(
            self._channel_image(),
            self.labels,
            alpha=float(self.alpha_var.get()),
            show_outlines=bool(self.outline_var.get()),
            selected_label=self.selected_label,
        )
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        height, width = rgb.shape[:2]
        fit = min(canvas_width / width, canvas_height / height)
        scale = max(0.03, min(4.0, fit * self.zoom))
        rendered_width = max(1, int(round(width * scale)))
        rendered_height = max(1, int(round(height * scale)))
        image = Image.fromarray(rgb).resize(
            (rendered_width, rendered_height), Image.Resampling.NEAREST
        )
        self.photo = ImageTk.PhotoImage(image)
        origin_x = (canvas_width - rendered_width) / 2 + self.pan_x
        origin_y = (canvas_height - rendered_height) / 2 + self.pan_y
        self.display_scale = scale
        self.display_origin = (origin_x, origin_y)
        self.canvas.delete("all")
        self.canvas.create_image(origin_x, origin_y, image=self.photo, anchor="nw")

    def _canvas_to_image(self, event: object) -> tuple[int, int] | None:
        if self.labels is None:
            return None
        origin_x, origin_y = self.display_origin
        x = int((event.x - origin_x) / self.display_scale)
        y = int((event.y - origin_y) / self.display_scale)
        if 0 <= y < self.labels.shape[0] and 0 <= x < self.labels.shape[1]:
            return y, x
        return None

    def _snapshot(self) -> None:
        if self.labels is None:
            return
        self.undo_stack.append(self.labels.copy())
        if len(self.undo_stack) > 8:
            self.undo_stack.pop(0)

    def _left_press(self, event: object) -> None:
        point = self._canvas_to_image(event)
        if point is None or self.labels is None:
            return
        y, x = point
        clicked = int(self.labels[y, x])
        mode = self.mode_var.get()
        if mode == "select":
            self.selected_label = clicked
            self._update_selected_label()
            self._schedule_render()
            return
        if mode == "merge":
            if self.selected_label > 0 and clicked > 0 and clicked != self.selected_label:
                self._snapshot()
                self.labels[self.labels == clicked] = self.selected_label
                self.dirty = True
                self._update_status_and_metrics()
                self._schedule_render()
            return
        if mode == "paint" and self.selected_label <= 0:
            self._new_cell()
        self._snapshot()
        self.stroke_active = True
        self._paint_at(y, x)

    def _left_drag(self, event: object) -> None:
        if not self.stroke_active:
            return
        point = self._canvas_to_image(event)
        if point is not None:
            self._paint_at(*point)

    def _left_release(self, _event: object) -> None:
        if self.stroke_active:
            self._update_status_and_metrics()
        self.stroke_active = False

    def _paint_at(self, y: int, x: int) -> None:
        if self.labels is None:
            return
        label_id = 0 if self.mode_var.get() == "erase" else self.selected_label
        paint_instance_disk(
            self.labels,
            y,
            x,
            max(1, int(round(float(self.brush_var.get())))),
            label_id,
        )
        self.dirty = True
        self._update_status_and_metrics(calculate_metrics=False)
        self._schedule_render()

    def _new_cell(self) -> None:
        if self.labels is None:
            return
        self.selected_label = int(self.labels.max(initial=0)) + 1
        self.mode_var.set("paint")
        self._update_selected_label()
        self._schedule_render()

    def _delete_selected(self) -> None:
        if self.labels is None or self.selected_label <= 0:
            return
        if not np.any(self.labels == self.selected_label):
            return
        self._snapshot()
        self.labels[self.labels == self.selected_label] = 0
        self.selected_label = 0
        self.dirty = True
        self._update_selected_label()
        self._update_status_and_metrics()
        self._schedule_render()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        self.labels = self.undo_stack.pop()
        self.dirty = True
        if self.selected_label > int(self.labels.max(initial=0)):
            self.selected_label = 0
        self._update_selected_label()
        self._update_status_and_metrics()
        self._schedule_render()

    def _save(self) -> None:
        from tkinter import messagebox

        if self.labels is None:
            return
        tiff_path = self.correction_directory / f"{self.current_image_id}.tiff"
        overwrite = False
        if tiff_path.exists():
            overwrite = bool(
                messagebox.askyesno(
                    "Replace saved correction?",
                    f"A correction already exists for {self.current_image_id}. Replace it?",
                    parent=self.root,
                )
            )
            if not overwrite:
                return
        try:
            result = save_corrected_instances(
                self.current_image_id,
                self.labels,
                self.correction_directory,
                self.dataset.images[self.current_image_id],
                export_cellpose=True,
                overwrite=overwrite,
                cellpose_channels=(1, 3) if self.microscopy.image.shape[0] >= 3 else (1, 2),
            )
            self.labels = load_instance_mask(result.tiff_path).copy()
            self.dirty = False
            self.undo_stack.clear()
            self._update_status_and_metrics()
            messagebox.showinfo(
                "Correction saved",
                f"Saved {result.cell_count} cells to:\n{result.tiff_path}\n{result.cellpose_path}",
                parent=self.root,
            )
        except Exception as exception:
            messagebox.showerror("Cannot save correction", str(exception), parent=self.root)

    def _update_selected_label(self) -> None:
        self.selected_var.set(
            f"Selected cell: {self.selected_label}" if self.selected_label else "Selected cell: none"
        )

    def _update_status_and_metrics(self, calculate_metrics: bool = True) -> None:
        if self.labels is None:
            return
        count = int(np.unique(self.labels[self.labels > 0]).size)
        suffix = " · unsaved" if self.dirty else ""
        self.status_var.set(
            f"{self.current_image_id} · {self.current_model} · {count} cells{suffix} · zoom {self.zoom:.2f}×"
        )
        truth_path = self.dataset.ground_truth.get(self.current_image_id)
        if truth_path is None:
            self.metrics_var.set("Ground-truth metrics: unavailable")
            return
        if not calculate_metrics:
            self.metrics_var.set("Ground-truth metrics: changed; release/select to recalculate")
            return
        try:
            truth = load_instance_mask(truth_path)
            metrics = instance_segmentation_metrics(self.labels, truth, 0.5)
            self.metrics_var.set(
                f"F1 {metrics['f1']:.3f} · PQ {metrics['panoptic_quality']:.3f} · "
                f"precision {metrics['precision']:.3f} · recall {metrics['recall']:.3f}"
            )
        except Exception as exception:
            self.metrics_var.set(f"Metrics unavailable: {exception}")

    def _show_comparison(self) -> None:
        from PIL import Image, ImageTk
        from tkinter import messagebox

        if self.microscopy is None:
            return
        try:
            panels: list[tuple[str, np.ndarray, str]] = [
                ("Raw", render_instance_overlay(self._channel_image(), None), "")
            ]
            truth = None
            truth_path = self.dataset.ground_truth.get(self.current_image_id)
            if truth_path is not None:
                truth = load_instance_mask(truth_path)
                panels.append(
                    (
                        "Ground truth",
                        render_instance_overlay(
                            self._channel_image(), truth, float(self.alpha_var.get()), True
                        ),
                        f"{np.unique(truth[truth > 0]).size} cells",
                    )
                )
            for model_name in self.dataset.model_names:
                path = self.dataset.model_masks[model_name].get(self.current_image_id)
                if path is None:
                    continue
                mask = (
                    self.labels
                    if model_name == self.current_model and self.labels is not None
                    else load_instance_mask(path)
                )
                detail = f"{np.unique(mask[mask > 0]).size} cells"
                if truth is not None:
                    metrics = instance_segmentation_metrics(mask, truth, 0.5)
                    detail += f" · F1 {metrics['f1']:.3f} · PQ {metrics['panoptic_quality']:.3f}"
                panels.append(
                    (
                        model_name,
                        render_instance_overlay(
                            self._channel_image(), mask, float(self.alpha_var.get()), True
                        ),
                        detail,
                    )
                )
            window = self.tk.Toplevel(self.root)
            window.title(f"Model comparison — {self.current_image_id}")
            window.configure(bg="#17191d")
            window.geometry("1380x900")
            container = self.ttk.Frame(window, style="Dark.TFrame")
            container.pack(fill="both", expand=True, padx=8, pady=8)
            references = []
            columns = 3
            for index, (title, rgb, detail) in enumerate(panels):
                frame = self.ttk.Frame(container, style="Dark.TFrame")
                frame.grid(row=index // columns, column=index % columns, sticky="nsew", padx=5, pady=5)
                self.ttk.Label(frame, text=title, style="Title.TLabel").pack()
                self.ttk.Label(frame, text=detail or "Source fluorescence", style="Dark.TLabel").pack()
                image = Image.fromarray(rgb)
                image.thumbnail((430, 350), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                references.append(photo)
                label = self.tk.Label(frame, image=photo, bg="#090a0d")
                label.pack(fill="both", expand=True, pady=(4, 0))
            for column in range(columns):
                container.columnconfigure(column, weight=1)
            for row in range((len(panels) + columns - 1) // columns):
                container.rowconfigure(row, weight=1)
            window._image_references = references
        except Exception as exception:
            messagebox.showerror("Cannot compare models", str(exception), parent=self.root)

    def _reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._update_status_and_metrics()
        self._schedule_render()

    def _mouse_wheel(self, event: object) -> None:
        self._zoom_by(1.15 if event.delta > 0 else 1 / 1.15, event)

    def _zoom_by(self, factor: float, _event: object = None) -> None:
        self.zoom = max(0.25, min(8.0, self.zoom * factor))
        self._update_status_and_metrics(calculate_metrics=False)
        self._schedule_render()

    def _pan_start(self, event: object) -> None:
        self.pan_anchor = (event.x, event.y)

    def _pan_drag(self, event: object) -> None:
        if self.pan_anchor is None:
            return
        previous_x, previous_y = self.pan_anchor
        self.pan_x += event.x - previous_x
        self.pan_y += event.y - previous_y
        self.pan_anchor = (event.x, event.y)
        self._schedule_render()

    def _show_help(self) -> None:
        from tkinter import messagebox

        messagebox.showinfo(
            "Mask review controls",
            "Left click: select or edit according to the active tool\n"
            "Right drag: pan\nMouse wheel: zoom\n"
            "Left/Right arrows: previous/next image\n"
            "S: select · P: paint · E: erase · N: new cell\n"
            "Ctrl+Z: undo · Ctrl+S: save\n\n"
            "Paint a new ID over a merged cell to split it manually. Use Merge to combine fragments.",
            parent=self.root,
        )

    def _close(self) -> None:
        if self._confirm_discard():
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    """Parse desktop data directories and correction destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--masks",
        action="append",
        type=_model_specification,
        required=True,
        metavar="NAME=DIRECTORY",
        help="Repeat once per model to compare",
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--corrections", type=Path, default=Path("outputs/manual_corrections"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and summarize matching files without opening the desktop window",
    )
    return parser.parse_args()


def main() -> None:
    """Discover aligned files and enter the Tk event loop."""
    args = parse_args()
    model_directories: dict[str, Path] = {}
    for name, directory in args.masks:
        if name in model_directories:
            raise SystemExit(f"Duplicate --masks model name: {name!r}")
        model_directories[name] = directory
    dataset = MaskReviewDataset.discover(
        args.images, model_directories, args.ground_truth
    )
    if args.check_only:
        print(f"Matched images: {len(dataset.image_ids)}")
        for name in dataset.model_names:
            print(f"{name}: {len(dataset.model_masks[name])} masks")
        print(f"Ground truth: {len(dataset.ground_truth)} masks")
        return

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    try:
        MaskReviewApplication(root, dataset, args.corrections)
        root.mainloop()
    except Exception as exception:
        messagebox.showerror("Cannot start mask review", str(exception), parent=root)
        root.destroy()
        raise


if __name__ == "__main__":
    main()
