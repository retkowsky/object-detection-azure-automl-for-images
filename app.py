"""
ONNX Object Detection Web UI — Flask Backend
Serves inference from a user-uploaded ONNX model and labels file.
Compatible with Azure AutoML for Images YOLO-based ONNX exports.
"""

import base64
import io
import json
import os
import time
from typing import Any

import numpy as np
import onnxruntime
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# ---------------------------------------------------------------------------
# Preprocessing utilities (Azure AutoML / YOLO-compatible)
# ---------------------------------------------------------------------------

def letterbox_image(image: Image.Image, target_size: tuple[int, int]) -> tuple[Image.Image, float, tuple[int, int]]:
    """Resize an image with letterboxing (aspect-ratio preserved, padded).

    Args:
        image: Input PIL image.
        target_size: Desired (width, height) output size.

    Returns:
        Tuple of (padded PIL image, scale factor, (pad_left, pad_top)).
    """
    iw, ih = image.size
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (tw, th), (114, 114, 114))
    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, (pad_left, pad_top)


def preprocess_image(
    image: Image.Image,
    height: int = 640,
    width: int = 640,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Preprocess a PIL image for ONNX YOLO inference.

    Converts to RGB, letterboxes to (height, width), normalises to [0, 1],
    and returns a float32 NCHW tensor.

    Args:
        image: Input PIL image (any mode).
        height: Model input height in pixels.
        width: Model input width in pixels.

    Returns:
        Tuple of:
          - img_data: numpy ndarray of shape (1, 3, height, width), dtype float32.
          - scale: Scale factor applied during letterboxing.
          - padding: (pad_left, pad_top) pixel offsets applied.
    """
    image = image.convert("RGB")
    padded, scale, padding = letterbox_image(image, (width, height))
    img_array = np.array(padded, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_data = np.expand_dims(img_array, axis=0)     # CHW → NCHW
    return img_data, scale, padding


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (vectorised NumPy)
# ---------------------------------------------------------------------------

def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes.

    Args:
        box: Shape (4,) — [x1, y1, x2, y2].
        boxes: Shape (N, 4) — [x1, y1, x2, y2].

    Returns:
        IoU values, shape (N,).
    """
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter_area

    return inter_area / np.maximum(union, 1e-6)


def non_max_suppression(
    predictions: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict[str, np.ndarray]]:
    """Apply NMS to raw YOLO output.

    Args:
        predictions: Raw model output, shape (batch, num_anchors, 5 + num_classes).
        conf_threshold: Minimum objectness × class confidence.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of per-image dicts with keys 'boxes', 'scores', 'labels'.
    """
    results = []
    batch_size = predictions.shape[0]

    for b in range(batch_size):
        pred = predictions[b]  # (num_anchors, 5 + C)
        obj_conf = pred[:, 4]
        class_scores = pred[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        scores = obj_conf * class_conf

        mask = scores >= conf_threshold
        pred = pred[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(pred) == 0:
            results.append({"boxes": np.empty((0, 4)), "scores": np.array([]), "labels": np.array([], dtype=int)})
            continue

        # cx, cy, w, h → x1, y1, x2, y2
        boxes = np.zeros((len(pred), 4), dtype=np.float32)
        boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
        boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
        boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
        boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2

        # Per-class NMS
        keep_indices: list[int] = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_orig_idx = np.where(cls_mask)[0]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores_sorted = cls_scores[order]
            cls_orig_idx = cls_orig_idx[order]

            suppressed = np.zeros(len(cls_boxes), dtype=bool)
            for i in range(len(cls_boxes)):
                if suppressed[i]:
                    continue
                keep_indices.append(cls_orig_idx[i])
                iou = _iou(cls_boxes[i], cls_boxes[i + 1:])
                suppressed[i + 1:] = suppressed[i + 1:] | (iou > iou_threshold)

        keep_indices = sorted(keep_indices)
        results.append({
            "boxes": boxes[keep_indices],
            "scores": scores[keep_indices],
            "labels": class_ids[keep_indices],
        })

    return results


# ---------------------------------------------------------------------------
# Coordinate de-normalisation
# ---------------------------------------------------------------------------

def rescale_boxes(
    boxes: np.ndarray,
    orig_size: tuple[int, int],
    model_size: tuple[int, int],
    scale: float,
    padding: tuple[int, int],
) -> np.ndarray:
    """Convert letterboxed model-space coordinates back to original image space.

    Args:
        boxes: Shape (N, 4) in model pixel coords [x1, y1, x2, y2].
        orig_size: Original image (width, height).
        model_size: Model input (width, height).
        scale: Scale factor used during letterboxing.
        padding: (pad_left, pad_top) applied during letterboxing.

    Returns:
        Boxes in original image pixel coordinates, shape (N, 4).
    """
    if len(boxes) == 0:
        return boxes

    pad_left, pad_top = padding
    rescaled = boxes.copy()
    rescaled[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    rescaled[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

    orig_w, orig_h = orig_size
    rescaled[:, [0, 2]] = np.clip(rescaled[:, [0, 2]], 0, orig_w)
    rescaled[:, [1, 3]] = np.clip(rescaled[:, [1, 3]], 0, orig_h)
    return rescaled


def boxes_to_normalised(
    boxes: np.ndarray,
    orig_size: tuple[int, int],
) -> list[dict[str, float]]:
    """Convert pixel-coordinate boxes to normalised [0, 1] dict format.

    Args:
        boxes: Shape (N, 4), pixel coords [x1, y1, x2, y2].
        orig_size: Original image (width, height).

    Returns:
        List of dicts with keys topX, topY, bottomX, bottomY ∈ [0, 1].
    """
    orig_w, orig_h = orig_size
    normalised = []
    for box in boxes:
        normalised.append({
            "topX": float(box[0]) / orig_w,
            "topY": float(box[1]) / orig_h,
            "bottomX": float(box[2]) / orig_w,
            "bottomY": float(box[3]) / orig_h,
        })
    return normalised


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def run_inference(
    session: onnxruntime.InferenceSession,
    img_data: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass through the ONNX session.

    Args:
        session: Loaded ONNX Runtime inference session.
        img_data: Preprocessed input tensor, shape (1, 3, H, W), dtype float32.

    Returns:
        First output tensor from the session (raw predictions).
    """
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names=output_names, input_feed={input_name: img_data})
    return outputs[0]


# ---------------------------------------------------------------------------
# Visualisation (draw on PIL image, return base64 PNG)
# ---------------------------------------------------------------------------

# A vibrant palette cycling for multi-class detection
_PALETTE: list[tuple[int, int, int]] = [
    (0, 200, 100),    # lime-green  (primary)
    (0, 168, 255),    # azure-blue
    (255, 80, 80),    # coral-red
    (255, 200, 0),    # amber
    (180, 80, 255),   # violet
    (255, 140, 0),    # orange
    (0, 230, 230),    # cyan
    (255, 100, 180),  # pink
]


def _color_for_class(class_index: int) -> tuple[int, int, int]:
    """Return a stable RGB colour for a given class index.

    Args:
        class_index: Integer class index.

    Returns:
        RGB tuple.
    """
    return _PALETTE[class_index % len(_PALETTE)]


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    confidence_threshold: float = 0.5,
    line_width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """Draw bounding boxes and labels onto a PIL image.

    Args:
        image: Original PIL image (RGB).
        detections: List of dicts with keys: label, score, box (normalised coords),
                    class_index.
        confidence_threshold: Only draw detections above this score.
        line_width: Bounding box stroke width.
        font_size: Label font size.

    Returns:
        Annotated PIL image (RGB).
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 2)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    for det in detections:
        if det["score"] < confidence_threshold:
            continue

        box = det["box"]
        x1 = int(box["topX"] * w)
        y1 = int(box["topY"] * h)
        x2 = int(box["bottomX"] * w)
        y2 = int(box["bottomY"] * h)

        color = _color_for_class(det.get("class_index", 0))
        color_alpha = color + (180,)
        text_bg_alpha = (0, 0, 0, 210)

        # Draw box with a thicker outer stroke for readability
        for offset in range(line_width, 0, -1):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                           outline=color + (100,))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Corner accents
        corner = 14
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = 1 if cx == x1 else -1
            dy = 1 if cy == y1 else -1
            draw.line([(cx, cy), (cx + dx * corner, cy)], fill=color, width=line_width + 1)
            draw.line([(cx, cy), (cx, cy + dy * corner)], fill=color, width=line_width + 1)

        # Label
        label_text = f"{det['label'].upper()}  {det['score']:.1%}"
        bbox_text = font.getbbox(label_text)
        txt_w = bbox_text[2] - bbox_text[0]
        txt_h = bbox_text[3] - bbox_text[1]
        pad = 6
        label_y = max(0, y1 - txt_h - pad * 2 - 2)

        draw.rectangle(
            [x1 - 1, label_y, x1 + txt_w + pad * 2 + 1, label_y + txt_h + pad * 2],
            fill=color + (230,),
        )
        draw.text((x1 + pad, label_y + pad - 1), label_text, fill=(0, 0, 0), font=font)

    return img


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string.

    Args:
        image: PIL image to encode.
        fmt: Image format string (e.g. 'PNG', 'JPEG').

    Returns:
        Base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    """Serve the main Web UI page."""
    return render_template("index.html")


@app.route("/infer", methods=["POST"])
def infer() -> Any:
    """Run object detection inference on an uploaded image.

    Expects a multipart/form-data POST with:
        - onnx_model  : .onnx model file
        - labels_file : .json labels file (list of class names)
        - image       : image file (jpg / png / bmp / tiff)
        - conf_threshold : float in [0, 1] (default 0.5)
        - iou_threshold  : float in [0, 1] (default 0.45)

    Returns:
        JSON with annotated_image (base64), detections list, and timing info.
    """
    try:
        # ---- Parse form fields ----
        conf_threshold = float(request.form.get("conf_threshold", 0.5))
        iou_threshold = float(request.form.get("iou_threshold", 0.45))

        # ---- Load labels ----
        labels_file = request.files.get("labels_file")
        if labels_file is None:
            return jsonify({"error": "Missing labels_file"}), 400
        classes: list[str] = json.load(labels_file)

        # ---- Load image ----
        image_file = request.files.get("image")
        if image_file is None:
            return jsonify({"error": "Missing image"}), 400
        orig_image = Image.open(image_file).convert("RGB")
        orig_w, orig_h = orig_image.size

        # ---- Load ONNX model ----
        onnx_file = request.files.get("onnx_model")
        if onnx_file is None:
            return jsonify({"error": "Missing onnx_model"}), 400

        onnx_bytes = onnx_file.read()
        t_load_start = time.perf_counter()
        session = onnxruntime.InferenceSession(onnx_bytes)
        t_load = time.perf_counter() - t_load_start

        # ---- Determine model input shape ----
        input_shape = session.get_inputs()[0].shape
        _, _, model_h, model_w = input_shape  # batch, C, H, W

        # ---- Preprocess ----
        t_pre_start = time.perf_counter()
        img_data, scale, padding = preprocess_image(orig_image, height=model_h, width=model_w)
        t_pre = time.perf_counter() - t_pre_start

        # ---- Inference ----
        t_inf_start = time.perf_counter()
        raw_pred = run_inference(session, img_data)
        t_inf = time.perf_counter() - t_inf_start

        # ---- Post-process ----
        t_post_start = time.perf_counter()
        nms_results = non_max_suppression(raw_pred, conf_threshold, iou_threshold)
        result = nms_results[0]  # single image

        pixel_boxes = rescale_boxes(
            result["boxes"],
            orig_size=(orig_w, orig_h),
            model_size=(model_w, model_h),
            scale=scale,
            padding=padding,
        )
        norm_boxes = boxes_to_normalised(pixel_boxes, (orig_w, orig_h))

        detections: list[dict] = []
        for i, (norm_box, score, label_idx) in enumerate(
            zip(norm_boxes, result["scores"], result["labels"])
        ):
            label_idx = int(label_idx)
            detections.append({
                "id": i + 1,
                "label": classes[label_idx] if label_idx < len(classes) else str(label_idx),
                "class_index": label_idx,
                "score": float(score),
                "box": norm_box,
            })

        # Sort by score descending
        detections.sort(key=lambda d: d["score"], reverse=True)
        t_post = time.perf_counter() - t_post_start

        # ---- Draw annotations ----
        annotated = draw_detections(orig_image, detections, confidence_threshold=conf_threshold)
        annotated_b64 = image_to_base64(annotated)
        orig_b64 = image_to_base64(orig_image)

        return jsonify({
            "annotated_image": annotated_b64,
            "original_image": orig_b64,
            "detections": detections,
            "meta": {
                "total_detections": len(detections),
                "above_threshold": sum(1 for d in detections if d["score"] >= conf_threshold),
                "model_input_shape": list(input_shape),
                "original_image_size": [orig_w, orig_h],
                "timing": {
                    "model_load_s": round(t_load, 3),
                    "preprocess_s": round(t_pre, 3),
                    "inference_s": round(t_inf, 3),
                    "postprocess_s": round(t_post, 3),
                    "total_s": round(t_load + t_pre + t_inf + t_post, 3),
                },
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
