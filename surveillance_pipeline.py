import cv2
import csv
import torch
import datetime
import torchreid
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import torchvision.transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from pathlib import Path

# Action Recognition
from collections import defaultdict
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


INPUT_FOLDER = "Unmodified_assets"
OUTPUT_FOLDER = "Tested_Videos_VideoMAE"
RESULTS_CSV = "Results_VideoMAE.csv"
CHARADES_CSV = "Charades_v1.csv"

# Detection / Tracking
CONFIDENCE_THRESHOLD = 0.5
OSNET_EVERY_N_FRAMES = 5
IOU_MATCH_THRESHOLD = 0.3

# Action Recognition config
NUM_VIDEO_SEQUENCE_SAMPLES = 16   
SKIP_FRAME = 8
CROP_MARGIN_PERCENT = 10
VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
USE_FP16 = True                    

HOME_ACTIONS = {
    "drinking", "drinking beer", "drinking shots",
    "eating burger", "eating cake", "eating carrots", "eating chips",
    "eating doughnuts", "eating hotdog", "eating ice cream",
    "eating spaghetti", "eating watermelon",
    "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages",
    "frying vegetables",
    "walking the dog", "running on treadmill",
    "reading book", "writing",
    "washing hands", "washing dishes", "cleaning floor",
    "brushing teeth", "folding clothes", "making bed", "texting",
}

# Drawing
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)


# Evaluation labels (identical to the MViT / X-CLIP pipelines)
EVAL_LABELS = {
    "sitting":       ["c010", "c011", "c059", "c123", "c125", "c135", "c151"],
    "drinking":      ["c106"],
    "eating":        ["c065", "c156"],
    "reading book":  ["c032"],
    "opening door":  ["c008"],
    "lying down":    ["c122", "c124", "c134"],
    "running":       ["c150"],
    "walking":       ["c097"],
    "cooking":       ["c066", "c147"],
    "writing":       ["c145"],
    "watching tv":   ["c131", "c132"],
    "washing hands": ["c139"],
}

VIDEOMAE_TO_EVAL = {
    "drinking":               "drinking",
    "drinking beer":          "drinking",
    "drinking shots":         "drinking",
    "eating burger":          "eating",
    "eating cake":            "eating",
    "eating carrots":         "eating",
    "eating chips":           "eating",
    "eating doughnuts":       "eating",
    "eating hotdog":          "eating",
    "eating ice cream":       "eating",
    "eating spaghetti":       "eating",
    "eating watermelon":      "eating",
    "cooking chicken":        "cooking",
    "cooking egg":            "cooking",
    "cooking on campfire":    "cooking",
    "cooking sausages":       "cooking",
    "frying vegetables":      "cooking",
    "walking the dog":        "walking",
    "running on treadmill":   "running",
    "reading book":           "reading book",
    "writing":                "writing",
    "washing hands":          "washing hands",
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}")

# Model loading
print("Loading YOLOv8 model..")
yolo_model = YOLO("yolov8n.pt")

print("Loading OSNet re-identification model...")
osnet_model = torchreid.models.build_model(
    name='osnet_x0_25', num_classes=1000, pretrained=True
)
osnet_model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"Loading VideoMAE action recognition model ({VIDEOMAE_MODEL_NAME})...")
videomae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL_NAME)
videomae_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL_NAME).to(DEVICE)
if USE_FP16 and DEVICE.type == "cuda":
    videomae_model = videomae_model.half()
videomae_model = videomae_model.eval()

videomae_id2label = videomae_model.config.id2label
# How many frames the model expects
try:
    VIDEOMAE_NUM_FRAMES = videomae_model.config.num_frames
except AttributeError:
    VIDEOMAE_NUM_FRAMES = 16

# Constrain to the same home-relevant Kinetics labels as the MViT pipeline.
_label_set = set(videomae_id2label.values())
HOME_ACTION_INDICES = [idx for idx, name in videomae_id2label.items()
                       if name in HOME_ACTIONS and name in _label_set]
print(f"[init] Constraining action recognition to {len(HOME_ACTION_INDICES)} home-relevant Kinetics labels.")
print(f"[init] VideoMAE expects {VIDEOMAE_NUM_FRAMES} frames per clip.")
print("[init] All models loaded.\n")


# Charades annotation loading
def load_charades_annotations(csv_path):
    df = pd.read_csv(csv_path)
    gt_lookup = {}
    for _, row in df.iterrows():
        if pd.isna(row['actions']):
            actions = []
        else:
            actions = []
            for entry in row['actions'].split(';'):
                parts = entry.strip().split()
                if len(parts) == 3:
                    code, start, end = parts[0], float(parts[1]), float(parts[2])
                    actions.append((code, start, end))
        gt_lookup[row['id']] = (actions, float(row['length']), str(row['scene']))
    return gt_lookup


# Scoring helpers
def active_eval_labels_at(gt_actions, t_sec):
    active = set()
    for code, start, end in gt_actions:
        if start <= t_sec <= end:
            for eval_label, codes in EVAL_LABELS.items():
                if code in codes:
                    active.add(eval_label)
    return active


def gt_eval_labels_in_video(gt_actions):
    present = set()
    for code, _, _ in gt_actions:
        for eval_label, codes in EVAL_LABELS.items():
            if code in codes:
                present.add(eval_label)
    return present


def compute_macro_metrics(tp, fp, fn):
    precisions, recalls, f1s = [], [], []
    for cls in EVAL_LABELS:
        c_tp, c_fp, c_fn = tp[cls], fp[cls], fn[cls]
        if c_tp + c_fn == 0:
            continue
        p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_p, macro_r, macro_f1


# Pipeline helpers
def compute_iou(box_a, box_b):
    """IoU between two boxes in (x1, y1, x2, y2) pixel format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_and_pad(frame, box, margin_percent):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)
    size = max(y2 - y1, x2 - x1)
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    half = size // 2
    square_crop = frame[
        max(0, cy - half): min(frame.shape[0], cy + half),
        max(0, cx - half): min(frame.shape[1], cx + half),
    ]
    if square_crop.size == 0:
        return None
    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

def classify_action_from_crops_videomae(crops):
    if len(crops) >= VIDEOMAE_NUM_FRAMES:
        step = len(crops) / VIDEOMAE_NUM_FRAMES
        sampled = [crops[int(i * step)] for i in range(VIDEOMAE_NUM_FRAMES)]
    else:
        sampled = crops + [crops[-1]] * (VIDEOMAE_NUM_FRAMES - len(crops))

    # OpenCV gives BGR; the pretrained VideoMAE weights expect RGB.
    rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in sampled]

    inputs = videomae_processor(rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    if USE_FP16 and DEVICE.type == "cuda":
        pixel_values = pixel_values.half()

    with torch.inference_mode():
        logits = videomae_model(pixel_values=pixel_values).logits
    probs = logits[0].softmax(0)
    best_idx = max(HOME_ACTION_INDICES, key=lambda idx: probs[idx].item())
    return videomae_id2label[best_idx], probs[best_idx].item()


# Per-video processing
def process_video(input_path, output_path, gt_lookup):
    video_stem = Path(input_path).stem

    if video_stem not in gt_lookup:
        print(f"[!] No Charades ground-truth for {video_stem}, skipping.")
        return None
    gt_actions, gt_length, gt_scene = gt_lookup[video_stem]

    tracker = DeepSort(max_age=200, n_init=5, nms_max_overlap=0.5)

    track_embedding_gallery = {}
    GALLERY_SIZE = 10
    REID_THRESHOLD = 0.35

    track_action_buffer = defaultdict(list)
    track_last_action = {}

    video_cap = cv2.VideoCapture(input_path)
    if not video_cap.isOpened():
        print(f"[!] Could not open {input_path}, skipping.")
        return None

    frame_width  = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = video_cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    # Scoring counters
    tp = {cls: 0 for cls in EVAL_LABELS}
    fp = {cls: 0 for cls in EVAL_LABELS}
    fn = {cls: 0 for cls in EVAL_LABELS}
    total_predictions   = 0
    correct_predictions = 0
    predicted_eval_labels = set()

    # Timing
    frame_count     = 0
    total_proc_time = 0.0

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        frame_count += 1
        start_time = datetime.datetime.now()

        results = yolo_model(frame, verbose=False)[0]
        detections = []
        embeddings = []
        det_boxes  = []
        for data in results.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) >= CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = int(data[5])
                if class_id == 0:
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    detections.append([bbox, confidence, class_id])
                    det_boxes.append((xmin, ymin, xmax, ymax))
                    if frame_count % OSNET_EVERY_N_FRAMES == 0:
                        person_img = frame[ymin:ymax, xmin:xmax]
                        if person_img.size > 0:
                            # OpenCV gives BGR; OSNet pretrained weights expect RGB.
                            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            person_img = transform(person_img).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                embedding = osnet_model(person_img)
                            embeddings.append(embedding.squeeze(0).cpu().numpy())
                        else:
                            embeddings.append(None)
                    else:
                        embeddings.append(None)

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.det_class != 0:
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            current_embedding = None
            track_box = (xmin, ymin, xmax, ymax)
            best_iou, best_det = 0.0, -1
            for d_idx, d_box in enumerate(det_boxes):
                iou_val = compute_iou(track_box, d_box)
                if iou_val > best_iou:
                    best_iou, best_det = iou_val, d_idx
            if (best_det != -1 and best_iou >= IOU_MATCH_THRESHOLD
                    and best_det < len(embeddings) and embeddings[best_det] is not None):
                current_embedding = embeddings[best_det]

            if current_embedding is not None:
                if track_id not in track_embedding_gallery:
                    best_match_id       = None
                    best_match_distance = float('inf')
                    for past_id, past_embeddings in track_embedding_gallery.items():
                        distances    = [cosine(current_embedding, e) for e in past_embeddings]
                        top_matches  = sorted(distances)[:3]
                        avg_distance = sum(top_matches) / len(top_matches)
                        if avg_distance < best_match_distance:
                            best_match_distance = avg_distance
                            best_match_id       = past_id
                    if best_match_id is not None and best_match_distance < REID_THRESHOLD:
                        track_id = best_match_id
                if track_id not in track_embedding_gallery:
                    track_embedding_gallery[track_id] = []
                track_embedding_gallery[track_id].append(current_embedding)
                if len(track_embedding_gallery[track_id]) > GALLERY_SIZE:
                    track_embedding_gallery[track_id] = track_embedding_gallery[track_id][-GALLERY_SIZE:]

            # Action recognition
            crop = crop_and_pad(frame, [xmin, ymin, xmax, ymax], CROP_MARGIN_PERCENT)
            if crop is not None:
                track_action_buffer[track_id].append(crop)
                if len(track_action_buffer[track_id]) > NUM_VIDEO_SEQUENCE_SAMPLES:
                    track_action_buffer[track_id].pop(0)

            if (frame_count % SKIP_FRAME == 0
                    and len(track_action_buffer[track_id]) == NUM_VIDEO_SEQUENCE_SAMPLES):
                try:
                    pred_label, pred_conf = classify_action_from_crops_videomae(
                        track_action_buffer[track_id])
                    track_last_action[track_id] = (pred_label, pred_conf)

                    eval_label    = VIDEOMAE_TO_EVAL.get(pred_label, "OTHER")
                    t_sec         = frame_count / fps if fps > 0 else 0.0
                    active_gt_now = active_eval_labels_at(gt_actions, t_sec)

                    total_predictions += 1
                    if eval_label != "OTHER":
                        predicted_eval_labels.add(eval_label)
                        if eval_label in active_gt_now:
                            correct_predictions += 1
                    for cls in EVAL_LABELS:
                        in_gt    = cls in active_gt_now
                        was_pred = (cls == eval_label)
                        if in_gt and was_pred:
                            tp[cls] += 1
                        elif (not in_gt) and was_pred:
                            fp[cls] += 1
                        elif in_gt and (not was_pred):
                            fn[cls] += 1
                except torch.cuda.OutOfMemoryError:
                    print(f"[!] CUDA OOM at frame {frame_count}, skipping.")
                    torch.cuda.empty_cache()

            # Boxes and labels
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            if track_id in track_last_action:
                act_label, act_conf = track_last_action[track_id]
                display_label = VIDEOMAE_TO_EVAL.get(act_label, act_label)
                action_text   = f"Action: {display_label} {act_conf*100:.0f}%"
            else:
                action_text = "Action: warming up"
            cv2.putText(frame, f"ID: {track_id}",
                (xmin + 4, ymax - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 2)
            cv2.putText(frame, action_text,
                (xmin + 4, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 2)

        end_time = datetime.datetime.now()
        elapsed  = (end_time - start_time).total_seconds()
        total_proc_time += elapsed
        if elapsed > 0:
            cv2.putText(frame, f"FPS: {1/elapsed:.2f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

        cv2.imshow("Person Detection & Tracking + VideoMAE Action Recognition", frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[i] Skipped by user")
            break
        if key == 27:
            video_cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("[i] Exited by user")
            exit(0)

    video_cap.release()
    out.release()

    # Finalise metrics
    macro_p, macro_r, macro_f1 = compute_macro_metrics(tp, fp, fn)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    avg_fps  = frame_count / total_proc_time if total_proc_time > 0 else 0.0
    gt_present = gt_eval_labels_in_video(gt_actions)

    print(f"[OK] {video_stem}  preds={total_predictions}  acc={accuracy*100:.1f}%  "
          f"macroF1={macro_f1:.2f}  avgFPS={avg_fps:.1f}")

    return {
        "video_name":        video_stem,
        "length_sec":        round(gt_length, 2),
        "num_predictions":   total_predictions,
        "accuracy":          round(accuracy, 4),
        "macro_precision":   round(macro_p, 4),
        "macro_recall":      round(macro_r, 4),
        "macro_F1":          round(macro_f1, 4),
        "avg_fps":           round(avg_fps, 2),
        "gt_actions":        ";".join(sorted(gt_present)),
        "predicted_actions": ";".join(sorted(predicted_eval_labels)),
        "scene":             gt_scene,
    }


# Main
def main():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(CHARADES_CSV):
        print(f"[!] Cannot find {CHARADES_CSV} in the working directory.")
        print(f"    Put the Charades annotations CSV next to this script and re-run.")
        return
    print(f"Loading Charades annotations from {CHARADES_CSV}...")
    gt_lookup = load_charades_annotations(CHARADES_CSV)
    print(f"[init] Loaded ground truth for {len(gt_lookup)} videos.\n")

    videos = sorted([f for f in os.listdir(INPUT_FOLDER)
                     if f.lower().endswith(".mp4")])
    if not videos:
        print(f"No videos found in '{INPUT_FOLDER}'.")
        return
    print(f"Found {len(videos)} video(s) to process with VideoMAE.\n")

    fieldnames = ["video_name", "length_sec", "num_predictions", "accuracy",
                  "macro_precision", "macro_recall", "macro_F1", "avg_fps",
                  "gt_actions", "predicted_actions", "scene"]
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        all_results = []
        for video_name in videos:
            input_path  = os.path.join(INPUT_FOLDER, video_name)
            name_stem   = Path(video_name).stem
            output_path = os.path.join(OUTPUT_FOLDER, f"{name_stem}_videomae.mp4")
            result = process_video(input_path, output_path, gt_lookup)
            if result is not None:
                writer.writerow(result)
                f.flush()
                all_results.append(result)

        if all_results:
            n = len(all_results)
            overall = {
                "video_name":        "OVERALL",
                "length_sec":        round(sum(r["length_sec"] for r in all_results), 2),
                "num_predictions":   sum(r["num_predictions"] for r in all_results),
                "accuracy":          round(sum(r["accuracy"] for r in all_results) / n, 4),
                "macro_precision":   round(sum(r["macro_precision"] for r in all_results) / n, 4),
                "macro_recall":      round(sum(r["macro_recall"] for r in all_results) / n, 4),
                "macro_F1":          round(sum(r["macro_F1"] for r in all_results) / n, 4),
                "avg_fps":           round(sum(r["avg_fps"] for r in all_results) / n, 2),
                "gt_actions":        "",
                "predicted_actions": "",
                "scene":             f"N={n}",
            }
            writer.writerow(overall)

            print("\n" + "=" * 60)
            print(f"=== OVERALL  (N={n} clips, VideoMAE) ===")
            print(f"Macro-precision:  {overall['macro_precision']:.4f}")
            print(f"Macro-recall:     {overall['macro_recall']:.4f}")
            print(f"Macro-F1:         {overall['macro_F1']:.4f}")
            print(f"Accuracy:         {overall['accuracy']*100:.1f}%")
            print(f"Avg FPS:          {overall['avg_fps']:.1f}")
            print("=" * 60)

    cv2.destroyAllWindows()
    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()