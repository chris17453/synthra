from PIL import Image
import cv2
from.config import face_detection

class HeadTracker:
    def __init__(self, smoothing_factor=0.2):
        self.prev_bbox = None
        self.smoothing_factor = smoothing_factor

    def smooth_bbox(self, bbox):
        if self.prev_bbox is None:
            self.prev_bbox = bbox
        else:
            self.prev_bbox = [
                int(self.smoothing_factor * b + (1 - self.smoothing_factor) * p)
                for b, p in zip(bbox, self.prev_bbox)
            ]
        return self.prev_bbox


def detect_and_crop_head(frame, tracker):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        ih, iw, _ = frame.shape
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        bbox = [x, y, w, h]
        smoothed_bbox = tracker.smooth_bbox(bbox)

        padding = int(2 * smoothed_bbox[2])
        x1 = max(0, smoothed_bbox[0] - padding)
        y1 = max(0, smoothed_bbox[1] - padding)
        x2 = min(frame.shape[1], smoothed_bbox[0] + smoothed_bbox[2] + padding)
        y2 = min(frame.shape[0], smoothed_bbox[1] + smoothed_bbox[3] + padding)

        return frame[y1:y2, x1:x2], smoothed_bbox
    else:
        return frame, None
