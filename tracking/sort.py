import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
        (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh
    )
    return o


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)
        self.kf.x[:4] = bbox.reshape((4, 1))
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.x[:4] = bbox.reshape((4, 1))

    def predict(self):
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))


class Sort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        tracked_objects = []

        for det in detections:
            if len(self.trackers) == 0:
                self.trackers.append(KalmanBoxTracker(det))
                continue

            ious = [iou(det, trk.predict()) for trk in self.trackers]
            max_iou = max(ious)
            idx = ious.index(max_iou)

            if max_iou > 0.3:
                self.trackers[idx].update(det)
            else:
                self.trackers.append(KalmanBoxTracker(det))

        for trk in self.trackers:
            bbox = trk.predict()
            tracked_objects.append(
                np.concatenate((bbox, [trk.id]))
            )

        return np.array(tracked_objects)
