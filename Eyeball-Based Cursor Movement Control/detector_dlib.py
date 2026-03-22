from logging import DEBUG

import numpy as np

self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

if DEBUG:
    print("[DEEPGAZE] PnpHeadPoseEstimator: estimated camera matrix: \n" + str(self.camera_matrix) + "\n")

    self._detector = dlib.get_frontal_face_detector()
self._shape_predictor = dlib.shape_predictor(dlib)
