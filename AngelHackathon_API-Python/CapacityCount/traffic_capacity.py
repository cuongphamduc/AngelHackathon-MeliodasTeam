import logging
import logging.handlers
import os
import time
import cv2
import numpy as np
from CapacityCount import utils

cv2.ocl.setUseOpenCL(False)

IMAGE_DIR = "./out"
VIDEO_SOURCE = '1.mp4'
SHAPE = (720, 1280)
AREA_PTS = np.array([[780, 716], [686, 373], [883, 383], [1280, 636], [1280, 720]]) 

from CapacityCount.pipeline import (
    PipelineRunner,
    CapacityCounter,
    ContextCsvWriter
)

def main():
    log = logging.getLogger("main")

    base = np.zeros(SHAPE + (3,), dtype='uint8')
    area_mask = cv2.fillPoly(base, [AREA_PTS], (255, 255, 255))[:, :, 0]

    pipeline = PipelineRunner(pipeline=[
        CapacityCounter(area_mask=area_mask, save_image=True, image_dir=IMAGE_DIR),
        # saving every 10 seconds
        ContextCsvWriter('./report.csv', start_time=1505494325, fps=1, faster=10, field_names=['capacity'])
    ], log_level=logging.DEBUG)

    # Set up image source
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    frame_number = -1
    st = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret == False):
            break

        frame_number += 1

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        context = pipeline.run()

        for i in range(50):
            ret, frame = cap.read()
            if (ret == False):
                break
    cap.release()

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
