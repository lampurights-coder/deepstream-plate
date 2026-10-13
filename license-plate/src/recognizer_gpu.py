import os
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.decoder import SARLabelDecode
from src.trt_loader import TensorRTInferenceSession

logger = logging.getLogger("recognizer")

# ----------------------------------------------------------------------
# License Plate Recognition (unchanged except for inference session)
# ----------------------------------------------------------------------
class LicensePlateRecognition:
    def __init__(
        self,
        model_path,
        character_dict_path,
        use_gpu=True,
        gpu_id=0,
        engine_type="tensorrt",
    ):
        self.model_path = model_path
        self.character_dict_path = character_dict_path
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.engine_type = engine_type

        if engine_type != "tensorrt":
            raise NotImplementedError("Only 'tensorrt' engine is supported.")

        if not use_gpu:
            logger.warning("TensorRT requires GPU, enabling use_gpu=True")
            self.use_gpu = True

        # Use the fixed TensorRT session
        self.inference_session = TensorRTInferenceSession(model_path, self.use_gpu, gpu_id)

        self.postprocess_op = self._init_postprocessor()
        self.rec_image_shape = [3, 48, 48, 160]  # C, H, minW, maxW
        self.rec_batch_num = 6
        self.rec_algorithm = "RobustScanner"
        self.return_word_box = False

    def _init_postprocessor(self):
        return SARLabelDecode(self.character_dict_path, use_space_char=False, rm_symbol=True)

    def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h, w = img.shape[:2]
        valid_ratio = 1.0
        width_divisor = int(1 / width_downsample_ratio)
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)

        resized = cv2.resize(img, (resize_w, imgH))
        resized = resized.astype("float32")
        if imgC == 1:
            resized = resized / 255.0
            resized = resized[np.newaxis, ...]
        else:
            resized = resized.transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5

        # Pad to imgW_max
        padded = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padded[:, :, :resize_w] = resized
        return padded, resized.shape, padded.shape, valid_ratio

    def preprocess_images(self, img_list):
        norm_img_batch = []
        valid_ratios = []
        word_positions_list = []

        for img in img_list:
            if self.rec_algorithm == "RobustScanner":
                norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                    img, self.rec_image_shape, width_downsample_ratio=0.25
                )
                norm_img = norm_img[np.newaxis, ...]  # [1, C, H, W]
                valid_ratio = np.expand_dims(valid_ratio, axis=0)  # [1,1]
                norm_img_batch.append(norm_img)
                valid_ratios.append(valid_ratio)

                word_positions = np.arange(0, 40, dtype=np.int64)  # [40]
                word_positions = np.expand_dims(word_positions, axis=0)  # [1,40]
                word_positions_list.append(word_positions)

        if norm_img_batch:
            norm_img_batch = np.concatenate(norm_img_batch, axis=0).astype(np.float32)
            valid_ratios = np.concatenate(valid_ratios, axis=0).astype(np.float32)
            word_positions_list = np.concatenate(word_positions_list, axis=0).astype(np.int64)
        else:
            norm_img_batch = np.empty((0, *self.rec_image_shape[:3]), dtype=np.float32)
            valid_ratios = np.empty((0, 1), dtype=np.float32)
            word_positions_list = np.empty((0, 40), dtype=np.int64)

        return norm_img_batch, valid_ratios, word_positions_list

    def process_images(self, img_list):
        if not img_list:
            return []

        img_num = len(img_list)
        for i, img in enumerate(img_list):
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array at index {i}, got {type(img)}")

        # Sort by aspect ratio (RobustScanner likes variable width)
        width_ratios = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(width_ratios)
        rec_res = [["", 0.0]] * img_num

        for beg in range(0, img_num, self.rec_batch_num):
            end = min(img_num, beg + self.rec_batch_num)
            batch_imgs = [img_list[indices[i]] for i in range(beg, end)]

            norm_img_batch, valid_ratios, word_positions = self.preprocess_images(batch_imgs)

            if norm_img_batch.size == 0:
                for rno in range(len(batch_imgs)):
                    rec_res[indices[beg + rno]] = ["", 0.0]
                continue

            # Run inference
            input_dict = {
                "x": norm_img_batch,
                "data_0": valid_ratios,
                "data_1": word_positions,
            }
            t0 = time.time()
            outputs = self.inference_session.run(input_dict)
            print(f"Inference time for batch: {time.time() - t0:.4f}s")

            preds = outputs[0]   # shape [batch, time, num_classes]
            batch_results = self.postprocess_op(preds)

            for rno, res in enumerate(batch_results):
                rec_res[indices[beg + rno]] = res

        return rec_res

    def __call__(self, img_list):
        return self.process_images(img_list)

if __name__ == "__main__":
    print("[INFO]: starting...")
    # Paths – adjust to your files
    trt_engine_path = "./models/rb_scaner.trt"
    dict_path = "./dictionary_plate/dict90.txt"
    image_paths = ["./cropped_plates/test_infer_2_plate_0.jpg", "./cropped_plates/test_infer_3_plate_0.jpg"]  # replace with your image

    # Check files exist
    if not Path(trt_engine_path).exists():
        print(f"ERROR: TRT engine not found at {trt_engine_path}")
        sys.exit(1)
    if not Path(dict_path).exists():
        print(f"ERROR: dictionary not found at {dict_path}")
        sys.exit(1)

    # Initialize recognizer
    recognizer = LicensePlateRecognition(
        model_path=trt_engine_path,
        character_dict_path=dict_path,
        use_gpu=True,
        gpu_id=0,
        engine_type="tensorrt",
    )
    print("[INFO]: Model Successfully loaded.")
    # Load image
    imgs = [cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in image_paths]
    if imgs is None:
        print(f"Failed to load image: {imgs}")
        sys.exit(1)

    # Run recognition
    results = recognizer.process_images(imgs)
    print("Recognition result:", results)  # text