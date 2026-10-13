import cv2
import numpy as np
import math
import os
from src.inference import ONNXInferenceSession
from src.decoder import SARLabelDecode
import logging
import time

logger = logging.getLogger("recognizer")


class LicensePlateRecognition:
    def __init__(self, model_path, character_dict_path, use_gpu=False, gpu_id=0):
        self.model_path = model_path
        self.character_dict_path = character_dict_path
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # Initialize components
        self.inference_session = ONNXInferenceSession(model_path, use_gpu, gpu_id)
        self.postprocess_op = self._init_postprocessor()
        # keep shapes as in your original code; adjust if your model expects different
        self.rec_image_shape = [3, 48, 48, 160]
        self.rec_batch_num = 6
        self.rec_algorithm = 'RobustScanner'
        self.return_word_box = False

    def _init_postprocessor(self):
        postprocess_params = {
            "name": "SARLabelDecode",
            "character_dict_path": self.character_dict_path,
            "use_space_char": False,
            "rm_symbol": True,
        }
        return SARLabelDecode(**postprocess_params)

    def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
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
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype("float32")
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape
        return padding_im, resize_shape, pad_shape, valid_ratio

    def preprocess_images(self, img_list):
        norm_img_batch = []
        valid_ratios = []
        word_positions_list = []

        for img in img_list:
            if self.rec_algorithm == "RobustScanner":
                norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                    img, self.rec_image_shape, width_downsample_ratio=0.25
                )
                norm_img = norm_img[np.newaxis, :]
                valid_ratio = np.expand_dims(valid_ratio, axis=0)
                valid_ratios.append(valid_ratio)
                norm_img_batch.append(norm_img)
                word_positions = np.array(range(0, 40)).astype("int64")
                word_positions = np.expand_dims(word_positions, axis=0)
                word_positions_list.append(word_positions)

        if norm_img_batch:
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
        else:
            norm_img_batch = np.array([], dtype=np.float32)

        if valid_ratios:
            valid_ratios = np.concatenate(valid_ratios).astype(np.float32)
        else:
            valid_ratios = np.array([], dtype=np.float32)

        if word_positions_list:
            word_positions_list = np.concatenate(word_positions_list).astype(np.int64)
        else:
            word_positions_list = np.array([], dtype=np.int64)

        return norm_img_batch, valid_ratios, word_positions_list

    def process_images(self, img_list):
        """Process a list of images and return recognition results"""
        if not img_list:
            return []

        img_num = len(img_list)
        # Ensure each item is a valid np image
        for i, img in enumerate(img_list):
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy.ndarray at index {i}, got {type(img)}")

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            batch_imgs = [img_list[indices[i]] for i in range(beg_img_no, end_img_no)]

            # Preprocess
            norm_img_batch, valid_ratios, word_positions_list = self.preprocess_images(batch_imgs)

            # If preprocessing failed or returned empty arrays, skip
            if norm_img_batch.size == 0:
                logger.warning("Empty preprocessed batch, skipping.")
                for rno in range(len(batch_imgs)):
                    rec_res[indices[beg_img_no + rno]] = ["", 0.0]
                continue

            # Inference
            if self.rec_algorithm == "RobustScanner":
                input_dict = {
                    'x': norm_img_batch,
                    'data_0': valid_ratios,
                    'data_1': word_positions_list
                }
                t1 = time.time()
                outputs = self.inference_session.run(input_dict)
                print("infernce time : ",time.time()-t1)
                preds = outputs[0]
            else:
                input_dict = {'x': norm_img_batch}
                outputs = self.inference_session.run(input_dict)
                preds = outputs[0]

            # Postprocess
            rec_result = self.postprocess_op(preds)

            # Store results
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res

    def __call__(self, img_list):
        """Make the class callable for convenience"""
        return self.process_images(img_list)

if __name__ == "__main__":
    plate_recognition = LicensePlateRecognition(
            model_path="./models/rb_scaner.onnx",
            character_dict_path="./dictionary_plate/dict90.txt"
        )
    image = "/home/arshia/Downloads/projects/plate_v1/output.jpg"
    plate_images = [cv2.imread(image)]
    results = [res for res in plate_recognition.process_images(plate_images)]
    print(results)