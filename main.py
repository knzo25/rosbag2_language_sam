import os
from pathlib import Path
import sys
import argparse

from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage
import rosbag2_py  # noqa
import numpy as np
import torch
from language_sam import LanguageSAM
import cv2
import json
from PIL import Image
import re

from collections import defaultdict
from typing import Dict

if os.environ.get('ROSBAG2_PY_TEST_WITH_RTLD_GLOBAL', None) is not None:
    # This is needed on Linux when compiling with clang/libc++.
    # TL;DR This makes class_loader work when using a python extension compiled with libc++.
    #
    # For the fun RTTI ABI details, see https://whatofhow.wordpress.com/2015/03/17/odr-rtti-dso/.
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)


def process_image(model : LanguageSAM, image_array : np.array, tasks : Dict[str, None]):
    
    # Image is in BGR format. Convert to RGB
    image_array = image_array[...,::-1]

    blurred_image_array = np.array(image_array)
    segmented_image_array = np.array(image_array)
    detections_image_array = np.array(image_array)
    
    height, width, _ = image_array.shape

    print(f"image.shape = {image_array.shape}")
    
    # Just debug
    pil_image = Image.fromarray(image_array)
    #pil_image.save("input.jpg")
    
    results = [model.predict(pil_image, task["prompt"]) for task in tasks]
    masks, bboxes, phrases, logits = zip(*results)
    print(phrases, logits)

    masks = torch.cat(masks, dim=0).bool()
    bboxes = torch.cat(bboxes, dim=0)
    tasks_output = [tasks[i] for (i, aux) in enumerate(phrases) for _ in aux]
    phrases = [x for xs in phrases for x in xs]    
    logits = torch.cat(logits, dim=0)

    if len(masks) == 0:
        print(f"No matches found")

    for i, (mask, bbox, phrase, logit, task) in enumerate(zip(masks, bboxes, phrases, logits, tasks_output)):
        
        mask_numpy = mask.cpu().numpy()
        print(f"{i}: {phrase} ({logit:.2f}) bbox={bbox}")

        blur = task["blur"]
        draw_segmentation = task["draw_segmentation"]
        draw_detection = task["draw_detection"]

        # Some objects need to be properly inside others. like licenses in cars
        valid = len(task["is_inside"]) == 0
        self_area = mask.sum().float()

        for required_key in task["is_inside"]:
            inside_min_ratio = task["inside_min_ratio"]
            inside_max_iou = task["inside_max_iou"]

            # Iterate over all other detections. inneficcient but works
            for j, (other_mask, other_bbox, other_phrase, other_logit, other_task) in enumerate(zip(masks, bboxes, phrases, logits, tasks_output)):
                if i == j:
                    continue

                if required_key not in other_phrase:
                    continue

                intersection = torch.logical_and(mask, other_mask).sum().float()
                union = torch.logical_or(mask, other_mask).sum()
                inside_ratio = intersection / self_area
                iou = intersection.float() / union.float()


                if inside_ratio > inside_min_ratio and iou < inside_max_iou:
                    valid = True
                    break

        if not valid:
            print(f"Skipping {phrase} as it is not inside a valid object")
            continue

        for required_key in task["is_not_inside"]:
            not_inside_max_ratio = task["not_inside_max_ratio"]
            not_inside_max_iou = task["not_inside_max_iou"]

            # Iterate over all other detections. inneficcient but works
            for j, (other_mask, other_bbox, other_phrase, other_logit, other_task) in enumerate(zip(masks, bboxes, phrases, logits, tasks_output)):
                if i == j:
                    continue

                if required_key not in other_phrase:
                    continue

                intersection = torch.logical_and(mask, other_mask).sum().float()
                union = torch.logical_or(mask, other_mask).sum()
                inside_ratio = intersection / self_area
                iou = intersection.float() / union.float()


                if inside_ratio > not_inside_max_ratio or iou > not_inside_max_iou:
                    valid = False
                    break

        if not valid:
            print(f"Skipping {phrase} as it is not a not inside a valid object")
            continue

        if blur:
            min_i, min_j, max_i, max_j = bbox.cpu().numpy().astype(int)
            margin = 20
            min_i = max(0, min_i - margin)
            min_j = max(0, min_j - margin)
            max_i = min(width, max_i + margin)
            max_j = min(height, max_j + margin)
            #image_array[min_j:max_j, min_i:max_i, :] = 0

            roi = image_array[min_j:max_j, min_i:max_i, :]
            mask_roi = mask_numpy[min_j:max_j, min_i:max_i]
            #tag_mask_roi = tag_mask[min_j:max_j, min_i:max_i]
            roi_height, roi_width, _ = roi.shape

            margin_kernel = 7
            scores_roi = mask_roi.astype(np.float32)
            scores_roi = cv2.GaussianBlur(scores_roi, (margin_kernel, margin_kernel), 0)
            scores_roi = np.expand_dims(scores_roi, axis=-1)

            factor = task["blur_factor"]

            smooth_kernel = min(roi_height, roi_width) / factor
            smooth_kernel = 1 + 2*(int(smooth_kernel) // 2)
            
            smoothed_roi =  cv2.GaussianBlur(roi, (smooth_kernel, smooth_kernel), 0)

            combined = scores_roi * smoothed_roi + (1 - scores_roi) * roi
            #combined[tag_mask_roi] = roi[tag_mask_roi]

            blurred_image_array[min_j:max_j, min_i:max_i] = combined.astype(np.uint8)

        if draw_segmentation:
            color = np.array(task["color"])
            segmented_image_array[mask_numpy] = color

        if draw_detection:
            color = np.array(task["color"])

            min_i, min_j, max_i, max_j = bbox.cpu().numpy().astype(int)
            detections_image_array = cv2.rectangle(detections_image_array, [min_i, min_j], [max_i, max_j], color=color.tolist(), thickness=4)

            text = f'{phrase}: {logit:.2f}'

            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    
            text_offset_x = min_i
            text_offset_y = min_j
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
            
            cv2.rectangle(detections_image_array, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
            cv2.putText(detections_image_array, text, (min_i, min_j - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    blurred_image_array = blurred_image_array[...,::-1]
    segmented_image_array = segmented_image_array[...,::-1]
    detections_image_array = detections_image_array[...,::-1]
    
    #cv2.imwrite("output_blurred.jpg", blurred_image_array)
    #cv2.imwrite("output_segmented.jpg", segmented_image_array)
    #cv2.imwrite("output_detections.jpg", detections_image_array)

    return blurred_image_array, segmented_image_array, detections_image_array


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

def create_topic(writer, topic_name, topic_type, serialization_format='cdr'):
    """
    Create a new topic.
    :param writer: writer instance
    :param topic_name:
    :param topic_type:
    :param serialization_format:
    :return:
    """
    topic_name = topic_name
    topic = rosbag2_py.TopicMetadata(name=topic_name, type=topic_type,
                                     serialization_format=serialization_format)

    writer.create_topic(topic)


def process_bag(bag_path, output_folder, config):

    model = LanguageSAM()

    # Bag Reader
    input_bag_path = str(bag_path)
    input_storage_options, input_converter_options = get_rosbag_options(input_bag_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(input_storage_options, input_converter_options)

    topic_types = reader.get_all_topics_and_types()

    # Create a map for quicker lookup
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    # Bag Writer
    output_bag_path = str(output_folder)
    output_storage_options, output_converter_options = get_rosbag_options(output_bag_path)

    writer = rosbag2_py.SequentialWriter()
    writer.open(output_storage_options, output_converter_options)

    #import debugpy
    #debugpy.listen(5678)
    #debugpy.wait_for_client() 

    topic_count_dict = defaultdict(int)
    subsample_factor = config["subsample_factor"]
    write_blurred_image = config["blurred_image"]
    write_segmented_image = config["segmented_image"]
    write_detections_image = config["detections_image"]

    blurred_suffix = config["blurred_suffix"]
    segmented_suffix = config["segmented_suffix"]
    detections_suffix = config["detections_suffix"]
    topic_pattern = config["topic_pattern"]

    # create topic
    # [create_topic(writer, tname, ttype) for tname, ttype in type_map.items()]

    for tname, ttype in type_map.items():
        create_topic(writer, tname, ttype)

        if "CompressedImage" in ttype:
            blurred_topic = tname.replace("image_raw", "image_raw_" + blurred_suffix).replace("image_rect_color", "image_rect_color_" + blurred_suffix)
            segmented_topic = tname.replace("image_raw", "image_raw_" + segmented_suffix).replace("image_rect_color", "image_rect_color_" + segmented_suffix)
            detections_topic = tname.replace("image_raw", "image_raw_" + detections_suffix).replace("image_rect_color", "image_rect_color_" + detections_suffix)

            print(f"Creating new topic = {blurred_topic}")
            print(f"Creating new topic = {segmented_topic}")
            print(f"Creating new topic = {detections_topic}")

            create_topic(writer, blurred_topic, ttype)
            create_topic(writer, segmented_topic, ttype)
            create_topic(writer, detections_topic, ttype)

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if "image" not in topic or topic_count_dict[topic] % subsample_factor != 0 or re.search(topic_pattern, topic) is None:
            writer.write(topic, data, t)
            topic_count_dict[topic] += 1
            continue

        try:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if not isinstance(msg, CompressedImage):
                raise TypeError(f"Message type {type(msg)} not supported")

            print("Processing compressed image")
            image_data = np.frombuffer(msg.data, np.uint8)
            image_array = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            blurred_topic = topic.replace("image_raw", "image_raw_" + blurred_suffix).replace("image_rect_color", "image_rect_color_" + blurred_suffix)
            segmented_topic = topic.replace("image_raw", "image_raw_" + segmented_suffix).replace("image_rect_color", "image_rect_color_" + segmented_suffix)
            detections_topic = topic.replace("image_raw", "image_raw_" + detections_suffix).replace("image_rect_color", "image_rect_color_" + detections_suffix)

            blurred_image, segmented_image, detections_image = process_image(model, image_array, config["tasks"])

            if write_blurred_image:
                new_msg = CompressedImage()
                new_msg.header = msg.header
                new_msg.format = "jpeg"
                new_msg.data = np.array(cv2.imencode('.jpg', blurred_image)[1]).tostring()
                writer.write(blurred_topic, serialize_message(new_msg), t)

            if write_segmented_image:
                new_msg = CompressedImage()
                new_msg.header = msg.header
                new_msg.format = "jpeg"
                new_msg.data = np.array(cv2.imencode('.jpg', segmented_image)[1]).tostring()
                writer.write(segmented_topic, serialize_message(new_msg), t)

            if write_detections_image:
                new_msg = CompressedImage()
                new_msg.header = msg.header
                new_msg.format = "jpeg"
                new_msg.data = np.array(cv2.imencode('.jpg', detections_image)[1]).tostring()
                writer.write(detections_topic, serialize_message(new_msg), t)

            msg_count = topic_count_dict[topic]
            print(f"[{t}] Transformed messages {msg_count} - Topic: {topic}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
        
        writer.write(topic, data, t)
        topic_count_dict[topic] += 1

    del writer

    

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--config', required=True)
    
    args = parser.parse_args()
    
    bag_paths = sorted(Path(args.input_path).glob("*.db3"))

    print(f"bags = {bag_paths}")

    with open(args.config) as f:
        config = json.load(f)

    process_bag(Path(args.input_path), Path(args.output_path), config)

    #for bag_path in bag_paths:
    #    print(f"Processing: {bag_path}")
        