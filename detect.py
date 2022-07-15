import argparse
import collections
import csv

# import datetime
import os
import re
import time
from operator import le
from time import gmtime, strftime

import numpy as np

# from tracemalloc import stop
import requests

# from pandas import concat
from decouple import config

import common
import gstreamer

# import svgwrite


url = config("URL")
START_TIME = time.time()

Object = collections.namedtuple("Object", ["id", "score", "bbox"])

# target_items = [
#     "person",
#     "bicycle",
#     "car",
#     "motorcycle",
#     "bus",
#     "truck",
# ]

person_count = []
car_count = []


def load_labels(path):
    p = re.compile(r"\s*(\d+)(.+)")
    with open(path, "r", encoding="utf-8") as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill="black", font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill="white", font_size=font_size))


def generate_svg(
    src_size, inference_size, inference_box, objs, labels, trdata, trackerFlag,
):
    # dwg = svgwrite.Drawing('', size=src_size)
    src_w, src_h = src_size
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    # for y, line in enumerate(text_lines, start=1):
    #     shadow_text(dwg, 10, y*20, line)
    # tracking success with object ID
    if trackerFlag and (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = (
                td[0].item(),
                td[1].item(),
                td[2].item(),
                td[3].item(),
                td[4].item(),
            )
            overlap = 0
            for ob in objs:

                dx0, dy0, dx1, dy1 = (
                    ob.bbox.xmin.item(),
                    ob.bbox.ymin.item(),
                    ob.bbox.xmax.item(),
                    ob.bbox.ymax.item(),
                )
                area = (min(dx1, x1) - max(dx0, x0)) * (min(dy1, y1) - max(dy0, y0))
                if area > overlap:
                    overlap = area
                    obj = ob

            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y * inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = "{}% {} ID:{}".format(
                percent, labels.get(obj.id, obj.id), int(trackID)
            )
            if labels.get(obj.id, obj.id) == "person":
                if trackID in person_count:
                    continue
                person_count.append(trackID)
                print(f"p: {person_count}")

            if labels.get(obj.id, obj.id) == "car":
                if trackID in car_count:
                    continue
                car_count.append(trackID)
                print(f"c: {car_count}")


class BBox(collections.namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])):
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):

    # returns list of detected objects
    boxes = common.output_tensor(interpreter, 0)
    category_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=BBox(
                xmin=np.maximum(0.0, xmin),
                ymin=np.maximum(0.0, ymin),
                xmax=np.minimum(1.0, xmax),
                ymax=np.minimum(1.0, ymax),
            ),
        )

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def main():
    model_dir = "../models"
    model = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
    labels = "coco_labels.txt"
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', help='.tflite model path',
    #                     default=os.path.join(default_model_dir, default_model))
    # parser.add_argument('--labels', help='label file path',
    #                     default=os.path.join(default_model_dir, default_labels))
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="number of categories with highest score to display",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="classifier score threshold"
    )
    parser.add_argument(
        "--videosrc", help="Which video source to use. ", default="/dev/video0"
    )
    parser.add_argument(
        "--videofmt",
        help="Input video format.",
        default="raw",
        choices=["raw", "h264", "jpeg"],
    )
    parser.add_argument(
        "--tracker",
        help="Name of the Object Tracker To be used.",
        default=None,
        choices=[None, "sort"],
    )
    args = parser.parse_args()

    # print(
    #     f"LOADING {os.path.join(default_model_dir, default_model)} model with {os.path.join(default_model_dir, default_labels)}"
    # )

    print("\nLoading model with labels.\n")
    interpreter = common.make_interpreter(os.path.join(model_dir, model))
    interpreter.allocate_tensors()
    labels = load_labels(os.path.join(model_dir, labels))

    w, h, _ = common.input_image_size(interpreter)
    inference_size = (w, h)
    # Average fps over last 30 frames.
    # fps_counter = common.avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box, mot_tracker):
        # nonlocal fps_counter
        # start_time = time.monotonic()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        # // For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = get_output(interpreter, args.threshold, args.top_k)
        # end_time = time.monotonic()
        detections = []  # np.array([])
        for n in range(0, len(objs)):
            element = []  # np.array([])
            element.append(objs[n].bbox.xmin)
            element.append(objs[n].bbox.ymin)
            element.append(objs[n].bbox.xmax)
            element.append(objs[n].bbox.ymax)
            element.append(objs[n].score)  # print('element= ',element)
            detections.append(element)  # print('dets: ',dets)
        # // convert to numpy array #      print('npdets: ',dets)
        detections = np.array(detections)
        trdata = []
        trackerFlag = False
        if detections.any():
            if mot_tracker != None:
                trdata = mot_tracker.update(detections)
                trackerFlag = True
            # text_lines = [
            #     "Inference: {:.2f} ms".format((end_time - start_time) * 1000),
            #     "FPS: {} fps".format(round(next(fps_counter))),
            # ]
        if len(objs) != 0:
            return generate_svg(
                src_size,
                inference_size,
                inference_box,
                objs,
                labels,
                trdata,
                trackerFlag,
            )

        def post_server(person_flow, car_flow, current_time):
            client = requests.session()

            # Retrieve the CSRF token first
            client.get(URL)  # sets cookie
            if "csrftoken" in client.cookies:
                # Django 1.6 and up
                csrftoken = client.cookies["csrftoken"]
            else:
                # older versions
                csrftoken = client.cookies["csrf"]

            num1 = person_flow
            num2 = car_flow

            # print(csrftoken)
            mutation = """mutation{
            multiinsertFlow(inCategory: ["human", "car"], inCoordinates: ["25.0418,121.5344", "25.0418,121.5344"], flow:[%d, %d], time: "%s", affectedrows: 2){
                IsSuccessed,
                status
            }
            }""" % (
                num1,
                num2,
                current_time,
            )

            r = client.post(
                url, data={"query": mutation, "csrfmiddlewaretoken": csrftoken}
            )

        printerLooper = True
        while printerLooper == True:
            initial_time = time.time()
            time.sleep(2)
            end_time = time.time()
            current_time = strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())

            # print(f"{len(person_count)} \t {person_count} \t {end_time - initial_time}")
            person_flow = len(person_count)
            car_flow = len(car_count)

            post_server(person_flow, car_flow, current_time)
            print("post to server suceeded\n")

            person_count.clear()
            car_count.clear()

            printerLooper = False

    result = gstreamer.run_pipeline(
        user_callback,
        src_size=(640, 480),
        appsink_size=inference_size,
        trackerName=args.tracker,
        videosrc=args.videosrc,
        videofmt=args.videofmt,
    )


if __name__ == "__main__":
    print("\nProcessing ... press control C to exit")
    main()
    print("\n\nCoral running %s seconds " % (time.time() - START_TIME))
