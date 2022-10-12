import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager

import numpy
from PIL import Image
from insightface.app import FaceAnalysis


@contextmanager
def suppress_stdout_without_verbose(debug: bool):
    with open(os.devnull, "w") as devnull:
        if debug:
            yield
        else:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


class ProfilePicture:
    def __init__(self, img_path: str, model: FaceAnalysis):
        self.__model = model
        self.__init_image(img_path)
        self.__face = None

    def __init_image(self, img_path):
        self.__image = Image.open(img_path)
        (self.width, self.height) = self.__image.size

    def __init_face(self):
        faces = self.__model.get(numpy.asarray(self.__image))  # get faces from image
        if len(faces) != 1:
            raise Exception('Too many faces')
        self.__face = {
            'x1': faces[0].bbox[0],
            'y1': faces[0].bbox[1],
            'x2': faces[0].bbox[2],
            'y2': faces[0].bbox[3],
            'w': faces[0].bbox[2] - faces[0].bbox[0],
            'h': faces[0].bbox[3] - faces[0].bbox[1]
        }

    def crop_image(self, target_path: str, width: int, height: int, height_face_scale: float = 0.5,
                   x_face_pos: float = 0.5,
                   y_face_pos: float = 0.5):
        if self.__face is None:
            self.__init_face()
        # initially set the crop coordinates
        row_begin = self.__face['y1']
        row_end = self.__face['y2']
        col_begin = self.__face['x1']
        col_end = self.__face['x2']
        # add margin above and under image to match the height_face_scale
        y_append = self.__face['h'] * (1 - height_face_scale) / height_face_scale / 2
        row_begin -= y_append
        row_end += y_append
        # check if end height or end width is bigger than the original image
        real_height = (row_end - row_begin)
        real_width = real_height / height * width
        if real_width > self.width or real_height > self.height:
            raise Exception('Cropped image size bigger than original image size')
        # add margin left and right to the image to match the ratio
        x_append = (real_width - self.__face['w']) / 2
        col_begin -= x_append
        col_end += x_append
        # shift the image in y direction to match the y_face_pos
        y_move = (row_end - row_begin) * (y_face_pos - 0.5)
        row_begin -= y_move
        row_end -= y_move
        # shift the image in x direction to match the x_face_pos
        x_move = (col_end - col_begin) * (x_face_pos - 0.5)
        col_begin -= x_move
        col_end -= x_move
        # move back cropped image if it is above an edge
        if row_begin < 0:
            row_begin = 0
        if col_begin < 0:
            col_begin = 0
        if row_end > self.height:
            row_end = self.height
        if col_end > self.width:
            col_end = self.width
        # crop and save image
        crop = self.__image.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))
        if os.path.exists(target_path):
            os.remove(target_path)
        crop.save(target_path)

    def resize_and_convert(self, src_path: str, dest_path: str, target_width: int, target_height: int,
                           resize: bool = True, convert: str = None):
        im = Image.open(src_path)
        if resize:
            (cropped_width, cropped_height) = im.size
            if cropped_height > target_height and cropped_width > target_width:
                im = im.resize((target_width, target_height))
        if os.path.exists(dest_path):
            os.remove(dest_path)
        if convert is None:
            im.save(dest_path)
        else:
            im = im.convert('RGB')
            im.save(dest_path, convert)


def init_argparser():
    parser = argparse.ArgumentParser()
    # positional
    parser.add_argument('image_paths', nargs='*', type=str,
                        help='Paths to all jpg images to crop. Is not required if [--folder] is set.')
    # optional
    parser.add_argument('--width', type=int,
                        help='The width of the output images in px. If [--no-resize] is set, height and width are the ratio of the cropped images.')
    parser.add_argument('--height', type=int,
                        help='The height of the output images in px. If [--no-resize] is set, height and width are the ratio of the cropped images.')
    parser.add_argument('--scale', default=0.35, type=float, help='Size of the face as a percentage of height [0..1]')
    parser.add_argument('--xfacepos', default=0.5, type=float,
                        help='Horizontal face position as a percentage of image width [0..1]')
    parser.add_argument('--yfacepos', default=0.35, type=float,
                        help='Vertical face position as a percentage of image height [0..1]')
    parser.add_argument('--folder_path', type=str, help='Path to a folder where the jpg images are in')
    parser.add_argument('--convert', type=str, help='Converts images to the given file format')
    parser.add_argument('--model_name', type=str, default='buffalo_sc',
                        help='Choose model which should be used by insightface')
    # flags
    parser.add_argument('--no_resize', action='store_true',
                        help='Keeps the aspect ratio given in width and height without resizing the image')
    parser.add_argument('-v', '--verbose', action='store_true', help='Debug output')
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    args = init_argparser()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)
    image_paths = args.image_paths
    if args.folder_path:
        for r, _, files in os.walk(args.folder_path):
            for f in files:
                if f.endswith('.jpg'):
                    image_paths.append(f'./{r}/{f}')
    with suppress_stdout_without_verbose(args.verbose):
        model = FaceAnalysis(root="./insightface", name=args.model_name, allowed_modules=['detection'])
        model.prepare(ctx_id=0, det_size=(640, 640))
    for image_path in image_paths:
        logging.debug(f'Process {image_path}')
        ppc = ProfilePicture(image_path, model)
        crop_image_path = f'{os.path.splitext(image_path)[0]}-ppc.jpg'
        width = args.width if args.width is not None else ppc.width
        height = args.height if args.height is not None else ppc.height
        ppc.crop_image(crop_image_path, width, height, args.scale, args.xfacepos, args.yfacepos)
        resize = False if args.no_resize else True
        if resize is True or args.convert is not None:
            dest_path = f'{os.path.splitext(image_path)[0]}-ppc'
            if resize is True:
                dest_path += f'-{width}-{height}'
            if args.convert is None:
                dest_path += '.jpg'
            else:
                dest_path += f'.{args.convert}'
            ppc.resize_and_convert(crop_image_path, dest_path, width, height, resize, args.convert)
            os.remove(crop_image_path)  # remove temporary file
    logging.debug(f'Runtime: {(time.time() - start_time)} seconds')
