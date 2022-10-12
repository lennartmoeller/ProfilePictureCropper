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
    def __init__(self, path: str, model: FaceAnalysis):
        self.__face = None
        self.__image = Image.open(path)
        self.__model = model
        self.__dest_path = f'{os.path.splitext(path)[0]}-ppc'
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

    def crop(self, width: int = 2, height: int = 3, height_face_scale: float = 0.35, x_face_pos: float = 0.5,
             y_face_pos: float = 0.35):
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
        dest_height = (row_end - row_begin)
        dest_width = dest_height / height * width
        if dest_width > self.width or dest_height > self.height:
            raise Exception('Cropped image size bigger than original image size')
        # add margin left and right to the image to match the ratio
        x_append = (dest_width - self.__face['w']) / 2
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
        # reset props, crop image
        self.__image = self.__image.crop((int(col_begin), int(row_begin), int(col_end), int(row_end)))
        self.width = dest_width
        self.height = dest_height

    def resize(self, width: int = 600, height: int = 900):
        if self.height > height and self.width > width:
            self.__image = self.__image.resize((width, height))
            self.__dest_path += f'-{width}-{height}'
            self.width = width
            self.height = height

    def finalize(self, convert: str = 'jpg'):
        if convert != 'jpg':
            self.__image = self.__image.convert('RGB')
        self.__dest_path += f'.{convert}'
        if os.path.exists(self.__dest_path):
            os.remove(self.__dest_path)
        self.__image.save(self.__dest_path, convert if convert != 'jpg' else None)


def init_argparser():
    parser = argparse.ArgumentParser()
    # positional
    parser.add_argument('image_paths', nargs='*', type=str,
                        help='Paths to all jpg images to crop. Is not required if [--folder] is set.')
    # optional
    parser.add_argument('--width', type=int,
                        help='The width of the output images in px. If [--no-resize] is set, height and width are the ratio of the cropped images. If no width or no height is set, the ratio of the original image is used.')
    parser.add_argument('--height', type=int,
                        help='The height of the output images in px. If [--no-resize] is set, height and width are the ratio of the cropped images. If no width or no height is set, the ratio of the original image is used.')
    parser.add_argument('--scale', default=0.35, type=float, help='Size of the face as a percentage of height [0..1]')
    parser.add_argument('--xfacepos', default=0.5, type=float,
                        help='Horizontal face position as a percentage of image width [0..1]')
    parser.add_argument('--yfacepos', default=0.35, type=float,
                        help='Vertical face position as a percentage of image height [0..1]')
    parser.add_argument('--folder_path', type=str, help='Path to a folder where the jpg images are in')
    parser.add_argument('--convert', type=str, default='jpg',
                        help='Converts images to the given file format')
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
        logging.debug(f' Process {image_path}')
        ppc = ProfilePicture(image_path, model)
        if args.width is None or args.height is None:
            args.width = ppc.width
            args.height = ppc.height
            args.no_resize = True
        ppc.crop(args.width, args.height, args.scale, args.xfacepos, args.yfacepos)
        if not args.no_resize:
            ppc.resize(args.width, args.height)
        ppc.finalize(args.convert)
    logging.debug(f' Runtime: {(time.time() - start_time)} seconds')
