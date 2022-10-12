import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager

from insightface.app import FaceAnalysis

from ProfilePicture import ProfilePicture


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


def init_argparser():
    parser = argparse.ArgumentParser()
    # positional
    parser.add_argument('image_paths', nargs='*', type=str,
                        help='Paths to all jpg images to crop. Is not required if [--folder] is set.')
    # optional
    parser.add_argument('--width', type=int,
                        help='The width of the output images in px. If [--no-resize] is set, height and width are the \
                        ratio of the cropped images. If no width or no height is set, the ratio of the original image \
                        is used.')
    parser.add_argument('--height', type=int,
                        help='The height of the output images in px. If [--no-resize] is set, height and width are the \
                        ratio of the cropped images. If no width or no height is set, the ratio of the original image \
                        is used.')
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
