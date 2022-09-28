import argparse
import os
from typing import Dict

import cv2
from PIL import Image
from insightface.app import FaceAnalysis


class ProfilePicture:
    def __init__(self, img_path: str):
        self.__img_path = img_path
        self.__cv_img_read = None
        self.__face_coordinates = None

    def __calc_face_coordinates(self) -> Dict[str, float]:
        if self.__cv_img_read is None:
            self.__cv_img_read = cv2.imread(self.__img_path)
        app = FaceAnalysis(allowed_modules=['detection'])  # enable detection model only
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(self.__cv_img_read)  # get faces from image
        if len(faces) != 1:
            raise Exception("Too many faces")
        return {
            'x1': faces[0].bbox[0],
            'y1': faces[0].bbox[1],
            'x2': faces[0].bbox[2],
            'y2': faces[0].bbox[3]
        }

    def __get_image_dimensions(self) -> Dict[str, int]:
        if self.__cv_img_read is None:
            self.__cv_img_read = cv2.imread(self.__img_path)
        [h, w, _] = self.__cv_img_read.shape
        return {'w': w, 'h': h}

    def __get_face_dimensions(self) -> Dict[str, float]:
        if self.__face_coordinates is None:
            self.__face_coordinates = self.__calc_face_coordinates()
        return {
            'w': self.__face_coordinates['x2'] - self.__face_coordinates['x1'],
            'h': self.__face_coordinates['y2'] - self.__face_coordinates['y1']
        }

    def crop_image(self, target_path: str, width: int, height: int, height_face_scale: float = 0.5,
                   x_face_pos: float = 0.5,
                   y_face_pos: float = 0.5):
        face_dim = self.__get_face_dimensions()
        img_dim = self.__get_image_dimensions()
        # initially set the crop coordinates
        row_begin = self.__face_coordinates['y1']
        row_end = self.__face_coordinates['y2']
        col_begin = self.__face_coordinates['x1']
        col_end = self.__face_coordinates['x2']
        # add margin above and under image to match the height_face_scale
        y_append = face_dim['h'] * (1 - height_face_scale) / height_face_scale / 2
        row_begin -= y_append
        row_end += y_append
        # check if end height or end width is bigger than the original image
        real_height = (row_end - row_begin)
        real_width = real_height / height * width
        if real_width > img_dim['w'] or real_height > img_dim['h']:
            raise Exception('Cropped image size bigger than original image size')
        # add margin left and right to the image to match the ratio
        x_append = (real_width - face_dim['w']) / 2
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
        if row_end > img_dim['h']:
            row_end = img_dim['h']
        if col_end > img_dim['w']:
            col_end = img_dim['w']
        # crop and save image
        if self.__cv_img_read is None:
            self.__cv_img_read = cv2.imread(self.__img_path)
        crop = self.__cv_img_read[int(row_begin):int(row_end), int(col_begin):int(col_end)]
        cv2.imwrite(target_path, crop)

    def resize_and_convert(self, src_path: str, dest_path: str, target_width: int, target_height: int,
                           resize: bool = True, convert: str = None):
        im = Image.open(src_path)
        if resize:
            (cropped_width, cropped_height) = im.size
            if cropped_height > target_height and cropped_width > target_width:
                im = im.resize((target_width, target_height))
        if convert is None:
            im.save(dest_path)
        else:
            im = im.convert('RGB')
            im.save(dest_path, convert)


def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("width", type=int,
                        help="In resize mode the width of the output images in px, otherwise the ratio of the output images")
    parser.add_argument("height", type=int,
                        help="In convert mode the height of the output images in px, otherwise the ratio of the output images")
    parser.add_argument("image_paths", nargs="*", type=str, help="Paths to the jpg images")
    parser.add_argument("-s", "--scale", nargs='?', const=0.5, type=float,
                        help="Size of the face as a percentage of height [0..1]")
    parser.add_argument("-x", "--xfacepos", nargs='?', const=0.5, type=float,
                        help="Horizontal face position as a percentage of image width [0..1]")
    parser.add_argument("-y", "--yfacepos", nargs='?', const=0.5, type=float,
                        help="Vertical face position as a percentage of image height [0..1]")
    parser.add_argument("-f", "--folder_path", type=str, help="Path to a folder where the jpg images are in")
    parser.add_argument("-c", "--convert", type=str, help="Converts images to the given file format")
    parser.add_argument("--no_resize", action='store_true',
                        help="Keeps the aspect ratio given in width and height without resizing the image")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_argparser()
    image_paths = args.image_paths
    if args.folder_path:
        for f in os.listdir(args.folder_path):
            if f.endswith('.jpg'):
                image_paths.append(args.folder_path + '/' + f)
    file_path_template = '{0}-ppc-{1}-{2}.{3}'
    for image_path in image_paths:
        ppc = ProfilePicture(image_path)
        crop_image_path = '{0}-ppc.{1}'.format(os.path.splitext(image_path)[0], 'jpg')
        ppc.crop_image(crop_image_path, args.width, args.height, args.scale, args.xfacepos, args.yfacepos)
        resize = False if args.no_resize else True
        if resize is True or args.convert is not None:
            dest_path = os.path.splitext(image_path)[0] + '-ppc'
            if resize is True:
                dest_path += '-' + str(args.width) + '-' + str(args.height)
            if args.convert is None:
                dest_path += '.jpg'
            else:
                dest_path += '.' + args.convert
            ppc.resize_and_convert(crop_image_path, dest_path, args.width, args.height, resize, args.convert)
            os.remove(crop_image_path)  # remove temporary file
