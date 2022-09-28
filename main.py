import argparse
import os
from typing import Dict

import cv2
from PIL import Image
from insightface.app import FaceAnalysis


class ProfilePicture:
    def __init__(self, img_path: str):
        self.__img_path = img_path
        self.__img_path_without_extension = os.path.splitext(img_path)[0]
        self.__img_read = None
        self.__face_coordinates = None

    def __calc_face_coordinates(self) -> Dict[str, float]:
        if self.__img_read is None:
            self.__img_read = cv2.imread(self.__img_path)
        app = FaceAnalysis(allowed_modules=['detection'])  # enable detection model only
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(self.__img_read)  # get faces from image
        if len(faces) != 1:
            raise Exception("Too many faces")
        return {
            'x1': faces[0].bbox[0],
            'y1': faces[0].bbox[1],
            'x2': faces[0].bbox[2],
            'y2': faces[0].bbox[3]
        }

    def __get_image_dimensions(self) -> Dict[str, int]:
        if self.__img_read is None:
            self.__img_read = cv2.imread(self.__img_path)
        [h, w, _] = self.__img_read.shape
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
        # check if end height or end weight is bigger than the original image
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
        if self.__img_read is None:
            self.__img_read = cv2.imread(self.__img_path)
        crop = self.__img_read[int(row_begin):int(row_end), int(col_begin):int(col_end)]
        cv2.imwrite(target_path, crop)

    def resize_and_convert_jpg_to_webp(self, src_path: str, target_width: int, target_height: int,
                                       dest_path: str = None):
        im = Image.open(src_path).convert('RGB')
        (cropped_width, cropped_height) = im.size
        if cropped_height > target_height and cropped_width > target_width:
            im = im.resize((target_width, target_height))
        if dest_path is None:
            dest_path = '{0}-ppc-{1}-{2}.webp'.format(self.__img_path_without_extension, target_width, target_height)
        im.save(dest_path, 'webp')


def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image", type=str)
    parser.add_argument("width", help="Width of the output image in px", type=int)
    parser.add_argument("height", help="Height of the output image in px", type=int)
    parser.add_argument("-s", "--scale", help="Size of the face as a percentage of height [0..1]", nargs='?', const=0.5,
                        type=float)
    parser.add_argument("-x", "--xfacepos", help="Horizontal face position as a percentage of image width [0..1]",
                        nargs='?', const=0.5, type=float)
    parser.add_argument("-y", "--yfacepos", help="Vertical face position as a percentage of image height [0..1]",
                        nargs='?', const=0.5, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_argparser()
    ppc = ProfilePicture(args.image_path)
    crop_image_path = args.image_path + '.tmp.jpg'
    ppc.crop_image(crop_image_path, args.width, args.height, args.scale, args.xfacepos, args.yfacepos)
    ppc.resize_and_convert_jpg_to_webp(crop_image_path, args.width, args.height)
    os.remove(crop_image_path)  # remove temporary file
