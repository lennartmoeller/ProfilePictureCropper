import os

import numpy
from PIL import Image
from insightface.app import FaceAnalysis


class ProfilePicture:
    def __init__(self, path: str, model: FaceAnalysis):
        self.__face = None
        self.__image = Image.open(path)
        self.__model = model
        (self.width, self.height) = self.__image.size
        dest_dir = f'{os.path.dirname(path)}/ppc'
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        self.__dest_path = f'{dest_dir}/{os.path.splitext(os.path.basename(path))[0]}-ppc'

    def __init_face(self):
        faces = self.__model.get(numpy.asarray(self.__image))  # get faces from image
        # Image.fromarray(self.__model.draw_on(numpy.asarray(self.__image), faces)).save(self.__dest_path + '-face.jpg')
        if len(faces) == 0:
            raise Exception('No face detected...')
        area = lambda e: (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])
        f = faces[0]
        for face in faces:
            if area(f) < area(face):
                f = face
        self.__face = {
            'x1': f.bbox[0],
            'y1': f.bbox[1],
            'x2': f.bbox[2],
            'y2': f.bbox[3],
            'w': f.bbox[2] - f.bbox[0],
            'h': f.bbox[3] - f.bbox[1],
            'eyes': [f.kps[0], f.kps[1]],
            'nose': f.kps[2],
            'mouth': [f.kps[3], f.kps[4]]
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
