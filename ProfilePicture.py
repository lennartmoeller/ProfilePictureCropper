import os

import numpy
from PIL import Image
from insightface.app import FaceAnalysis

AVG_EYE_HEIGHT_PERCENTAGE = 0.405
AVG_MOUTH_HEIGHT_PERCENTAGE = 0.725
AVG_FACE_WIDTH_HEIGHT_RATIO = 1.375


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
        if len(faces) == 0:
            raise Exception('No face detected...')
        area = lambda e: (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])
        f = faces[0]
        for face in faces:
            if area(f) < area(face):
                f = face
        # bboxes by insightface
        face_tops = {'insightface': f.bbox[1]}
        face_bottoms = {'insightface': f.bbox[3]}
        # bboxes by eye and mouth position
        mouth_y = (f.kps[3][1] + f.kps[4][1]) / 2
        eye_y = (f.kps[0][1] + f.kps[1][1]) / 2
        face_height = (mouth_y - eye_y) / (AVG_MOUTH_HEIGHT_PERCENTAGE - AVG_EYE_HEIGHT_PERCENTAGE)
        face_tops['eye_mouth'] = eye_y - face_height * AVG_EYE_HEIGHT_PERCENTAGE
        face_bottoms['eye_mouth'] = eye_y + face_height * (1 - AVG_EYE_HEIGHT_PERCENTAGE)
        # bboxes by face width
        face_height = (f.bbox[2] - f.bbox[0]) * AVG_FACE_WIDTH_HEIGHT_RATIO
        face_tops['face_width'] = eye_y - face_height * AVG_EYE_HEIGHT_PERCENTAGE
        face_bottoms['face_width'] = eye_y + face_height * (1 - AVG_EYE_HEIGHT_PERCENTAGE)
        # calculate average
        face_top = sum(face_tops.values()) / len(face_tops)
        face_bottom = sum(face_bottoms.values()) / len(face_bottoms)
        self.__face = {
            'left': f.bbox[0],
            'top': face_top,
            'top_options': face_tops,
            'right': f.bbox[2],
            'bottom': face_bottom,
            'bottom_options': face_bottoms,
            'width': f.bbox[2] - f.bbox[0],
            'height': face_bottom - face_top,
            'eyes': [{'x': f.kps[0][0], 'y': f.kps[0][1]}, {'x': f.kps[1][0], 'y': f.kps[1][1]}],
            'nose': {'x': f.kps[2][0], 'y': f.kps[2][1]},
            'mouth': [{'x': f.kps[3][0], 'y': f.kps[3][1]}, {'x': f.kps[4][0], 'y': f.kps[4][1]}],
        }

    def draw_face_detections(self):
        from PIL import ImageDraw
        self.__init_face()
        draw = ImageDraw.Draw(self.__image)
        draw_rect = lambda c, x1, y1, x2, y2: draw.rectangle((x1, y1, x2, y2), outline=c, width=5)
        eye_pos = (self.__face['eyes'][0]['y'] + self.__face['eyes'][1]['y']) / 2
        draw_rect((255, 255, 0), 0, eye_pos, self.width, eye_pos)
        draw_rect((0, 255, 255), self.__face['left'], self.__face['top_options']['insightface'], self.__face['right'],
                  self.__face['bottom_options']['insightface'])
        draw_rect((0, 255, 0), self.__face['left'], self.__face['top_options']['eye_mouth'], self.__face['right'],
                  self.__face['bottom_options']['eye_mouth'])
        draw_rect((0, 0, 255), self.__face['left'], self.__face['top_options']['face_width'], self.__face['right'],
                  self.__face['bottom_options']['face_width'])
        draw_rect((255, 0, 0), self.__face['left'], self.__face['top'], self.__face['right'], self.__face['bottom'])

    def crop(self, width: int = 2, height: int = 3, height_face_scale: float = 0.35, x_face_pos: float = 0.5,
             y_face_pos: float = 0.35):
        if self.__face is None:
            self.__init_face()
        # initially set the crop coordinates
        row_begin = self.__face['top']
        row_end = self.__face['bottom']
        col_begin = self.__face['left']
        col_end = self.__face['right']
        # add margin above and under image to match the height_face_scale
        y_append = self.__face['height'] * (1 - height_face_scale) / height_face_scale / 2
        row_begin -= y_append
        row_end += y_append
        # check if end height or end width is bigger than the original image
        dest_height = (row_end - row_begin)
        dest_width = dest_height / height * width
        if dest_width > self.width or dest_height > self.height:
            raise Exception('Cropped image size bigger than original image size')
        # add margin left and right to the image to match the ratio
        x_append = (dest_width - self.__face['width']) / 2
        col_begin -= x_append
        col_end += x_append
        # shift the image in y direction
        current_eye_pos = (self.__face['eyes'][0]['y'] + self.__face['eyes'][1]['y']) / 2
        desired_eye_pos = self.__face['height'] * AVG_EYE_HEIGHT_PERCENTAGE + self.__face['top']
        y_move = current_eye_pos - desired_eye_pos  # shift to have all eyes in line
        y_move += (row_end - row_begin) * (0.5 - y_face_pos)  # shift to match y_face_pos
        row_begin += y_move
        row_end += y_move
        # shift the image in x direction to match the x_face_pos
        x_move = (col_end - col_begin) * (0.5 - x_face_pos)
        col_begin += x_move
        col_end += x_move
        # move back cropped image if it is above an edge
        if row_begin < 0:
            row_begin = 0
            # TODO log image shifting
        if col_begin < 0:
            col_begin = 0
            # TODO log image shifting
        if row_end > self.height:
            row_end = self.height
            # TODO log image shifting
        if col_end > self.width:
            col_end = self.width
            # TODO log image shifting
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
