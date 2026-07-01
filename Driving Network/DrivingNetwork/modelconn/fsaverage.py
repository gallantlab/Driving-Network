""" Functions to project data to fsaverage surface."""

from enum import IntEnum

import numpy
from numpy.typing import ArrayLike

from typing import Optional, Tuple

class fsaverageResolution(IntEnum):
    """
    An enum for fsaverage resolutions. The underlying values
    are the number of vertices used in that resolution
    """
    fsaverage5 = 10242
    fsaverage6 = 40962
    fsaverage7 = 163842
    fsaverage = 163842


class Converter(object):
    """
    A class that converts from subject native space to fsaverage
    """
    def __init__(self, projections: Tuple[ArrayLike, ArrayLike]) -> None:
        """
        Constructor

        :param projections: 	 tuple of <left hemisphere, right hemisphere> projection matrices
        """
        self.projections = projections


    def to_fsaverage(self, data: numpy.ndarray,
                     resolution: fsaverageResolution = fsaverageResolution.fsaverage6) -> numpy.ndarray:
        """
        Projects data to fsaverage

        :param data: 		data to project
        :param resolution: 	fsaverage resolution to project to
        :return: 	data projected to fsaverage
        """
        if (len(data.shape) > 1):
            transpose = (self.projections[0].shape[1] + self.projections[1].shape[1]) == data.shape[1]
        else:
            transpose = False

        left = (data.transpose() if transpose else data)[:self.projections[0].shape[1]]
        right = (data.transpose() if transpose else data)[self.projections[0].shape[1]:]

        left_fsaverage = (self.projections[0] * left).transpose()
        right_fsaverage = (self.projections[1] * right).transpose()

        if (len(left_fsaverage.shape) == 1):
            left_fsaverage = left_fsaverage[numpy.newaxis, :]
            right_fsaverage = right_fsaverage[numpy.newaxis, :]


        left_fsaverage = left_fsaverage[:, :int(resolution)]
        right_fsaverage = right_fsaverage[:, :int(resolution)]

        vertices = numpy.hstack((left_fsaverage, right_fsaverage)).squeeze()
        return vertices


class FSAverageUpscaler(object):
    """
    An object that upscales fsaverage5/6 vertex data to 7
    """

    def __init__(self, fsaverage5_projections: Optional[Tuple[numpy.ndarray, numpy.ndarray]],
                 fsaverage6_projections: Optional[Tuple[numpy.ndarray, numpy.ndarray]]) -> None:
        """
        Constructor

        :param fsaverage5_projections:	tuple of <left hemisphere, right hemisphere> projection matrices from fs5 to fs7
        :param fsaverage6_projections: 	tuple of <left hemisphere, right hemisphere> projection matrices from fs6 to fs7
        """

        self.fsaverage5 = Converter(fsaverage5_projections)
        self.fsaverage6 = Converter(fsaverage6_projections)


    def to_fsaverage(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Converts data to fsaverage7 space

        :param data: data to convert
        :return: fsaverage7 data
        """
        fsaverage5 = {20484, 10242}
        fsaverage6 = {81924, 40962}
        if len(fsaverage5.intersection(set(data.shape))) > 0:
            return self.fsaverage5.to_fsaverage(data)
        elif len(fsaverage6.intersection(set(data.shape))) > 0:
            return self.fsaverage6.to_fsaverage(data)
        else:
            raise ValueError('Data is not in fsaverage5 or fsaverage6 space')

    def to_fsaverage6(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Converts data to fsaverage6 space by projecting to fs7, then truncating to the fs6 vertices
        
        :param data:
        :return:
        """
        fs7 = self.to_fsaverage(data)
        left = fs7[:163842][:40962]
        right = fs7[163842:][:40962]
        return numpy.hstack((left, right))