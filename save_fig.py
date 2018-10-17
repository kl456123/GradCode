#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy
import matplotlib.pyplot as plt

a = numpy.zeros((10, 10))
triangle = numpy.array([[1, 3], [4, 8], [1, 9]],
                       numpy.int32)  #[1，3]，[4，8],[1,9]为要填充的轮廓坐标
cv2.fillConvexPoly(a, triangle, 1)

plt.imshow(a)
plt.show()
