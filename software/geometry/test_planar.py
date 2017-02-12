import matplotlib.pyplot as plt
import numpy as np
import unittest

import planar


class TestFiniteLine(unittest.TestCase):

    def TestLineConstruction(self):
        pt1 = np.array([1.0, 2.0])
        pt2 = np.array([3.0, 4.0])

        line1 = planar.FiniteLine(pt1=pt1, pt2=pt2)
        self.assertTrue(np.allclose(line1.x, np.array([pt1[0], pt2[0]])))
        self.assertTrue(np.allclose(line1.y, np.array([pt1[1], pt2[1]])))

        line2 = planar.FiniteLine(pts=np.hstack((pt1, pt2)))
        self.assertTrue(np.allclose(line1.x, line2.x))
        self.assertTrue(np.allclose(line1.y, line2.y))

    def TestBasicLine(self):
        figure, axis = plt.subplots()
        line = planar.FiniteLine()
        line.plot(axis, color=(0.0, 1.0, 0.0), thickness=3)
        plt.title("Should be a basic line from 0 to 1 (x) (green, thick)")
        plt.show()
        self.assertTrue(True)

    def TestComplexLine(self):
        figure, axis = plt.subplots()
        pts = np.array([1.5, 96.9, -5.3, 43.0])
        line = planar.FiniteLine(pts=pts)
        line.plot(axis, color=(0.0, 0.0, 1.0), thickness=1.0)
        plt.title("Line from {} to {}, thin and blue".format(pts[:2], pts[2:]))
        plt.show()


class TestInfiniteLine(unittest.TestCase):

    def TestBasicLine(self):
        figure, axis = plt.subplots()
        line = planar.InfiniteLine()
        line.plot(axis, color=(0.0, 1.0, 0.0), thickness=3)
        plt.title("Should be a basic line from -10 to 10 (y) (green, thick)")
        plt.show()
        self.assertTrue(True)

    def TestComplexLine(self):
        figure, axis = plt.subplots()
        pt1 = np.random.rand(2) * 10
        pt2 = np.random.rand(2) * 10
        line = planar.InfiniteLine(pt1=pt1, pt2=pt2)
        line.plot(axis, color=(0.0, 0.0, 1.0), thickness=1.0)
        plt.plot(pt1[0], pt1[1], 'g.', markersize=10)
        plt.plot(pt2[0], pt2[1], 'r.', markersize=10)
        plt.title("Line from {} to {}, thin and blue".format(pt1, pt2))
        plt.show()
