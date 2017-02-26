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

    def TestLineConstruction(self):
        pt1 = np.array([1.0, 2.0])
        pt2 = np.array([3.0, 4.0])

        line = planar.FiniteLine(pt1=pt1, pt2=pt2)
        self.assertTrue(np.allclose(line.getMidpoint(), np.array([2, 3])))

        tupleMid = line.getMidpoint(returnTupleInt=True)
        self.assertTrue(isinstance(tupleMid, tuple))
        self.assertTrue(tupleMid == (2, 3))

    def TestAverageLine(self):
        pts1 = [0.0, 0.0, 10.0, 0.0]
        line1 = planar.FiniteLine(pts=pts1)

        pts2 = [0.0, 2.0, 10.0, 4.0]
        line2 = planar.FiniteLine(pts=pts2)

        pts3 = [10.0, 6.0, 0.0, 2.0]
        line3 = planar.FiniteLine(pts=pts3)

        line4 = line1.average(line2)
        line5 = line1.average(line3)

        figure, axis = plt.subplots()
        line1.plot(axis, color=(1.0, 0.0, 0.0), thickness=1)
        line2.plot(axis, color=(1.0, 0.0, 0.0), thickness=1)
        line3.plot(axis, color=(0.0, 0.0, 1.0), thickness=1)
        line4.plot(axis, color=(1.0, 0.0, 0.0), thickness=2)
        line5.plot(axis, color=(1.0, 0.0, 1.0), thickness=2)
        plt.title("1+2=4, 1+3=5?")
        plt.show()

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

    def TestInfiniteFromFinite(self):
        pt1 = np.array([-1.0, 1.0])
        pt2 = np.array([1.0, 1.0])
        fLine = planar.FiniteLine(pt1=pt1, pt2=pt2)
        iLine = planar.InfiniteLine(fLine=fLine)
        self.assertTrue(np.allclose(iLine.normal, np.array([0.0, 1.0])))
        self.assertTrue(np.isclose(iLine.bias, 1.0))

    def TestIntersection(self):
        p11 = np.array([0.0, 0.0])
        p12 = np.array([1.0, 1.0])
        p21 = np.array([2.0, 0.0])
        p22 = np.array([3.0, 1.0])
        iLine1 = planar.InfiniteLine(pt1=p11, pt2=p12)
        iLine2 = planar.InfiniteLine(pt1=p21, pt2=p22)
        result = iLine1.intersection(iLine2)
        self.assertTrue(result is None)

        p31 = np.array([0.0, -3.0])
        p32 = np.array([3.0, 3.0])
        iLine3 = planar.InfiniteLine(pt1=p31, pt2=p32)
        result1 = iLine3.intersection(iLine1)
        result2 = iLine1.intersection(iLine3)
        print("result1: {}".format(result1))
        print("result2: {}".format(result2))
        print("p32: {}".format(p32))
        self.assertTrue(np.allclose(result1, result2))
        self.assertTrue(np.allclose(result1, p32))
