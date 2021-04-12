#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   SlideBars.py
@Time    :   2021/03/20 16:36:57
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import pybullet as p


class SlideBars():
    def __init__(self, Id):
        self.Id = Id
        self.motorNames = []
        self.motorIndices = []
        self.motorLowerLimits = []
        self.motorUpperLimits = []
        self.slideIds = []

        self.numJoints = p.getNumJoints(self.Id)

    def add_slidebars(self):
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.Id, i)
            jointName = jointInfo[1].decode('ascii')
            qIndex = jointInfo[3]
            lowerLimits = jointInfo[8]
            upperLimits = jointInfo[9]
            if qIndex > -1:
                self.motorNames.append(jointName)
                self.motorIndices.append(i)
                self.motorLowerLimits.append(lowerLimits)
                self.motorUpperLimits.append(upperLimits)

        for i in range(len(self.motorIndices)):
            if self.motorLowerLimits[i] <= self.motorUpperLimits[i]:
                slideId = p.addUserDebugParameter(self.motorNames[i],
                                                  self.motorLowerLimits[i],
                                                  self.motorUpperLimits[i], 0)
            else:
                slideId = p.addUserDebugParameter(self.motorNames[i],
                                                  self.motorUpperLimits[i],
                                                  self.motorLowerLimits[i], 0)
            self.slideIds.append(slideId)

        return self.motorIndices

    def get_slidebars_values(self):
        slidesValues = []
        for i in self.slideIds:
            value = p.readUserDebugParameter(i)
            slidesValues.append(value)
        return slidesValues