#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import numpy as np
from sympy.matrices import Matrix



def h_transform(alp, a, t, d):
    r_x = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(alp, 0., 0.))
    t_x = tf.transformations.translation_matrix([a, 0., 0.])
    r_z   = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(0.0, 0.0, t))
    t_z = tf.transformations.translation_matrix([0.0, 0.0, d])
    return tf.transformations.concatenate_matrices(r_x, t_x, r_z, t_z)

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))


    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        degree_to_radians = np.pi/180.
        radians_to_degree = 180./np.pi

        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

        # Joint angles
        alpha01 = 0.
        alpha12 = -90. * degree_to_radians
        alpha23 = 0.
        alpha34 = -90. * degree_to_radians
        alpha45 = -90. * degree_to_radians
        alpha56 = -90. * degree_to_radians
        alpha67 = 0.

        # Distances between zi-1 to z along x
        z01 = 0.0; z12 = 0.35; z23 = 1.25; z34 = -0.054; z45 = 0.; z56 = 0.; z67 = 0.

        # Distances between xi-1 to x along z
        x01 = 0.75; x12 = 0.; x23 = 0.; x34 = 1.5; x45 = 0.; x56 = 0.; x67 = 0.45

        dhParams = {
		   alpha0: alpha01, a0: z01, d1: x01,
		   alpha1: alpha12, a1: z12, d2: x12,
		   alpha2: alpha23, a2: z23, d3: x23,
		   alpha3: alpha34, a3: z34, d4: x34,
		   alpha4: alpha45, a4: z45, d5: x45,
		   alpha5: alpha56, a5: z56, d6: x56,
	  	   alpha6: alpha67, a6: z67, d7: x67,
	           }

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

	    # Extract end-effector position and orientation from request
	    # px,py,pz = end-effector position
	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])


            p = np.array([px, py, pz])
            r = tf.transformations.euler_matrix(roll, pitch, yaw)
            q = tf.transformations.quaternion_matrix([req.poses[x].orientation.x, req.poses[x].orientation.y, req.poses[x].orientation.z, req.poses[x].orientation.w])
            t = np.dot(tf.transformations.translation_matrix(p), q)

            # correcting transforms
            t1 = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(0., 0., np.pi))

            t2 = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(0., -np.pi/2., 0.))

            t_dh = np.dot(t, np.linalg.inv(np.dot(t1, t2)))
            p = t_dh[:3, 3]
            r = t_dh[:3, :3]

            p05 = p - dhParams['d7'] * r[:, 2]
            theta_1 = np.arctan2(p05[1], p05[0])


            t01 = h_transform(dhParams['alpha0'], dhParams['a0'], theta_1, dhParams['d1'])
            t12 = h_transform(dhParams['alpha1'], dhParams['a1'], -np.pi/2., dhParams['d2'])

            t02 = np.dot(t01, t12)
            p02 = t02[:3, 3]
            r02 = t02[:3, :3]

            t20 = np.linalg.inv(t02)
            r20 = np.linalg.inv(r02)

            p25 = p05 - p02 # Base

            p25_2 = np.dot(r20, p25) # link 5 from point of frame 2
            d3_5 = np.sqrt(dhParams['a3'] ** 2 + (dhParams['d4'] + dhParams['d5']) ** 2)

            l25_2 = np.linalg.norm(p25_2)
            beta_1 = np.arctan2(p25_2[0], p25_2[1])
            beta_2 = np.arccos((l25_2 ** 2 + dhParams['a2'] ** 2 - d3_5 ** 2)/(2. * dhParams['a2'] * l25_2))

            theta_2 = np.pi/2. - (beta_1 + beta_2)

            phi = np.arccos((dhParams['a2'] ** 2 + d3_5 ** 2 - l25_2 ** 2)/(2. * dhParams['a2'] * d3_5))
            alpha = np.arctan2(-dhParams['a3'], dhParams['d4'])
            theta_3 = np.pi/2. - (phi + alpha)


	        t12 = h_transform(dhParams['alpha1'], dhParams['a1'], theta_2 - np.pi/2, dhParams['d2'])
            t23 = h_transform(dhParams['alpha2'], dhParams['a2'], theta_3, dhParams['d3'])
            t02 = np.dot(t01, t12)
            t03 = np.dot(t02, t23)

            r03 =t03[:3, :3]
            r36 = np.dot(np.linalg.inv(r03),r)

            theta_4 = np.arctan2(r36[2,2],-r36[0,2])

            theta_5 = np.arctan2(np.sqrt(r36[1,0]**2 + r36[1,1]**2), r36[1,2])

            theta_6 = np.arctan2(-r36[1,1],r36[1,0])

            if np.sin(theta_5) < 0. :
                theta_4 = np.arctan2(-r36[2,2],r36[0,2])
                theta_6 = np.arctan2(r36[1,1],-r36[1,0])
            if np.allclose(theta_5, 0.):
                theta_4 = 0.
		        theta_6 = np.arctan2(-r36[0,1],-r36[2,1])


            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
	    joint_trajectory_point.positions = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
	    joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
