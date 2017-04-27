/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   rotation.h
 * Author: fabian
 *
 * Created on April 4, 2017, 3:00 PM
 */

#ifndef ROTATION_H
#define ROTATION_H

#include <iostream>
#include <cmath>
#include <limits>
#include <iomanip>
#include <type_traits>
#include <algorithm>
#include <assert.h>

#include <chrono>
#include <ctime>

#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>


/* This is from cppreference.com std::numeric_limits::epsilon */
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp);


/* Determines the sign of object of type T */
template<typename T>
int sign( T x );

void center_of_mass( Eigen::MatrixXd const& M, Eigen::VectorXd& center );

Eigen::VectorXd center_of_mass( Eigen::MatrixXd const& M );

void translate( Eigen::MatrixXd& M, Eigen::VectorXd const& t );

void move_center_to_origin( Eigen::MatrixXd& M );

// returns true if unitary matrix R is reflection
bool isReflection( Eigen::MatrixXd const& R );


/*
 * Find rotation matrix R such that the Euclidean distance between R*P and Q is 
 * minimal, according to Kabsch' algorithm.
 * 
 * Parameter list:
 * - [Space dimension] x [number of points] matrices of points P and Q
 * - reference to rotation matrix R
 */
void find_rotation( Eigen::MatrixXd const& P, Eigen::MatrixXd const& Q, Eigen::MatrixXd& R );

void test();

#endif /* ROTATION_H */

