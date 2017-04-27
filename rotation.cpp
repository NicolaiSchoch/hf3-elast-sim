// mpicxx -fopenmp   -c -g -I/home/fabian/programs/Eigen_3_3_3 -I/home/fabian/programs/hiflow3/hiflow/contrib/boost_libraries -I/home/fabian/programs/hiflow3/hiflow/contrib/boost_libraries/boost/tr1 -I/home/fabian/programs/hiflow3/hiflow/contrib -I/home/fabian/programs/hiflow3/build/src/include -I/home/fabian/programs/parmetis-4.0.3/include -std=c++11 -MMD -MP -MF "build/Debug/GNU-Linux/rotation.o.d" -o build/Debug/GNU-Linux/rotation.o rotation.cpp

// mpicxx -fopenmp   -c -g -I/home/nschoch/Workspace/Programme/Eigen3_Lib -std=c++11 -o rotation.o rotation.cpp


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "rotation.h"

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
  // unless the result is subnormal
  || std::abs(x-y) < std::numeric_limits<T>::min();
}

template<typename T>
int sign( T x ) {
  if( almost_equal( x, 0.0, 2 ) )   return  0;
  else if( x > 0 )                  return  1;
  else                              return -1;
}

void center_of_mass( Eigen::MatrixXd const& M, Eigen::VectorXd& center ) {
  center = M.rowwise().sum()/M.cols();
}

Eigen::VectorXd center_of_mass( Eigen::MatrixXd const& M ) {
  return M.rowwise().sum()/M.cols();
}

void translate( Eigen::MatrixXd& M, Eigen::VectorXd const& t ) {
  assert( M.rows() == t.size() );
  M = M.colwise() + t;
}

void move_center_to_origin( Eigen::MatrixXd& M ) {
  Eigen::VectorXd c = center_of_mass( M );
  translate( M, -c );
}

// returns true if unitary matrix R is reflection
bool isReflection( Eigen::MatrixXd const& R ) {
  assert( R.isUnitary() );
  if( almost_equal( R.determinant(), -1.0, 2 ) ) 
    return true;
  else
    return false;
}

void find_rotation( Eigen::MatrixXd const& P, Eigen::MatrixXd const& Q, Eigen::MatrixXd& R ) {
  assert( P.rows() == Q.rows() && P.cols() == Q.cols() );

  Eigen::MatrixXd Covariance = P * Q.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd( Covariance, Eigen::ComputeThinU | Eigen::ComputeThinV );
  
  R = svd.matrixV() * svd.matrixU().transpose(); // resulting unitary matrix
  assert( R.isUnitary() );
  
  if( isReflection( R ) ) // if R is reflection, change it to a rotation
  {
    Eigen::VectorXd d;
    d = Eigen::VectorXd::Ones( svd.matrixV().cols() );
    d( d.size()-1 ) = -1.0;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D( d );
    R = svd.matrixV() * D * svd.matrixU().transpose();
  }
}

void test() {
  // define rotation
  Eigen::Matrix3d rot; 
  rot = Eigen::Quaterniond( 10.0, 2.45, 3.8, 4.04 ).normalized();//.toRotationMatrix();
  assert( almost_equal( rot.determinant(), 1.0, 2 ) && rot.isUnitary() ); 
  
  // define vetrices of the tetrahedrons
  Eigen::Vector3d p1( 0.0, 0.0, 0.0 );
  Eigen::Vector3d p2( 1.0, 0.0, 0.0 );
  Eigen::Vector3d p3( 0.0, 1.0, 0.0 );
  Eigen::Vector3d p4( 0.0, 0.0, 1.0 );
  
  // store the vertices in 3x4 matrix
  Eigen::MatrixXd P( 3, 4 );
  P << p1, p2, p3, p4;
  
  move_center_to_origin( P );
  
  Eigen::MatrixXd Q = rot*P; // compute test set
  Eigen::MatrixXd R;
  
  for( int i = 0; i < 100000; ++i )
    find_rotation(P, Q, R);
   
  if( Q.isApprox( R*P ) )
    std::cout << " Q is equal to R*P " << std::endl;
  else
    std::cout << " Something is wrong! " << std::endl;
}
