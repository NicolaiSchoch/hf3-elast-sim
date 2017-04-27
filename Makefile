CXX=mpicxx
#HIFLOW_DIR=/usr/local
HIFLOW_DIR=/home/nschoch/Workspace/HiFlow3/hiflow3_current/build/src
ILUPP_DIR=/home/nschoch/Workspace/HiFlow3/ILU++_1.1.1
CSV = ../../../
#INCLUDE = /home/schoch/Development/include
#LIB = /home/schoch/Development/lib
#USR = /usr
HIFLOW_CONTRIB=/home/nschoch/Workspace/HiFlow3/hiflow3_current/hiflow/contrib
METIS_DIR=/home/nschoch/Workspace/Programme/metis-5.1.0/build/libmetis/
EIGEN_DIR=/home/nschoch/Workspace/Programme/Eigen3_Lib

#CXXFLAGS=-I$(HIFLOW_DIR)/include/hiflow3 -g -pg -fopenmp
#CXXFLAGS=-I$(HIFLOW_DIR)/include/hiflow3 -O3 -g -pg -fopenmp
#CXXFLAGS=-I$(HIFLOW_DIR)/include/hiflow3 -O3 -fopenmp
CXXFLAGS=-g -I$(HIFLOW_DIR)/include -I$(HIFLOW_CONTRIB)/boost_libraries -I$(HIFLOW_CONTRIB) -I$(HIFLOW_CONTRIB)/boost_libraries/boost/tr1 -I/home/nschoch/Workspace/Programme/metis-5.1.0/include -I$(EIGEN_DIR) -O3 -fopenmp -std=c++11
#-I$(ILUPP_DIR)/lib

elastic: elasticity.o
	$(CXX) -o elasticity elasticity.o $(CXXFLAGS) -lm -L$(HIFLOW_DIR) -lhiflow $(METIS_DIR)/libmetis.a -lpthread rotation.o
#-L$(ILUPP_DIR) -liluplusplus-1.1 -lmetis

elasticity.o: elasticity.h

.PHONY clean:
	- rm elasticity.o elasticity SimResults/*_debug_log SimResults/*_info_log SimResults/*.vtu SimResults/*.pvtu

.PHONY dataclean:
	- rm SimResults/*_debug_log SimResults/*_info_log SimResults/*.vtu SimResults/*.pvtu
#	- rm ./*/*/*.*vtu ./*/*/*.h5

