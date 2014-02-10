PATH_TO_EIGEN_LIB:=$(HOME)/Main/local/include/
CXXFLAGS:=-std=c++0x 
CXX:=g++

test: test.cc em_gmm.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ -I$(PATH_TO_EIGEN_LIB)
