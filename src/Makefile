BASEDIR := ..
SRC_DIR := $(BASEDIR)/src
OBJ_DIR := $(BASEDIR)/obj
LIB_DIR := $(BASEDIR)/DSMC

CPP_SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
CPP_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRC_FILES))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRC_FILES))
OBJ_FILES=$(CPP_OBJ_FILES) $(CU_OBJ_FILES)

all: $(OBJ_FILES)
	nvcc -arch=sm_50 -shared $(OBJ_FILES) -o $(LIB_DIR)/cuDSMC`python3-config --extension-suffix` -lcudart -lcuda -lnvrtc -lcudadevrt

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	g++ -c -fPIC `python3 -m pybind11 --includes` -std=c++11 $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	nvcc -arch=sm_50 -Xcompiler '-fPIC' -dc -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o 
	rm $(LIB_DIR)/*.so
