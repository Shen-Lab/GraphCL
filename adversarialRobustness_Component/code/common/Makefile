dir_guard = @mkdir -p $(@D)

#INTEL_ROOT := /opt/intel
MKL_ROOT = $(INTEL_ROOT)/mkl
TBB_ROOT = $(INTEL_ROOT)/tbb

FIND := find
CXX := g++
CXXFLAGS += -Wall -O3 -std=c++11
LDFLAGS += -lm  -lmkl_rt -ltbb

CUDA_HOME := /usr/local/cuda-9.0
NVCC := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS += --default-stream per-thread
LDFLAGS += -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand -lcusparse

CUDA_ARCH :=  -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50

build_root = _ext
obj_build_root = $(build_root)

include_dirs = $(CUDA_HOME)/include $(MKL_ROOT)/include $(TBB_ROOT)/include include
CXXFLAGS += $(addprefix -I,$(include_dirs))
CXXFLAGS += -fPIC

NVCCFLAGS += $(addprefix -I,$(include_dirs))
NVCCFLAGS += -std=c++11 --use_fast_math --compiler-options '-fPIC'
cu_files = $(shell $(FIND) src/ -name "*.cu" -printf "%P\n")
cu_obj_files = $(subst .cu,.o,$(cu_files))
objs = $(addprefix $(obj_build_root)/,$(cu_obj_files))

DEPS = ${objs:.o=.d}
mylib = _ext/my_lib/_my_lib.so

all: $(objs) $(mylib)

$(obj_build_root)/%.o: src/%.cu
	$(dir_guard)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} -odir $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(mylib): src/*.c src/*.h src/*.cu
	python build.py

clean:
	rm -f $(obj_build_root)/*.o
	rm -f $(obj_build_root)/*.d
	rm -rf _ext
	rm -f functions/*.pyc
	rm -f modules/*.pyc
-include $(DEPS)
