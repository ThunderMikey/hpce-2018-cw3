CPPFLAGS += -I include
CPPFLAGS += -W -Wall
CPPFLAGS += -std=c++11
CPPFLAGS += -O3

LDLIBS += -lOpenCL

SHELL:=/bin/bash
MW_EXE=bin/make_world
SW_EXE=bin/step_world

all : bin/make_world bin/render_world bin/step_world

bin/% : src/%.cpp src/heat.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/test_opencl : src/test_opencl.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL

.PHONY: all test_v1 test_v2 \
	test_v3

test_v1: bin/yc12015/step_world_v1_lambda \
	$(MW_EXE) $(SW_EXE)
	$(MW_EXE) 10 0.1 | $(SW_EXE) 0.1 1000 \
		| diff - <($(MW_EXE) 10 0.1 | $< 0.1 1000)

test_v2: bin/yc12015/step_world_v2_function \
	$(MW_EXE) $(SW_EXE)
	$(MW_EXE) 10 0.1 | $(SW_EXE) 0.1 1000 \
		| diff - <($(MW_EXE) 10 0.1 | $< 0.1 1000)

test_v3: bin/yc12015/step_world_v3_opencl \
	$(MW_EXE) $(SW_EXE)
	# expect floating point in-accuracy
	$(MW_EXE) 10 0.1 | $(SW_EXE) 0.1 1000 \
		| diff - <($(MW_EXE) 10 0.1 | $< 0.1 1000)
	#$(MW_EXE) 10 0.1 | $< 0.1 1000
