CPPFLAGS += -I include
CPPFLAGS += -W -Wall
CPPFLAGS += -std=c++11
CPPFLAGS += -O3

LDLIBS += -lOpenCL

SHELL:=/bin/bash
MW_EXE=bin/make_world
SW_EXE=bin/step_world
W_BIN=/tmp/world.bin
V3_EXE := bin/yc12015/step_world_v3_opencl
V4_EXE := bin/yc12015/step_world_v4_double_buffered
V5_EXE := bin/yc12015/step_world_v5_packed_properties

time_it = time -p (cat $(W_BIN) | $(1) 0.1 500 1 > /dev/null)

all : bin/make_world bin/render_world bin/step_world

bin/% : src/%.cpp src/heat.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/test_opencl : src/test_opencl.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL

.PHONY: all test_v1 test_v2 \
	test_v3 \
	test_v4 \
	test_v5 \
	compare_v3_v4_v5

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
	$(MW_EXE) 10 0.1 1 > $(W_BIN)
	# expect floating point in-accuracy
	-cat $(W_BIN) | $(SW_EXE) 0.1 1000 \
		| diff - <(cat $(W_BIN) | $< 0.1 1000)
	#$(MW_EXE) 10 0.1 | $< 0.1 1000
	$(call time_it,$(SW_EXE))
	$(call time_it,$<)

test_v4: bin/yc12015/step_world_v4_double_buffered \
	$(MW_EXE) $(SW_EXE)
	# produce world binary file
	$(MW_EXE) 100 0.1 1 > $(W_BIN)
	# expect floating point in-accuracy
	-cat $(W_BIN) | $(SW_EXE) 0.1 10 0 \
		| diff - <(cat $(W_BIN) | $< 0.1 10 0)
	time -p (cat $(W_BIN) | $(SW_EXE) 0.1 1000 1 > /dev/null)
	time -p (cat $(W_BIN) | $< 0.1 1000 1 > /dev/null)

test_v5: bin/yc12015/step_world_v5_packed_properties \
	$(MW_EXE) $(SW_EXE)
	# produce world binary file
	$(MW_EXE) 10 0.1 1 > $(W_BIN)
	# expect floating point in-accuracy
	-cat $(W_BIN) | $(SW_EXE) 0.1 100 0 \
		| diff - <(cat $(W_BIN) | $< 0.1 100 0)
	time -p (cat $(W_BIN) | $(SW_EXE) 0.1 1000 1 > /dev/null)
	time -p (cat $(W_BIN) | $< 0.1 1000 1 > /dev/null)

c_time_it = time -p (cat $(W_BIN) | $(1) 0.1 2048 1 > /dev/null 2>&1)
compare_v3_v4_v5: $(V3_EXE) $(V4_EXE) $(V5_EXE) \
	$(MW_EXE) $(SW_EXE)
	# produce world binary file
	$(MW_EXE) 256 0.1 1 > $(W_BIN)
	@echo "==========="
	$(call c_time_it,$(SW_EXE))
	@echo "==========="
	$(call c_time_it,$(V3_EXE))
	@echo "==========="
	$(call c_time_it,$(V4_EXE))
	@echo "==========="
	$(call c_time_it,$(V5_EXE))

