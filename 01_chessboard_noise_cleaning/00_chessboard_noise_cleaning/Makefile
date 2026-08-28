CXX = nvcc
#COMP_SANI_FLAG = -lineinfo
CUDA_GDB_FLAGS = -g -G

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(COMP_SANI_FLAG) $(CUDA_GDB_FLAGS) $(OPENCV_CFLAGS) $< -o $@ $(OPENCV_LIBS)

.PHONY: clean
clean:
	rm -f $(NAME)

.PHONY: run
run: $(NAME)
	./$<
