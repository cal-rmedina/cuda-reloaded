CXX = nvcc
COMP_SANI_FLAG = -lineinfo

NVFLAGS = -Wno-deprecated-declarations -gencode arch=compute_89,code=compute_89 -lcuda
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(REDUCTION_FLAG) $(THREAD_FLAG) $(COMP_SANI_FLAG) $(NVFLAGS) $(OPENCV_CFLAGS) $< -o $@ $(OPENCV_LIBS)

.PHONY: clean
clean:
	rm -f $(NAME)

.PHONY: run
run: $(NAME)
	./$<
