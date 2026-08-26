CXX = nvcc
COMP_SANI_FLAG = -lineinfo

NVFLAGS = -Wno-deprecated-declarations -gencode arch=compute_75,code=compute_75 -lcuda

NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(REDUCTION_FLAG) $(THREAD_FLAG) $(COMP_SANI_FLAG) $(NVFLAGS) $(OPENCV_CFLAGS) $< -o $@ $(OPENCV_LIBS)

.PHONY: clean
clean:
	rm -f $(NAME)

.PHONY: run
run: $(NAME)
	./$<
