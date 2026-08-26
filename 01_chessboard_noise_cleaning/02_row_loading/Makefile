# Export CUDA path
# export PATH := /usr/local/cuda/bin:$(PATH)

CXX = nvcc

NVFLAGS =-arch=sm_75 -lineinfo -Xptxas -v
NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(NVFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -f $(NAME)

.PHONY: run

run: $(NAME)
	./$<
