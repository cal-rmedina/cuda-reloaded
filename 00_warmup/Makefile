CXX = nvcc
#COMP_SANI_FLAG = -lineinfo
#CUDA_LIBRARY = -lcuda
#NVFLAGS = -Wno-deprecated-declarations 

NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(COMP_SANI_FLAG) $(CUDA_LIBRARY) $(NVFLAGS) $< -o $@ 

.PHONY: clean
clean:
	rm -f $(NAME)

.PHONY: run
run: $(NAME)
	./$<
