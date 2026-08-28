CXX = nvcc
#COMP_SANI_FLAG = -lineinfo
CUDA_GDB_FLAGS = -g -G

NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(CUDA_GDB_FLAGS) $(COMP_SANI_FLAG) $< -o $@

.PHONY: clean
clean:
	rm -f $(NAME)

.PHONY: run
run: $(NAME)
	./$<
