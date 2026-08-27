CXX = nvcc
CUDA_GDB_FLAGS = -g -G

NAME = main

$(NAME): $(NAME).cu
	$(CXX) $(CUDA_GDB_FLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(NAME)

.PHONY: run
run: $(NAME)
	./$<
