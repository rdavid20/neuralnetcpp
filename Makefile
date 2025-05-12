# Compiler and flags
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -O2 -MMD -MP

# Directories
SRC_DIR := .
BUILD_DIR := build
BIN := $(BUILD_DIR)/neuralnet

# Find all .cpp files and generate corresponding .o and .d file names
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# Default target
all: $(BIN)

# Link object files into final binary
$(BIN): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile each .cpp to .o (with header dependency tracking)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean all build files
clean:
	rm -rf $(BUILD_DIR)

# Run the compiled binary
run: $(BIN)
	./$(BIN)

# Include generated .d dependency files
-include $(DEPS)

.PHONY: all clean run
