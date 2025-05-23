# taskflow path: /heorot/lchang21/taskflow/taskflow

DEBUG ?= 0

CXX ?= g++

CXXFLAGS = -std=c++17 -Wall -pthread
LDFLAGS  = -std=c++17 -Wall -pthread

ifeq ($(strip $(DEBUG)),1)
CXXFLAGS += -g -O0
LDFLAGS  += -g -O0
else
CXXFLAGS += -O3 -march=native
LDFLAGS  += -O3 -march=native
endif

# Include directories
INC        = -I../core -I/heorot/lchang21/taskflow/taskflow
INC_EXT    =  

# Location of the libraries.
LIB        = -L../core -lcore_s
LIB_EXT    = 

INC := $(INC) $(INC_EXT)
LIB := $(LIB) $(LIB_EXT)

CXXFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main
all: $(TARGET)

.PRECIOUS: %.cpp %.o

main.o: main.cpp ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main: main.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@ 

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean 