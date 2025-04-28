# ---- Edit the path to the boost library in the lines below --- #
BOOSTlib=/cygdrive/c/BoostTest/lib
BOOSTinclude=/cygdrive/c/BoostTest/include
# --------------------------------------------------------------#


CXX=g++
FLAGS=-Wall -O3 -march=native -g -Wno-deprecated -I. -ID:\C_env\boost_1_84_0
LIBS=-LD:\C_env\boost_1_84_0\stage\lib -lboost_program_options-mgw8-mt-x64-1_84
BIN=GND reinsertion

#FLAGS+=-Wno-unknown-pragmas -I $(BOOSTinclude)

all: $(BIN)

GND: GND.cpp 
	${CXX} ${FLAGS} GND.cpp ${LIBS} -O3 -o $@

reinsertion: reinsertion.cpp 
	${CXX} ${FLAGS} reinsertion.cpp ${LIBS} -o $@
clean:
	rm -f $(BIN) *.o


