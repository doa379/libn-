LIBS_PATH = /lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
INCS =
LIBS =

SRC_NN = nn.cpp
OBJ_NN = ${SRC_NN:.cpp=.o}

CC = g++
CFLAGS = -std=c++11 -c -g -Wall -Werror -pie -fPIC ${INCS}
LDFLAGS = ${LIBS}

all: libnn++.so

.cpp.o:
		@echo CC $<
		@${CC} ${CFLAGS} $<

libnn++.so: ${OBJ_NN}
		@echo CC -o $@
		@${CC} -shared -o $@ ${OBJ_NN} ${LDFLAGS}

clean:
		@echo cleaning
		@rm -f ${OBJ_NN}
		@rm -f test_nn
