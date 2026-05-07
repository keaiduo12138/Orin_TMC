# Simple Makefile to build read_timestamp

CXX := g++
CXXFLAGS := -std=c++11 -O2 \
	-I../usb/inc \
	-I..

LDFLAGS := -lusb-1.0 -lpthread

SRC := read_timestamp.cpp \
	../usb/cyusb/src/libcyusb.cpp

BIN := read_timestamp
BIN2 := read_timestamp_2

all: $(BIN) $(BIN2)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

$(BIN2): read_timestamp_2.cpp ../usb/cyusb/src/libcyusb.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN2) read_timestamp_2.cpp ../usb/cyusb/src/libcyusb.cpp $(LDFLAGS)

clean:
	rm -f $(BIN) $(BIN2)

.PHONY: all clean


