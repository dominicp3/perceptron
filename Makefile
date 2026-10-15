FLAGS= -std=c++17 -O2 -Wextra -Wall

.PHONY: all clean

all: main

main: main.cpp matrix.cpp network.cpp vec.cpp
	g++ $(FLAGS) $^ -o $@

clean:
	rm -f main *.txt
