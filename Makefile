FLAGS= -std=c++17 -march=native -ffast-math -flto -Wextra -Wall

.PHONY: all clean

all: main

main: main.cpp matrix.cpp network.cpp vec.cpp activation.cpp
	g++ $(FLAGS) $^ -o $@

clean:
	rm -f main *.txt
