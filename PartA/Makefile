all: data/input_16.in data/input_4096.in data/input_8192.in data/input_16384.in rmm

rmm: main.cpp header/single_thread.h header/multi_thread.h
	g++ -g -ggdb main.cpp -o rmm -I ./header -lpthread -mavx -mavx2

data/generate: data/generate.cpp
	g++ ./data/generate.cpp -o ./data/generate

data/input_16.in: data/generate
	./data/generate 16 

data/input_4096.in: data/generate
	./data/generate 4096 

data/input_8192.in: data/generate
	./data/generate 8192 

data/input_16384.in: data/generate
	./data/generate 16384 

run: data/input_16.in data/input_4096.in data/input_8192.in data/input_16384.in rmm
	./rmm data/input_16.in
	./rmm data/input_4096.in
	./rmm data/input_8192.in
	./rmm data/input_16384.in

clean:
	rm rmm
