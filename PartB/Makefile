sim: data/input_128.in cmm_sim

server: data/input_128.in cmm_server

cmm_server: main.cu header/gpu_thread.h
	nvcc main.cu -o cmm_server -I ./header

cmm_sim: main.cu header/gpu_thread.h
	nvcc main.cu -o cmm_sim -I ./header --cudart shared -gencode arch=compute_70,code=compute_70

data/generate: data/generate.cpp
	g++ ./data/generate.cpp -o ./data/generate

data/input_128.in: data/generate
	./data/generate 128

run_sim: data/input_128.in cmm_sim
	./cmm_sim data/input_128.in
	
run_server: data/input_128.in cmm_server
	./cmm_server data/input_128.in

clean:
	rm -f cmm_sim cmm_server
