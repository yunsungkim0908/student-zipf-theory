ROOTDIR = /piech/u/yunsung/.local

CC = g++
CFLAGS = -Wall -I $(ROOTDIR)/include -fopenmp
LDFLAGS = -Wl,--rpath=$(ROOTDIR)/lib -fopenmp -O3
LIBS = -L$(ROOTDIR)/lib -lm -lgsl -lgslcblas -lboost_program_options

TARGETS = generate sample vam_sample vam_generate

.PHONY: clean all
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

all: generate sample vam_generate
vam: vam_generate vam_sample
generate: grammar.o stateful.o simple.o main_generate.o
sample: grammar.o stateful.o main_sample.o
vam_generate: grammar.o varied_ability.o main_generate_vam.o
vam_sample: grammar.o varied_ability.o main_sample_vam.o

$(TARGETS):
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.o $(TARGETS)
