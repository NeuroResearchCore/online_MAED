CXX=g++
CXXOPTIMIZE= -O2
CXXFLAGS= -Wall $(CXXOPTIMIZE) -larmadillo -std=c++17 
DISTNAME=NAME

FILES=utils.cpp Model.cpp main.cpp

all: result

example: clean
	g++ example.cpp utils.cpp -o example -O2 -larmadillo -std=c++17

run: clean result
	./result

*.o: *.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $@.cpp

result: $(FILES:.cpp=.o)
	$(CXX) -o $@ $(CXXFLAGS) $(FILES:.cpp=.o)

clean:
	rm -rf *.o *~ *.gch *.swp *.dSYM result example *.tar.gz

tarball: clean
	tar -cvf $(DISTNAME).tar.gz *
