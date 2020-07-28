CC = g++
OBJ = main.o
LIB = -L/home/return/gsl/lib -lgsl -lgslcblas

CFLAGS  = -I/home/return/gsl/include

main:$(OBJ)
	$(CC) -o main $(OBJ)  $(LIB)
	rm -f $(OBJ)
	

$(OBJ): %.o: %.cpp
	$(CC) $(CFLAGS) -c $<
	export LD_LIBRARY_PATH=/home/return/gsl/lib:$LD_LIBRARY_PATH
clean:
	rm -f main $(OBJ)