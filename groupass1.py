

from mpi4py import MPI
import numpy as np

n=500

assert n == n

#print ("Initialising variables.\n")
a = np.zeros(shape=(n, n))
b = np.zeros(shape=(n, n))
c = np.zeros(shape=(n, n))

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()

#print ("Process %d started.\n" % (rank))
#print ("Running from processor %s, rank %d out of %d processors.\n" % (processorName, rank, worldSize))

if rank == 0:
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.zeros(shape=(n, n))

#Calculate the slice per worker
if (worldSize == 1):
    slice = n
else:
    slice = n // (worldSize-1) #make sure it is divisible

assert slice >= 1


comm.Barrier()
    
if rank == 0:
    #print ("Initialising Matrix A and B (%d,%d).\n" % (n, n))
    print ("Start")
        
    for i in range(1, worldSize):
        offset = (i-1)*slice #0, 10, 20
        row = a[offset,:]
        comm.send(offset, dest=i, tag=i)
        comm.send(row, dest=i, tag=i)
        for j in range(0, slice):
            comm.send(a[j+offset,:], dest=i, tag=j+offset)
    #print ("All sent to workers.\n")

comm.Barrier()

if rank != 0:

    #print ("Data Received from process %d.\n" % (rank))
    offset = comm.recv(source=0, tag=rank)
    recv_data = comm.recv(source=0, tag=rank)
    for j in range(1, slice):
        c = comm.recv(source=0, tag=j+offset)
        recv_data = np.vstack((recv_data, c))

    #print ("Start Calculation from process %d.\n" % (rank))

    #Loop through rows
    t_start = MPI.Wtime()
    for i in range(0, slice):
        res = np.zeros(shape=(n))
        if (slice == 1):
            r = recv_data
        else:
            r = recv_data[i,:]
        ai = 0
        for j in range(0, n):
            q = b[:,j] #get the column we want
            for x in range(0, n):
                res[j] = res[j] + (r[x]*q[x])
            ai = ai + 1
        if (i > 0):
           send = np.vstack((send, res))
        else:
            send = res
    t_diff = MPI.Wtime() - t_start
    
    print("Process %d finished in %5.4fs.\n" %(rank, t_diff))
    #Send large data
    #print ("Sending results to Master %d bytes.\n" % (send.nbytes))
    comm.Send([send, MPI.FLOAT], dest=0, tag=rank) #1, 12, 23

comm.Barrier()

if rank == 0:  
    #print ("Checking response from Workers.\n")
    res1 = np.zeros(shape=(slice, n))
    comm.Recv([res1, MPI.FLOAT], source=1, tag=1)
    #print ("Received response from 1.\n")
    kl = np.vstack((res1))
    for i in range(2, worldSize):
        resx= np.zeros(shape=(slice, n))
        comm.Recv([resx, MPI.FLOAT], source=i, tag=i)
        #print ("Received response from %d.\n" % (i))
        kl = np.vstack((kl, resx))
    print ("End")
    #print ("Result AxB.\n")
    #print (kl)   

comm.Barrier()





