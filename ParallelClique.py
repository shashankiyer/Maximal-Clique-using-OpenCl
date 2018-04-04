'''
Computes the maximal clique of a graph using PyOpenCL.

:authors Shashank, Kunal, Nick
'''
from __future__ import absolute_import
from __future__ import print_function
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import os, random
import argparse

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = ':'
   
def parse_adjacency_matrix( filename):
    if not os.path.exists(filename):
        raise Exception('Graph file not found')

    V=[]

    with open(filename, 'r') as f:
        for line in f:
            v = []
            for j in line.split():
                v.append(int(j))
            V.append(v)
    
    for i in range (len(V)):
        for j in range (len(V[0])):
            if i != j:
                V[i][j] = int(not V[i][j])

    return V

def generate_coin_flips(size):
    rand = np.zeros(size)
    for i in range (size):
        rand[i] = random.random()

    return rand

def degree(V):
    deg = np.zeros(np.shape(V)[0])
    for i in range (len(deg)):
        for j in V[i]:
            deg[i] = deg[i] + j
    return deg

def greedy_clique(V):
    v = np.array(V).astype(np.int32)
    v = v.ravel()
    dv = degree(V)
    random = generate_coin_flips(np.shape(V)[0])
    
    shape = []
    shape.append( np.shape(V)[0])
    shape.append( np.shape(V)[1])
    shape = np.array(shape).astype(np.int32)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    v_dev = cl_array.to_device(queue, v)
    dv_dev = cl_array.to_device(queue, dv)
    rand_dev = cl_array.to_device(queue, random)
    shape_dev = cl_array.to_device(queue, shape)

    result = cl_array.empty_like(dv_dev)

    prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        void mark_vertices(__global bool* mark, __global const float* rand, __global const int* dv)
        {
            int gid = get_global_id(0);            
            
            if(rand[gid] >= 1 - 1/(2*dv[gid]))
                mark[gid] = true;            

        }

        void unmark_lower_degree(__global const int* V, __global const int* shape, __global bool* mark, __global const int* dv)
        {
            
            int gid = get_global_id(0);
            int size = get_global_size(0);
            if(mark[gid])
            {
                int degU = dv[gid];
                
                for(int j =0; j< shape[1] && gid*shape[1] + j <= size; j++)
                {
                    if(V[gid*shape[1] + j] && mark[j])
                    {
                        int degV = dv[j];
                        if(degV>=degU)
                            mark[j] = false;
                        else
                        {
                            mark[gid] = false;
                            break;
                        }
                    }
                }
            }
        }

        void select_vertices(__global int* res, __global bool* mark)
        {
            int gid = get_global_id(0);
            int size = get_global_size(0);
            for(int i=0 ; i<size; i++)
                res[gid] = mark[gid]? 1:0;
        }


        __kernel void maximal_clique(__global const int* V, __global const int* shape,
        __global const float* rand, __global const int* dv, __global int* res, __global bool* mark)
        {
            int gid = get_global_id(0);

            mark_vertices(mark, rand, dv);
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            unmark_lower_degree(V, shape, mark, dv);
            barrier(CLK_GLOBAL_MEM_FENCE);

            select_vertices(res, mark);
        }
        """).build()

    mark = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, result.nbytes/8)
    prg.maximal_clique(queue, dv_dev.shape, None, v_dev.data, shape_dev.data, rand_dev.data, dv_dev.data, result.data, mark)
    
    for i in result:
        if i != 0:
            print(i)
           
                
parser = argparse.ArgumentParser(
    description='Computes the clique of a matrix using PyOpenCL.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions see the README'
)

parser.add_argument(
    'graph_file',
    help='Path to the file containing the adjacency matrix'
)

if __name__ == '__main__':
    args = parser.parse_args()

    V = parse_adjacency_matrix(args.graph_file)
    greedy_clique(V)        
