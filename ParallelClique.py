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
    dv = degree(V)
    random = generate_coin_flips(np.shape(V)[0])

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    v_dev = cl_array.to_device(queue, v)
    dv_dev = cl_array.to_device(queue, dv)
    rand_dev = cl_array.to_device(queue, random)

    result = cl_array.empty_like(dv_dev)

    prg = cl.Program(ctx, """
        
        void mark_vertices(int** V, bool* mark, int* rand, int* dv)
        {
            int gid = get_global_id(0);            
            
            if(rand[gid] >= 1 - 1/(2*dv[gid]))
                mark[gid] = true;            

        }

        void unmark_lower_degree(int** V, bool* mark, int* dv)
        {
            
            int gid = get_global_id(0);
            if(mark[gid])
            {
                int degU = dv[gid];
                unsigned int neighbours = sizeof(V[0])/sizeof(V[0][0]);
                for(unsigned int j =0; j< neighbours; j++)
                {
                    if(V[gid][j] && mark[j])
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

        void select_vertices(int* res, bool* mark)
        {
            int gid = get_global_id(0);
            unsigned int size = sizeof(mark)/sizeof(mark[0]);
            for(unsigned int i=0 ; i<size; i++)
                res[gid] = mark[gid]? 1:0;
        }


        __kernel void maximal_clique(__global const int** V,
        __global const int* rand, __global const int* dv, __global int* res, __global bool* mark)
        {
            int gid = get_global_id(0);

            mark_vertices(V, mark, rand, dv);
            barrier(CLK_GLOBAL_MEM_FENCE);

            //int group_size = get_global_size(0);
            
            unmark_lower_degree(V, mark, dv);
            barrier(CLK_GLOBAL_MEM_FENCE);

            select_vertices(res, mark);
        }
        """).build()

    prg.maximal_clique(queue, V.shape, dv.shape, v_dev.data, dv_dev.data, rand_dev.data, result.data)#, cl.CreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err))

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
