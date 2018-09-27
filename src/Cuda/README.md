# Logistics for CUDA programming

### Keep CUDA related functions in libraries.
 
All code related to CUDA APIs should be contained in `.cu | .cuh` and 
NOT exposed to `.h` files that will be included by C++ files.

To do this, we treat CUDA part as a `server` and write specific classes to 
manage GPU memory. 
 - On the CPU side, we start requests to allocate GPU memory,
 upload data, complete tasks, download data, and free GPU memory. 
 - On the GPU side, the `server` class will handle the requests correspondingly
  and play with GPU data. 

As a bridge, the `server` classes will be directly passed to ```__global__``` 
functions **BY VALUE** (because CPU and GPU code does not share address space, 
passing by reference is incorrect). In view of this, we **CANNOT** use 
intelligent pointers as well as the constructors and destructors. We have to 
use raw pointers and manage memory in CPU code. 
 
A typical header will look like this:
```cpp
class ACudaServer {
public:
    /* __device__ code to manipulate gpu data */
private:
    /* Everything (except small const varaibles)
     * should be stored in the form of raw pointer 
     */
    Type* data_;
    
    friend class ACuda;
};

class ACuda {
public:
    /* __host__ code for data transfer */        
    void Upload(shared_ptr<Type> data);
    void Download(shared_ptr<Type> data);
    
    /* __host__ code calling __global__ functions */
    void Method();             
    
private:
    ACudaServer server_;    
};

/* Pass `server` class BY VALUE */
__global__ void MethodKernel(ACudaServer server);
```  

To implement these declarations, we put things in `.cuh` and `.cu` instead of `
.cpp`, and 
let nvcc compile them. This can avoid loading CUDA headers when we do not 
want to play with CUDA directly. The general logic is that 
- Put `__device__` and `__host__` functions in `.cuh`.
- Put `__global__` functions in `.cu`, which will include `.cuh`.

Note if the class is templated, we have to instantiate them in the 
corresponding `.cuh` file, or in a `.cu` file that includes the `.cuh`.

One CPU class should hold only one `server` class, which can be nested with 
other `server` classes.