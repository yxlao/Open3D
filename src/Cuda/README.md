# Open3D: The CUDA Branch

This cuda branch is in **VERY EARLY STAGE** under development, and I cannot 
guarantee the
 stability or compatibility at all -- I do apologize for it. 
 
Since I work on my own, I expect to complete the basic functions in one or two
 months (**Christmas 2018** might be a good time to place a milestone). Before 
 that, I can only ensure it to be compilable on my machine.
 
Many parts of the code are migrated from my previous projects. They offer 
similar functions to Open3D CPU version, but the interfaces may vary. I will 
work on that part later. 

## Dependencies
- At current it requires OpenCV for basic image reading, writing, and 
displaying. I will remove that part and turn to Open3D's internal APIs.
- **Update 11.11.2018**, migration is partly done. Raytracing still requires 
it for displaying in several demo files, but can be removed.


## Designs for CUDA classes

### Architecture
Classes involving heterogeneous computations can involve both CPU computations 
(data preparation, kernel calls) and GPU computations (data manipulation on board).
Mixing them up will make the framework nasty, increase compilation time, and 
can cause linking problems.
 
I tried to design an architecture to split classes into smaller ones doing 
specific jobs. There may be better solutions. If better ones are suggested, I
 will think about changing my design. 

Typically, there will be 3 classes and 4 files holding a complete class. 
The classes include
- `Server` class, which holds GPU data and device functions. 
- `KernelCaller` class, which is a wrapper of kernel calls from CPU.
- `Host` class, which is the CPU interface that will prepare and transfer 
data, and launch kernels functions.
 
These classes will be distributed in 
- `.h` file, including all the declarations. In addition, global (kernel) 
functions will also be declared separately in the header.
- `.hpp` file, implementing the host functions for `Host` class. It will be 
compiled by g++.
- `.cuh` file, implementing the device functions for `Server` class. It will 
be compiled by nvcc.
- `.cuh` file, implementing the kernel functions for the global functions and
 its callers. It will be compiled by nvcc.

As a bridge, the `Server` classes will be directly passed to ```__global__``` 
functions **BY VALUE** (because CPU and GPU code does not share address space, 
passing by reference is incorrect). In view of this, we **CANNOT** use 
intelligent pointers as well as the constructors and destructors. We have to 
use raw pointers and manage memory in CPU code. 

The existance of `Server` is easy to understand -- it is a wrapper for the 
device side. The reason that I setup a KernelCaller class is twofold:
- Although kernel calls ```<<<blocks, threads >>>``` are launched from 
CPU, they can only be compiled by nvcc. It is more reasonable to separate 
these functions and put them together with corresponding kernel functions. 
- Throwing these kernel callers in `Host` class is possible, but this will 
make half of the `Host` methods compiled by nvcc and half by g++, which is 
weird. Of course we can compile everything by nvcc, but it will somehow 
increase compilation time, and suffer from warnings (e.g. from Eigen).
- A class wrapper for a bunch of functions can reduce the code to write for
instantiations. It can keep things organized.
 
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
    /* __host__ code for data transfer, allocation and free, etc */        
    void Upload(shared_ptr<Type> data);
    void Download(shared_ptr<Type> data);
    
    /* __host__ code calling ACudaKernelCallers functions */
    void Method();             
    
private:
    std::shared_ptr<ACudaServer> server_ = nullptr;    
};

class AcudaKernelCaller {
public: 
    static __host__ void MethodKernelCaller(AcudaServer& server);
}
/* Pass `Server` class BY VALUE */
__global__ void MethodKernel(ACudaServer server);
```  

Note if the class is templated, we have to instantiate them in a `.cu` file 
for `Server` and `KernelCaller` that includes the `.cuh` files, and a `.cpp` 
file that includes the `.hpp` file.

### CUDA Server Reference Counting
- To avoid frequent CUDA memory manipulation, we add reference count for 
servers. 
- Basically, every host class (say `ArrayCuda`) is connected to one 
server (say `ArrayCudaServer`) using a `shared_ptr`. Multiple host objects 
can share one server. 
- The copy constructor or assignment operators will just pass the 
`shared_ptr`, whose reference count will be monitored by STL.
- When all the host classes connected to the server are 
destroyed (i.e., when we detect `server_.use_count() == 1` meaning this object
 is 
the final host that points to the server), the data on GPU will be freed.

### Nested CUDA Servers
One CPU class should hold only one `Server` class, which can be nested with 
other `Server` classes. Nested `Server` MUST hold structs instead of their 
pointers, because otherwise the values cannot be correctly passed to CUDA. 
- For the classes with a simple `Server`, just handle cuda data correctly in 
`Create` and `Release`.
- For nested classes, write a function `UpdateServer` that correctly 
synchronize nested structs (not their ptrs!). One simplified sample goes here:
```cpp
class ACudaServer {
public:
    BCudaServer b_;
    Type* data_owned_by_a_;
}

class ACuda {
private:
    std::shared_ptr<BCudaServer> server_;
    BCuda b_;

public:
    void ACuda::UpdateServer() {
        server_->b_ = *b_.server();
    }
}
```

**DON'T** use inheritance for `Server`. It is not supported by CUDA -- there 
will be problems transforming *vtable* to kernels.

One thing to note is that how do we initialize the servers, if they are in 
containers (e.g.  HashTable, Array)? 
- One bad way to do this is to give every copy of them a host 
correspondent. That will make the code silly and impossible to maintain:
```cpp
ArrayCuda<BServer> array_;
std::vector<BCuda> barray_hosts_;
for (auto &barray_host : barray_hosts_) {
    barray_host.Create();    
}
/** On kernel, array_->server[i] = *barray_host.server() ??? **/
```   
- Another workaround is to pre-allocate the data in a plain cuda array 
(name them server memory pool will be a little bit easy to understand) using 
cudaMalloc on host side, and assign them on kernel. 
- ~~Another way I can think is to call malloc on kernels. This is intuitive, 
like you are calling a server to connect to some other servers on the server side.
 But there are problems of Cuda's heap size. Working on that.~~ That is too 
 slow and add extra dependencies (cuda driver). Forget it.

## Conventions of Object Memory Allocation 

For the classes such as ImageCuda, there are some conventions to follow:
- An object is uninitialized using the default constructor
- `Create` command ONLY allocate space;
- `Upload` command copy cpu objects (Image, cv::Mat) to gpu memory. If gpu 
memory is not allocated, call `Create`; if it is allocated with the same size, 
directly overwrite the memory with new data; if the size is incompatible, 
report it as an error. 
- `CopyFrom` command copy gpu objects (ImageCuda) to gpu memory. Size check 
is also needed. 
- Image processing such as `Downsample`, `Gaussian` follow the same 
convention as `Upload`.  

For the classes built upon ImageCuda, e.g. RGBDImageCuda, ImagePyramidCuda, 
we only allow to `Build` from ImageCuda. This restriction simplify the 
logistics.
 
## TODO

- ~~Add reference-count for the servers? Actually maybe we can enable the 
constructors and destructors for the client side... (works well for 
ImageCuda).~~ Done
- Add swap for the containers? (That will be really complicated for HashTable).
- Organize some error codes for functions to return.
- ~~See if we should replace servers instances with pointers?~~ Done
- Add macros for debugging code snippets (e.g. boundary check).

## Notes
- Use newer cmake. 3.5.1 has problems with cuda headers. 3.12.3 works for me.
- ~~Eigen has something wrong in their code. I believe not every functions 
are correctly prefixed with EIGEN_DEVICE_FUNC. Even if I don't call Eigen 
function in CUDA, as long as I compile with nvcc, the annoying warnings will 
come out. Work around: uncomment `EIGEN_DEVICE_FUNC` in Eigen's headers.~~ 
Fixed by separating cuda and cpp implementations.
- ~~DO NOT add boundary checks in frequently accessed functions, such as get(),
 push_back(), etc. Check them outside.~~ Add boundary check macros.
- Never abuse `inline` for non-trivial functions. NVCC will cry!

## Notations
- World coordinate (in unit of meter): Xw = (xw, yw, zw)$
- Volume coordinate (in unit of meter, transformed from world): Xv = (xv, yv,
 zv)
- Global voxel coordinate (in unit of voxel): X = (x, y, z)
- Local voxel coordinates in one specific subvolume: Xlocal = (xlocal, ylocal, zlocal)
- Voxel offset (global or local): dX = (dx, dy, dz)
- Subvolume coordinate (in unit of subvolume): Xsv = (xsv, ysv, zsv)
- Subvolume neighbor indices (in unit of subvolume): dXsv = (dxsv, dysv, dzsv)
