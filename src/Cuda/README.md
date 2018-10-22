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


## Designs for CUDA classes

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
    std::shared_ptr<ACudaServer> server_ = nullptr;    
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
One CPU class should hold only one `server` class, which can be nested with 
other `server` classes. Nested `server` MUST hold structs instead of their 
pointers, because otherwise the values cannot be correctly passed to CUDA. 
- For the classes with a simple `server`, just handle cuda data correctly in 
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

**DON'T** use inheritance for `server`. It is not supported by CUDA -- there 
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
- Another workaround is to pre-allocate the data in a plain cuda array using 
cudaMalloc on host side, and assign them on kernel. (Name them server memory 
pool will be a little bit easy to understand?)
- ~~The best way for me is to call malloc on kernels. This is intuitive, like 
you are calling a server to connect to some other servers on the server side.
 But there are problems of Cuda's heap size. Working on that.~~ That is too 
 slow and add extra dependencies (cuda driver). Forget it.

## Conventions of Creating Objects 

For the classes such as ImageCuda, there are some conventions to follow:
- Create some thing when `Create` is directly called
- When using functions such as `Upload`, `Downsample`, check if the object 
exists, i.e., if it is with a `server_`. If not, create for it for sure. If 
not, then we **CHOOSE TO CHECK SIZE**. If size meets then things are fine. If
 not, we **ABORT** and print an error. Note this is just a *strategy* or 
 *convention*. We can also choose to release it and create a new server.
- Yet functions such as `Resize` should force to release and re-create objects.
 

## TODO

- ~~Add reference-count for the servers? Actually maybe we can enable the 
constructors and destructors for the client side... (works well for 
ImageCuda).~~ Done
- Add swap for the containers? (That will be really complicated for HashTable).
- Organize some error codes for functions to return.
- ~~See if we should replace servers instances with pointers?~~ Done
- Maybe re-introduce built-in types? Some additional utility functions such as 
make_float(template<T>) may work?
- Add macros for debugging code snippets (e.g. boundary check).

## Notes
- Eigen has something wrong in their code. I believe not every functions 
are correctly prefixed with EIGEN_DEVICE_FUNC. Even if I don't call Eigen 
function in CUDA, as long as I compile with nvcc, the annoying warnings will 
come out. Work around: uncomment `EIGEN_DEVICE_FUNC` in Eigen's headers.
- DO NOT add boundary checks in frequently accessed functions, such as get(),
 push_back(), etc. Check them outside (or just ignore them first...)
- Never abuse `inline` for non-trivial functions. NVCC will cry!
2
## Notations
- World coordinate (in unit of meter): Xw = (xw, yw, zw)$
- Volume coordinate (in unit of meter, transformed from world): Xv = (xv, yv,
 zv)

- Global voxel coordinate (in unit of voxel): X = (x, y, z)
- Local voxel coordinates in one specific subvolume: Xlocal = (xlocal, ylocal, zlocal)
- Voxel offset (global or local): dX = (dx, dy, dz)

- Subvolume coordinate (in unit of subvolume): Xsv = (xsv, ysv, zsv)
- Subvolume neighbor indices (in unit of subvolume): dXsv = (dxsv, dysv, dzsv)
 