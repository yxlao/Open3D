//
// Created by wei on 9/24/18.
//

#pragma once

namespace open3d {

namespace cuda {
template<typename T>
class ArrayCudaDevice;

template<typename T>
class ArrayCuda;

template<typename T>
class Array2DCudaDevice;

template<typename T>
class Array2DCuda;

template<typename T>
class MemoryHeapCudaDevice;

template<typename T>
class MemoryHeapCuda;

template<typename T>
class LinkedListNodeCuda;

template<typename T>
class LinkedListCudaDevice;

template<typename T>
class LinkedListCuda;

template<typename Key>
class HashEntry;

template<typename Key, typename Value, typename Hasher>
class HashTableCuda;

enum ContainerReturnCode {
    Success = 0,
    HashEntryIsEmpty = -1,
    HashEntryAlreadyExists = -2,
    HashEntryIsLocked = -3,
    LinkedListEntryNotFound = -4
};

/** TODO: add resize() for containers **/
} // cuda
} // open3d