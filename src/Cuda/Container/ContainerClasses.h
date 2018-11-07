//
// Created by wei on 9/24/18.
//

#pragma once

namespace open3d {

template<typename T>
class ArrayCudaServer;

template<typename T>
class ArrayCuda;

template<typename T>
class MemoryHeapCudaServer;

template<typename T>
class MemoryHeapCuda;

template<typename T>
class LinkedListNodeCuda;

template<typename T>
class LinkedListCudaServer;

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
}