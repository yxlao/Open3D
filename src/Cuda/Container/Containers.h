//
// Created by wei on 9/24/18.
//

#ifndef OPEN3D_CONTAINERS_H
#define OPEN3D_CONTAINERS_H

#pragma once

namespace three {

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

}

#endif //OPEN3D_CONTAINERS_H
