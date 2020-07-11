#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include<assert.h>
#include <map>
#include<vector_functions.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include<thrust/iterator/zip_iterator.h>
typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;
class cached_allocator
{
public:
    typedef char value_type;
 
    cached_allocator() { };
 
    ~cached_allocator()
    {
        free_all();
    };
 
    char* allocate(std::ptrdiff_t num_bytes)
    {
        char* result = 0;
 
        free_blocks_type::iterator free_block = free_blocks.find(num_bytes);
 
        if (free_block != free_blocks.end())
        {
            result = free_block->second;
            free_blocks.erase(free_block);
        }
        else
        {
            try
            {
                result = thrust::cuda::malloc<char>(num_bytes).get();
            }
            catch(std::runtime_error &e)
            {
                throw;
            }
        }
        allocated_blocks.insert(std::make_pair(result, num_bytes));
 
        return result;
    };
 
	  void deallocate(char* ptr, size_t n)
    {
        allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
        std::ptrdiff_t num_bytes = iter->second;
        allocated_blocks.erase(iter);
 
        free_blocks.insert(std::make_pair(num_bytes, ptr));
    };
 
private:
 
    free_blocks_type free_blocks;
    allocated_blocks_type allocated_blocks;
 
    void free_all()
    {
        for (free_blocks_type::iterator i = free_blocks.begin();
                i != free_blocks.end(); i++)
        {
            thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
        }
 
        for (allocated_blocks_type::iterator i = allocated_blocks.begin();
                i != allocated_blocks.end(); i++)
        {
            thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
        }
    };
};
#endif
