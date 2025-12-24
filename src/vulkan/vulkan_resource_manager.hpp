#pragma once

#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

/**
 * Vulkan Resource Manager with Memory Pool
 * 
 * Manages the lifecycle of all Vulkan objects to ensure proper cleanup
 * and prevent resource leaks. Implements memory pooling to reduce
 * allocation overhead and prevent std::bad_alloc during initialization.
 */
class VulkanResourceManager {
public:
    VulkanResourceManager(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue computeQueue, uint32_t computeQueueFamilyIndex);
    ~VulkanResourceManager();

    // Buffer management
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void destroyBuffer(VkBuffer buffer, VkDeviceMemory memory);

    // Staging buffer helpers
    class StagingBuffer {
    public:
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        void* mappedData = nullptr;

        StagingBuffer(VkPhysicalDevice physDev, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
        ~StagingBuffer();

        void map();
        void unmap();
        void copyFrom(const void* data, VkDeviceSize size);
        void copyTo(void* data, VkDeviceSize size);

    private:
        VkPhysicalDevice physicalDevice;
        static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
    };

    std::unique_ptr<StagingBuffer> createStagingBuffer(VkDeviceSize size, VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    // Memory type finding
    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

    // Memory pool management
    struct PooledBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
        VkDeviceSize size;
        VkBufferUsageFlags usage;
        bool inUse;
    };
    
    VkBuffer getPooledBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory& memory);
    void addToPool(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer buffer, VkDeviceMemory memory);
    void releaseBuffer(VkBuffer buffer, VkDeviceMemory memory);
    void cleanupMemoryPool();
    void cleanupUnusedBuffers(); // Call between forward passes

    // Xe GPU specific optimizations
    bool isXeGPU();
    void createTiledBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                          std::vector<VkBuffer>& buffers, std::vector<VkDeviceMemory>& memories, size_t tileSize = 64 * 1024 * 1024);

private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    // Track all created objects for proper cleanup
    struct TrackedBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };
    std::vector<TrackedBuffer> trackedBuffers;

    // Memory pool for buffer reuse
    std::vector<PooledBuffer> bufferPool;
    std::mutex poolMutex;
    static constexpr VkDeviceSize MAX_POOL_SIZE = 1024 * 1024 * 1024; // 1GB pool limit

    // Helper functions
    void createCommandPool();
    void destroyCommandPool();
    bool findCompatibleBuffer(VkDeviceSize size, VkBufferUsageFlags usage, PooledBuffer& result);
};
#endif