#include "vulkan_resource_manager.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

VulkanResourceManager::VulkanResourceManager(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue computeQueue, uint32_t computeQueueFamilyIndex)
    : device(device), physicalDevice(physicalDevice), computeQueue(computeQueue), computeQueueFamilyIndex(computeQueueFamilyIndex), commandPool(VK_NULL_HANDLE) {
    createCommandPool();
}

VulkanResourceManager::~VulkanResourceManager() {
    // Cleanup memory pool first
    cleanupMemoryPool();

    // Destroy all tracked buffers in reverse order
    for (auto it = trackedBuffers.rbegin(); it != trackedBuffers.rend(); ++it) {
        if (it->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, it->buffer, nullptr);
        }
        if (it->memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, it->memory, nullptr);
        }
    }
    trackedBuffers.clear();

    destroyCommandPool();
}

void VulkanResourceManager::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    // Try to get from pool first
    buffer = getPooledBuffer(size, usage, bufferMemory);
    if (buffer != VK_NULL_HANDLE) {
        return;
    }

    // Check against TRUE GPU allocation limits (not heap size)
    // Xe GPUs limit individual allocations to ~4GB despite larger heap reports
    VkPhysicalDeviceMaintenance3Properties maint3Props = {};
    maint3Props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;

    VkPhysicalDeviceProperties2 deviceProps2 = {};
    deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProps2.pNext = &maint3Props;

    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProps2);

    VkDeviceSize trueMaxAllocation = maint3Props.maxMemoryAllocationSize;
    VkDeviceSize safeAllocationLimit = trueMaxAllocation > 1024 * 1024 * 1024 ?
                                       trueMaxAllocation * 7 / 10 : // 70% of limit for safety
                                       512 * 1024 * 1024; // 512MB minimum safe limit

    std::cout << "Buffer allocation check: requested " << size / (1024 * 1024) << "MB, GPU limit " << trueMaxAllocation / (1024 * 1024) << "MB, safe limit " << safeAllocationLimit / (1024 * 1024) << "MB" << std::endl;

    if (size > safeAllocationLimit) {
        // For very large tensors, attempt chunked allocation
        std::cout << "Large buffer requested (" << size / (1024 * 1024) << "MB), exceeds safe limit (" << safeAllocationLimit / (1024 * 1024) << "MB)" << std::endl;

        // TODO: Implement actual chunked buffer support
        // For now, provide clear error message with guidance
        throw std::runtime_error("Buffer allocation of " + std::to_string(size / (1024 * 1024)) + "MB exceeds GPU safe allocation limit of " + std::to_string(safeAllocationLimit / (1024 * 1024)) + "MB. This is likely due to Xe GPU per-allocation caps (~4GB). Try: 1) --cpu flag for CPU training, 2) Reduce model size (hidden/layers), or 3) Update Mesa drivers.");
    }

    // Create new buffer if not available in pool
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    if (vkBindBufferMemory(device, buffer, bufferMemory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        throw std::runtime_error("failed to bind buffer memory!");
    }

    // Add to buffer pool for reuse
    addToPool(size, usage, buffer, bufferMemory);

    // Track the buffer for cleanup
    trackedBuffers.push_back({buffer, bufferMemory});
}

void VulkanResourceManager::destroyBuffer(VkBuffer buffer, VkDeviceMemory memory) {
    // Remove from tracked buffers
    trackedBuffers.erase(
        std::remove_if(trackedBuffers.begin(), trackedBuffers.end(),
            [buffer, memory](const TrackedBuffer& tb) {
                return tb.buffer == buffer && tb.memory == memory;
            }),
        trackedBuffers.end()
    );

    // Destroy immediately
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
}

std::unique_ptr<VulkanResourceManager::StagingBuffer> VulkanResourceManager::createStagingBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    return std::unique_ptr<StagingBuffer>(new StagingBuffer(physicalDevice, device, size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
}

void VulkanResourceManager::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to submit resource manager copy command buffer!");
    }
    if (vkQueueWaitIdle(computeQueue) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("failed to wait for resource manager queue idle!");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

uint32_t VulkanResourceManager::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanResourceManager::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void VulkanResourceManager::destroyCommandPool() {
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
}

VkBuffer VulkanResourceManager::getPooledBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory& memory) {
    std::lock_guard<std::mutex> lock(poolMutex);
    
    PooledBuffer candidate;
    if (findCompatibleBuffer(size, usage, candidate)) {
        candidate.inUse = true;
        memory = candidate.memory;
        return candidate.buffer;
    }
    
    return VK_NULL_HANDLE; // No suitable buffer found
}

void VulkanResourceManager::addToPool(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer buffer, VkDeviceMemory memory) {
    std::lock_guard<std::mutex> lock(poolMutex);

    VkDeviceSize totalPoolSize = 0;
    for (const auto& pooled : bufferPool) {
        totalPoolSize += pooled.size;
    }

    if (totalPoolSize + size <= MAX_POOL_SIZE) {
        bufferPool.push_back({buffer, memory, size, usage, false});
    }
}

void VulkanResourceManager::releaseBuffer(VkBuffer buffer, VkDeviceMemory memory) {
    std::lock_guard<std::mutex> lock(poolMutex);

    for (auto& pooled : bufferPool) {
        if (pooled.buffer == buffer && pooled.memory == memory) {
            pooled.inUse = false;
            return;
        }
    }
}

bool VulkanResourceManager::findCompatibleBuffer(VkDeviceSize size, VkBufferUsageFlags usage, PooledBuffer& result) {
    VkDeviceSize totalPoolSize = 0;
    
    for (const auto& pooled : bufferPool) {
        totalPoolSize += pooled.size;
        if (!pooled.inUse && pooled.size >= size && (pooled.usage & usage) == usage) {
            result = pooled;
            return true;
        }
    }
    
    // Check if adding new buffer would exceed pool limit
    if (totalPoolSize + size > MAX_POOL_SIZE) {
        return false;
    }
    
    return false;
}

void VulkanResourceManager::cleanupMemoryPool() {
    std::lock_guard<std::mutex> lock(poolMutex);
    
    for (auto& pooled : bufferPool) {
        if (pooled.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, pooled.buffer, nullptr);
        }
        if (pooled.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, pooled.memory, nullptr);
        }
    }
    bufferPool.clear();
}

void VulkanResourceManager::cleanupUnusedBuffers() {
    std::lock_guard<std::mutex> lock(poolMutex);

    auto it = bufferPool.begin();
    while (it != bufferPool.end()) {
        if (!it->inUse) {
            if (it->buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, it->buffer, nullptr);
            }
            if (it->memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, it->memory, nullptr);
            }
            it = bufferPool.erase(it);
        } else {
            ++it;
        }
    }
}

bool VulkanResourceManager::isXeGPU() {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    std::string deviceName = deviceProperties.deviceName;
    return deviceName.find("Iris") != std::string::npos ||
           deviceName.find("Xe") != std::string::npos;
}

void VulkanResourceManager::createTiledBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                             std::vector<VkBuffer>& buffers, std::vector<VkDeviceMemory>& memories, size_t tileSize) {
    VkDeviceSize remainingSize = size;
    VkDeviceSize offset = 0;

    while (remainingSize > 0) {
        VkDeviceSize currentTileSize = std::min(remainingSize, static_cast<VkDeviceSize>(tileSize));

        VkBuffer buffer;
        VkDeviceMemory memory;
        createBuffer(currentTileSize, usage, properties, buffer, memory);

        buffers.push_back(buffer);
        memories.push_back(memory);

        remainingSize -= currentTileSize;
        offset += currentTileSize;
    }
}

// StagingBuffer implementation
VulkanResourceManager::StagingBuffer::StagingBuffer(VkPhysicalDevice physDev, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    : physicalDevice(physDev), device(dev), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE), mappedData(nullptr) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create staging buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        throw std::runtime_error("failed to allocate staging buffer memory!");
    }

    vkBindBufferMemory(device, buffer, memory, 0);
}

VulkanResourceManager::StagingBuffer::~StagingBuffer() {
    unmap();
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
}

void VulkanResourceManager::StagingBuffer::map() {
    if (mappedData == nullptr) {
        vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mappedData);
    }
}

void VulkanResourceManager::StagingBuffer::unmap() {
    if (mappedData != nullptr) {
        vkUnmapMemory(device, memory);
        mappedData = nullptr;
    }
}

void VulkanResourceManager::StagingBuffer::copyFrom(const void* data, VkDeviceSize size) {
    map();
    memcpy(mappedData, data, size);
    unmap();
}

void VulkanResourceManager::StagingBuffer::copyTo(void* data, VkDeviceSize size) {
    map();
    memcpy(data, mappedData, size);
    unmap();
}

uint32_t VulkanResourceManager::StagingBuffer::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}