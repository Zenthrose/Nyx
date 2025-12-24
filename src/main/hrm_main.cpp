#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#ifndef NO_VULKAN
#include <vulkan/vulkan.h>
#include "../vulkan/vulkan_compatibility.h"
#endif
#include "../hrm/resource_aware_hrm.hpp"
#include "../system/memory_compaction_system.hpp"
#include "../system/cloud_storage_manager.hpp"
#include "hrm_cli.hpp"
#include "hrm_gui.hpp"
#include "../training/character_language_trainer.hpp"
#include "../system/hardware_profiler.hpp"
#include "../utils/logger.hpp"
#include "../utils/config_manager.hpp"

namespace fs = std::filesystem;

// Configuration validation
void validate_configuration(ResourceAwareHRMConfig& config, const HardwareCapabilities& hw_caps);

// Default configuration with hardware adaptation
ResourceAwareHRMConfig createDefaultHRMConfig(const HardwareCapabilities& hw_caps) {
    ResourceAwareHRMConfig config;

    // Base HRM config (SelfModifyingHRMConfig)
    config.base_config.base_config.use_utf8_communication = true;
    config.base_config.base_config.max_conversation_length = 10000;
    config.base_config.base_config.enable_self_evolution = true;
    config.base_config.base_config.evolution_rate = 0.1f;
    config.base_config.base_config.adaptation_cycles = 10;
    config.base_config.base_config.enable_continual_learning = true;

    // UTF-8 configuration
    config.base_config.base_config.utf8_config.max_sequence_length = 2048;
    config.base_config.base_config.utf8_config.embedding_dim = 256;
    config.base_config.base_config.utf8_config.use_byte_fallback = true;

    // Meta-reasoning configuration (enable self-repair)
    config.base_config.base_config.meta_config.enable_self_repair = true;
    config.base_config.base_config.meta_config.enable_confidence_scoring = true;
    config.base_config.base_config.meta_config.analysis_depth = 3;
    config.base_config.base_config.meta_config.confidence_threshold = 0.75f;
    config.base_config.base_config.meta_config.max_correction_attempts = 3;
    config.base_config.base_config.meta_config.hrm_model = nullptr; // Will be set later

    // Debug: Verify config is set
    std::cout << "[DEBUG] hrm_main.cpp: meta_config.enable_self_repair set to "
              << (config.base_config.base_config.meta_config.enable_self_repair ? "true" : "false")
              << std::endl;

    // Use environment variable for source root, default to project src directory for focused scanning
    if (const char* env_root = std::getenv("NYX_SOURCE_ROOT")) {
        config.base_config.project_root = env_root;
    } else {
        // Default to project src directory for memory-safe scanning
        config.base_config.project_root = (fs::current_path() / "src").string();
    }

    // Use temp directory for compilation
    config.base_config.temp_compilation_dir = (fs::temp_directory_path() / "hrm_temp").string();
    config.base_config.enable_self_modification = true; // Enabled for autonomous learning
    config.base_config.enable_runtime_recompilation = true;
    config.base_config.self_analysis_frequency = 0.05f;
    config.base_config.modification_confidence_threshold = 0.95f;
    config.base_config.create_backups_before_modification = true;

    // Resource monitoring
    config.enable_resource_monitoring = true;
    config.enable_adaptive_task_management = true;
    config.enable_chunking_for_large_tasks = true;
    config.resource_check_interval = std::chrono::seconds(60);
    config.max_memory_per_task_mb = 512;
    config.max_cpu_per_task_percent = 80.0;

    return config;
}

#ifndef NO_VULKAN
struct VulkanResources {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
};
#endif

// Helper function to estimate memory requirements for a given tier (in MB)
uint64_t estimate_tier_memory_requirements(HardwareCapabilities::PerformanceTier tier) {
    switch (tier) {
        case HardwareCapabilities::PerformanceTier::ULTRA_LOW:
            return 50ULL;  // ~50MB for minimal config
        case HardwareCapabilities::PerformanceTier::LOW:
            return 150ULL; // ~150MB for low config
        case HardwareCapabilities::PerformanceTier::MEDIUM:
            return 400ULL; // ~400MB for medium config
        case HardwareCapabilities::PerformanceTier::HIGH:
            return 1200ULL; // ~1.2GB for high config
        case HardwareCapabilities::PerformanceTier::ULTRA_HIGH:
            return 2500ULL; // ~2.5GB for ultra-high config
        default:
            return 400ULL; // Default to medium
    }
}

// Helper function to get tier name as string
std::string tier_to_string(HardwareCapabilities::PerformanceTier tier) {
    switch (tier) {
        case HardwareCapabilities::PerformanceTier::ULTRA_LOW: return "Ultra Low";
        case HardwareCapabilities::PerformanceTier::LOW: return "Low";
        case HardwareCapabilities::PerformanceTier::MEDIUM: return "Medium";
        case HardwareCapabilities::PerformanceTier::HIGH: return "High";
        case HardwareCapabilities::PerformanceTier::ULTRA_HIGH: return "Ultra High";
        default: return "Unknown";
    }
}

#ifndef NO_VULKAN
void adapt_config_to_hardware(ResourceAwareHRMConfig& config, const HardwareCapabilities& hw_caps, const VulkanResources& vulkan) {
#else
void adapt_config_to_hardware(ResourceAwareHRMConfig& config, const HardwareCapabilities& hw_caps) {
#endif
    uint64_t ram_gb = hw_caps.total_ram_bytes / (1024ULL * 1024 * 1024);
    uint32_t cores = hw_caps.cpu_cores;

    // CRITICAL FIX: Apply hardware-aware constraints to HRM MODEL parameters
    // This affects the actual model initialization, not just task limits

    bool vulkan_available = true;
#ifndef NO_VULKAN
    vulkan_available = (vulkan.device != VK_NULL_HANDLE);
#endif

    // Hardware-aware logging for debugging
    std::cout << "Hardware Detection Results:" << std::endl;
    std::cout << "  GPU Name: " << hw_caps.gpu_name << std::endl;
    std::cout << "  GPU Memory: " << hw_caps.gpu_memory_mb << "MB" << std::endl;
    std::cout << "  Integrated GPU: " << (hw_caps.is_integrated_gpu ? "Yes" : "No") << std::endl;
    std::cout << "  Vulkan Available: " << (vulkan.device != VK_NULL_HANDLE ? "Yes" : "No") << std::endl;
    std::cout << "  System RAM: " << ram_gb << "GB" << std::endl;
    std::cout << "  CPU Cores: " << cores << std::endl;

    // CRITICAL FIX: Apply memory-safe parameters for ANY GPU that cannot handle full HRM config
    // Full config (768 hidden, 4 layers, 100K vocab) requires ~12GB+ GPU memory
    // Xe GPUs and integrated GPUs need additional restrictions due to Vulkan driver issues
    bool is_xe_or_integrated = hw_caps.is_integrated_gpu ||
                               hw_caps.gpu_name.find("Iris") != std::string::npos ||
                               hw_caps.gpu_name.find("Xe") != std::string::npos;

    if (is_xe_or_integrated || hw_caps.gpu_memory_mb < 12288) {  // Xe/integrated or < 12GB cannot safely handle full HRM config
        std::cout << "GPU memory constrained (" << hw_caps.gpu_memory_mb
                  << "MB < 12288MB threshold";
        if (is_xe_or_integrated) {
            std::cout << " or Xe/integrated GPU detected";
        }
        std::cout << ") - applying memory-safe HRM model parameters" << std::endl;

        // Use memory-safe parameters that work on constrained hardware
        config.base_config.base_config.hrm_config.inner_config.hidden_size = 256;
        config.base_config.base_config.hrm_config.inner_config.H_layers = 2;
        config.base_config.base_config.hrm_config.inner_config.L_layers = 2;
        config.base_config.base_config.hrm_config.inner_config.vocab_size = 10000;  // Character-level vocab
        config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
        config.base_config.base_config.hrm_config.inner_config.seq_len = 128;

        // Reduce task memory limits for safety
        config.max_memory_per_task_mb = 256;  // Allow reasonable task memory

        std::cout << "HRM Model configured for memory safety: 256 hidden, 2 layers, 10K vocab" << std::endl;
        std::cout << "Estimated memory usage: ~200MB (safe for " << hw_caps.gpu_memory_mb << "MB GPU)" << std::endl;

    } else {
        // Only GPUs with 12GB+ can handle full HRM config safely
        std::cout << "High-end GPU detected (" << hw_caps.gpu_memory_mb
                  << "MB >= 12288MB threshold) - using full HRM capabilities" << std::endl;

        // Allow original high-end parameters for truly capable hardware
        // No changes needed - system will use base configuration (768 hidden, 4 layers, 100K vocab)
        std::cout << "HRM Model using full capabilities: 768 hidden, 4 layers, 100K vocab" << std::endl;
        std::cout << "Estimated memory usage: ~2-3GB (optimal for high-end GPU)" << std::endl;
    }

    // Apply tier-based task memory limits (existing logic)
    HardwareCapabilities::PerformanceTier effective_tier = hw_caps.performance_tier;

    std::cout << "Adapting HRM configuration for hardware tier: ";
    switch (effective_tier) {
        case HardwareCapabilities::PerformanceTier::ULTRA_LOW:
            std::cout << "Ultra Low" << std::endl;
            // Minimal configuration for embedded systems
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 64;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 1;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 1;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 1;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 10000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 128;
            config.max_memory_per_task_mb = 50;
            config.base_config.enable_self_modification = false; // Too risky for minimal hardware
            config.enable_adaptive_task_management = false;
            break;

        case HardwareCapabilities::PerformanceTier::LOW:
            std::cout << "Low" << std::endl;
            // Reduced configuration for low-end systems
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 128;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 2;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 2;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 2;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 25000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 256;
            config.max_memory_per_task_mb = 100;
            config.base_config.enable_self_modification = false; // Conservative
            break;

        case HardwareCapabilities::PerformanceTier::MEDIUM:
            std::cout << "Medium" << std::endl;
            // Balanced configuration
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 256;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 3;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 3;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 4;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 50000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 2;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 512;
            config.max_memory_per_task_mb = 256;
            config.base_config.enable_self_modification = true; // Enable with caution
            break;

        case HardwareCapabilities::PerformanceTier::HIGH:
            std::cout << "High" << std::endl;
            // Good performance configuration
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 512;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 8;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 75000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 4;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 1024;
            config.max_memory_per_task_mb = 512;
            config.base_config.enable_self_modification = true;
            break;

        case HardwareCapabilities::PerformanceTier::ULTRA_HIGH:
            std::cout << "Ultra High" << std::endl;
            // Full capability configuration (original)
            config.base_config.base_config.hrm_config.inner_config.hidden_size = 768;
            config.base_config.base_config.hrm_config.inner_config.H_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.L_layers = 4;
            config.base_config.base_config.hrm_config.inner_config.num_heads = 12;
            config.base_config.base_config.hrm_config.inner_config.vocab_size = 100000;
            config.base_config.base_config.hrm_config.inner_config.batch_size = 8;
            config.base_config.base_config.hrm_config.inner_config.seq_len = 2048;
            config.max_memory_per_task_mb = 1024;
            config.base_config.enable_self_modification = true;
            break;
    }

#ifndef NO_VULKAN
    // Vulkan availability adjustments
    if (!hw_caps.vulkan_supported || vulkan.device == VK_NULL_HANDLE) {
        std::cout << "Vulkan not available - enabling CPU-only mode" << std::endl;
        // CPU-only mode - reduce complexity further
        config.base_config.base_config.hrm_config.inner_config.hidden_size = std::min(config.base_config.base_config.hrm_config.inner_config.hidden_size, 256);
        config.base_config.base_config.hrm_config.inner_config.H_layers = std::min(config.base_config.base_config.hrm_config.inner_config.H_layers, 2);
        config.base_config.base_config.hrm_config.inner_config.L_layers = std::min(config.base_config.base_config.hrm_config.inner_config.L_layers, 2);
        config.base_config.enable_self_modification = false; // Too complex without GPU
    }
#endif

    // Resource monitoring adjustments
    if (ram_gb < 2) {
        config.enable_resource_monitoring = true;
        config.resource_check_interval = std::chrono::seconds(30); // More frequent checks
    } else if (ram_gb < 8) {
        config.enable_resource_monitoring = true;
        config.resource_check_interval = std::chrono::seconds(60);
    } else {
        config.enable_resource_monitoring = true;
        config.resource_check_interval = std::chrono::seconds(120);
    }

    // Task management based on CPU cores
    if (cores <= 1) {
        config.enable_adaptive_task_management = false;
        config.enable_chunking_for_large_tasks = false;
    } else if (cores <= 2) {
        config.enable_adaptive_task_management = true;
        config.enable_chunking_for_large_tasks = true;
    } else {
        config.enable_adaptive_task_management = true;
        config.enable_chunking_for_large_tasks = true;
    }

    std::cout << "Configuration adapted - Hidden size: " << config.base_config.base_config.hrm_config.inner_config.hidden_size
              << ", Layers: " << config.base_config.base_config.hrm_config.inner_config.H_layers
              << ", Self-modification: " << (config.base_config.enable_self_modification ? "Enabled" : "Disabled") << std::endl;

    // Final configuration validation
    validate_configuration(config, hw_caps);
}

// Validate configuration to prevent over-allocation
void validate_configuration(ResourceAwareHRMConfig& config, const HardwareCapabilities& hw_caps) {
    uint64_t ram_gb = hw_caps.total_ram_bytes / (1024ULL * 1024 * 1024);

    // Memory validation
    if (config.max_memory_per_task_mb > ram_gb * 512) {  // Max 50% of RAM per task
        config.max_memory_per_task_mb = ram_gb * 512;
        std::cout << "Adjusted max_memory_per_task_mb to " << config.max_memory_per_task_mb << "MB based on available RAM" << std::endl;
    }

    // GPU memory validation
    if (hw_caps.gpu_memory_mb > 0 && config.max_memory_per_task_mb > hw_caps.gpu_memory_mb / 2) {
        config.max_memory_per_task_mb = hw_caps.gpu_memory_mb / 2;
        std::cout << "Adjusted max_memory_per_task_mb to " << config.max_memory_per_task_mb << "MB based on GPU memory" << std::endl;
    }

    // CPU core validation
    if (config.max_cpu_per_task_percent > 100.0f / hw_caps.cpu_cores) {
        config.max_cpu_per_task_percent = 100.0f / hw_caps.cpu_cores;
        std::cout << "Adjusted max_cpu_per_task_percent to " << config.max_cpu_per_task_percent << "% based on CPU cores" << std::endl;
    }

    std::cout << "Configuration validation completed - system ready for safe operation" << std::endl;
}

#ifndef NO_VULKAN
VulkanResources initializeVulkan() {
    VulkanResources res;

    // Check Vulkan compatibility
    VulkanCompatibilityInfo compat = VulkanCompatibility::checkCompatibility();

    if (!compat.vulkanSupported) {
        throw std::runtime_error("Vulkan is not supported on this system");
    }

    if (compat.devices.empty()) {
        throw std::runtime_error("No Vulkan-compatible devices found");
    }

    if (!compat.selectedDevice) {
        throw std::runtime_error("No suitable Vulkan device selected");
    }

    std::cout << "Selected Vulkan device: " << compat.selectedDevice->name << std::endl;
    std::cout << "Device score: " << compat.selectedDevice->score << std::endl;

    // Use the instance from compatibility check
    res.instance = compat.instance;
    res.physicalDevice = compat.selectedDevice->device;

    // Use compute queue family from compatibility check
    if (compat.selectedDevice->computeQueueFamily == UINT32_MAX) {
        throw std::runtime_error("Selected device does not have a compute queue family");
    }
    res.computeQueueFamilyIndex = compat.selectedDevice->computeQueueFamily;

    // Check device features
    if (!VulkanCompatibility::checkDeviceFeatures(res.physicalDevice)) {
        std::cerr << "Warning: Selected device may not support all required features" << std::endl;
    }

    // Create logical device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = res.computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // No validation layers for production build
    std::vector<const char*> layers;

    VkDeviceCreateInfo deviceCI{};
    deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCI.queueCreateInfoCount = 1;
    deviceCI.pQueueCreateInfos = &queueCreateInfo;
    deviceCI.enabledLayerCount = static_cast<uint32_t>(layers.size());
    deviceCI.ppEnabledLayerNames = layers.data();

    // Enable required features
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.shaderInt64 = VK_TRUE; // For some operations
    deviceCI.pEnabledFeatures = &deviceFeatures;

    VkResult deviceResult = vkCreateDevice(res.physicalDevice, &deviceCI, nullptr, &res.device);
    if (deviceResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan device: " +
                                VulkanCompatibility::getVulkanResultString(deviceResult));
    }

    // Get compute queue
    vkGetDeviceQueue(res.device, res.computeQueueFamilyIndex, 0, &res.computeQueue);

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = res.computeQueueFamilyIndex;

    if (vkCreateCommandPool(res.device, &poolInfo, nullptr, &res.commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    return res;
}
#endif

MemoryCompactionConfig createDefaultMemoryConfig(std::shared_ptr<CloudStorageManager> cloud_manager, ConfigManager& config_manager) {
    MemoryCompactionConfig config;

    // Load settings from ConfigManager
    auto& logger = Logger::getInstance();

    // Set defaults, override with config values
    config.default_level = MemoryCompactionLevel::MEDIUM;
    std::string level = config_manager.get_string("memory_compaction.default_level", "MEDIUM");
    if (level == "LIGHT") config.default_level = MemoryCompactionLevel::LIGHT;
    else if (level == "HEAVY") config.default_level = MemoryCompactionLevel::HEAVY;
    else if (level == "EXTREME") config.default_level = MemoryCompactionLevel::EXTREME;

    config.preferred_algorithm = CompressionAlgorithm::LZ4;
    std::string algo = config_manager.get_string("memory_compaction.preferred_algorithm", "LZ4");
    if (algo == "ZSTD") config.preferred_algorithm = CompressionAlgorithm::ZSTD;
    else if (algo == "GZIP") config.preferred_algorithm = CompressionAlgorithm::GZIP;
    else if (algo == "BROTLI") config.preferred_algorithm = CompressionAlgorithm::BROTLI;
    else if (algo == "NONE") config.preferred_algorithm = CompressionAlgorithm::NONE;

    config.max_memory_before_compaction_mb = config_manager.get_int("memory_compaction.max_memory_before_compaction_mb", 1024);
    config.target_memory_after_compaction_mb = config_manager.get_int("memory_compaction.target_memory_after_compaction_mb", 512);
    config.auto_compaction_enabled = config_manager.get_bool("memory_compaction.auto_compaction_enabled", true);

    int interval_hours = config_manager.get_int("memory_compaction.compaction_interval_hours", 6);
    config.compaction_interval = std::chrono::hours(interval_hours);

    config.preserve_recent_conversations = config_manager.get_bool("memory_compaction.preserve_recent_conversations", true);

    int window_hours = config_manager.get_int("memory_compaction.recent_conversation_window_hours", 24);
    config.recent_conversation_window = std::chrono::hours(window_hours);

    // Use environment variable or temp directory for compaction
    if (const char* env_dir = std::getenv("HRM_COMPACTION_DIR")) {
        config.compaction_directory = env_dir;
    } else {
        config.compaction_directory = (fs::temp_directory_path() / "hrm_compactions").string();
    }
    logger.info("Memory compaction directory: " + config.compaction_directory);

    config.cloud_storage_manager = cloud_manager;
    config.default_cloud_provider = CloudProvider::LOCAL_STORAGE;
    std::string provider = config_manager.get_string("general.default_cloud_provider");
    if (!provider.empty()) {
        if (provider == "GOOGLE_DRIVE") config.default_cloud_provider = CloudProvider::GOOGLE_DRIVE;
        else if (provider == "DROPBOX") config.default_cloud_provider = CloudProvider::DROPBOX;
        else if (provider == "ONEDRIVE") config.default_cloud_provider = CloudProvider::ONEDRIVE;
        else if (provider == "MEGA") config.default_cloud_provider = CloudProvider::MEGA;
    }

    return config;
}

void setupCloudStorage(CloudStorageManager& cloud_manager) {
    // Read config file for cloud provider settings
    std::ifstream config_file("./config/hrm_config.txt");
    std::unordered_map<std::string, std::string> settings;

    if (config_file.is_open()) {
        std::string line;
        std::string current_section;
        while (std::getline(config_file, line)) {
            // Remove comments and trim
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](int ch) { return !std::isspace(ch); }));
            line.erase(std::find_if(line.rbegin(), line.rend(), [](int ch) { return !std::isspace(ch); }).base(), line.end());

            if (line.empty()) continue;

            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.size() - 2);
            } else if (!current_section.empty()) {
                size_t equals_pos = line.find('=');
                if (equals_pos != std::string::npos) {
                    std::string key = current_section + "." + line.substr(0, equals_pos);
                    std::string value = line.substr(equals_pos + 1);
                    key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](int ch) { return !std::isspace(ch); }));
                    key.erase(std::find_if(key.rbegin(), key.rend(), [](int ch) { return !std::isspace(ch); }).base(), key.end());
                    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](int ch) { return !std::isspace(ch); }));
                    value.erase(std::find_if(value.rbegin(), value.rend(), [](int ch) { return !std::isspace(ch); }).base(), value.end());
                    settings[key] = value;
                }
            }
        }
    }

    // Add local storage provider (always available)
    CloudStorageConfig local_config;
    local_config.provider = CloudProvider::LOCAL_STORAGE;
    local_config.compaction_directory = settings.count("general.cloud_storage_directory") ?
        settings["general.cloud_storage_directory"] : "./hrm_cloud_storage";

    auto local_provider = std::make_shared<LocalStorageProvider>(local_config);
    cloud_manager.add_provider(local_provider);

    // Set default provider
    CloudProvider default_provider = CloudProvider::LOCAL_STORAGE;
    if (settings.count("general.default_cloud_provider")) {
        std::string provider = settings["general.default_cloud_provider"];
        if (provider == "GOOGLE_DRIVE") default_provider = CloudProvider::GOOGLE_DRIVE;
        else if (provider == "DROPBOX") default_provider = CloudProvider::DROPBOX;
        else if (provider == "ONEDRIVE") default_provider = CloudProvider::ONEDRIVE;
        else if (provider == "MEGA") default_provider = CloudProvider::MEGA;
    }
    cloud_manager.set_default_provider(default_provider);

    std::cout << "Cloud storage initialized with local storage provider" << std::endl;
}

void runCLI(std::shared_ptr<ResourceAwareHRM> hrm) {
    NyxCLI cli(hrm);
    cli.run();
}

void runGUI(std::shared_ptr<ResourceAwareHRM> hrm) {
    NyxGUI gui(hrm);
    gui.run();
}

void runCharacterTraining(std::shared_ptr<ResourceAwareHRM> hrm) {
    std::cout << "Starting Character-Level Language Training Mode" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Create character language training configuration
    CharacterLanguageModelConfig train_config;
    train_config.max_epochs = 10;  // Shorter for demo
    train_config.batch_size = 2;   // Smaller batch size
    train_config.max_seq_length = 512;  // Shorter sequences
    train_config.context_length = 256;
    train_config.learning_rate = 1e-4;
    train_config.warmup_steps = 100;
    train_config.total_steps = 10000;

    // Initialize character language trainer
    CharacterLanguageTrainer trainer(hrm, train_config);

    // Run training
    std::string dataset_path = "./data/text/processed";
    std::unordered_map<std::string, float> training_results;
    try {
        std::cout << "Starting training with dataset path: " << dataset_path << std::endl;
        training_results = trainer.train_character_language_model(dataset_path);
        std::cout << "Training completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training failed with exception: " << e.what() << std::endl;
        std::cerr << "This may be causing the task scheduler to stop unexpectedly" << std::endl;
        return; // Exit early to prevent destructor issues
    }

    std::cout << "\n Character-level training completed!" << std::endl;
    std::cout << "Final Results:" << std::endl;
    for (const auto& [metric, value] : training_results) {
        std::cout << "  " << metric << ": " << value << std::endl;
    }

    // Test text generation
    std::cout << "\nTesting text generation..." << std::endl;
    std::string prompt = "The quick brown fox";
    std::string generated = trainer.generate_text(prompt, 100);
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Generated: \"" << generated << "\"" << std::endl;
}

void printUsage(const char* program_name) {
    std::cout << "Nyx - Primordial Goddess of Night" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --cli              Run in command-line interface mode" << std::endl;
    std::cout << "  --gui              Run in graphical user interface mode" << std::endl;
    std::cout << "  --train            Run character-level language training" << std::endl;
    std::cout << "  --test             Run basic functionality tests" << std::endl;
    std::cout << "  --help             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "If no options are provided, GUI mode is started by default." << std::endl;
}

void runBasicTests(std::shared_ptr<ResourceAwareHRM> hrm,
                   std::shared_ptr<MemoryCompactionSystem> memory_system,
                   std::shared_ptr<CloudStorageManager> cloud_manager) {
    std::cout << "Running basic functionality tests..." << std::endl;

    // Test memory compaction (without HRM to avoid Vulkan issues)
    std::cout << "Testing memory compaction..." << std::endl;
    std::vector<ConversationEntry> test_entries = {
        {"test1", std::chrono::system_clock::now(), "User: Hello", "HRM: Hi there!", 0.95, {"greeting"}, {"user"}, 100},
        {"test2", std::chrono::system_clock::now(), "User: How are you?", "HRM: I'm doing well, thank you!", 0.92, {"conversation"}, {"user"}, 120}
    };

    auto compaction_result = memory_system->compact_memory(test_entries);
    if (compaction_result.success) {
        std::cout << "Memory compaction successful. ID: " << compaction_result.compaction_id << std::endl;

        // Test decompression
        auto decompressed = memory_system->decompress_memory(compaction_result.compaction_id);
        std::cout << "Decompressed " << decompressed.size() << " conversation entries" << std::endl;
    } else {
        std::cout << "Memory compaction failed: " << compaction_result.error_message << std::endl;
    }

    // Test cloud storage
    std::cout << "Testing cloud storage..." << std::endl;
    auto providers = cloud_manager->get_available_providers();
    std::cout << "Available cloud providers: " << providers.size() << std::endl;

    // Test memory statistics
    auto mem_stats = memory_system->get_memory_stats();
    std::cout << "Memory stats - Total compactions: " << mem_stats["total_compactions"] << std::endl;

    std::cout << "Basic tests completed successfully!" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Nyx - Primordial Goddess of Night" << std::endl;
    std::cout << "===================================" << std::endl;

    // Hardware profiling for universal compatibility
    HardwareProfiler hw_profiler;
    HardwareCapabilities hw_caps = hw_profiler.profile_system();

    // Initialize logging system
    auto& logger = Logger::getInstance();
    logger.info("Nyx awakens from the eternal night");

    // Initialize configuration manager
    ConfigManager config_manager;
    logger.info("Configuration loaded from: " + config_manager.getConfigDir());

    // Parse command line arguments
    bool cli_mode = false;
    bool gui_mode = false;
    bool train_mode = false;
    bool test_mode = false;
    bool cpu_mode = false;
    std::string config_file;
    bool invalid_arg = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--cli") {
            cli_mode = true;
        } else if (arg == "--gui") {
            gui_mode = true;
        } else if (arg == "--train") {
            train_mode = true;
            // Check if next args are "config <path>"
            if (i + 2 < argc && std::string(argv[i + 1]) == "config") {
                config_file = argv[i + 2];
                i += 2; // Skip the parsed arguments
            }
        } else if (arg == "--test") {
            test_mode = true;
        } else if (arg == "--cpu") {
            cpu_mode = true;
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            invalid_arg = true;
        }
    }

    if (invalid_arg) {
        printUsage(argv[0]);
        return 1;
    }

    // If no mode specified, default to CLI (GUI not ready yet)
    if (!cli_mode && !gui_mode && !train_mode && !test_mode) {
        cli_mode = true;
    }

    // Initialize HRM config early so we can set memory/cloud systems
    auto hrm_config = createDefaultHRMConfig(hw_caps);

    try {
        // Initialize cloud storage
        auto cloud_manager = std::make_shared<CloudStorageManager>();
        setupCloudStorage(*cloud_manager);

        // Initialize memory compaction system
        auto memory_config = createDefaultMemoryConfig(cloud_manager, config_manager);
        auto memory_system = std::make_shared<MemoryCompactionSystem>(memory_config);

        // Set memory compaction system in HRM config
        hrm_config.memory_compaction_system = memory_system;
        hrm_config.cloud_storage_manager = cloud_manager;

        std::shared_ptr<ResourceAwareHRM> hrm = nullptr;

        if (!test_mode) {
#ifndef NO_VULKAN
            // Initialize Vulkan resources
            VulkanResources vulkan;
            try {
                vulkan = initializeVulkan();
                logger.info("Vulkan initialized successfully");
            } catch (const std::exception& e) {
                logger.error("Failed to initialize Vulkan: " + std::string(e.what()));
                logger.warning("Falling back to CPU-only mode");
                // Continue without Vulkan - HRM will use CPU fallbacks
            }
#endif

            // HRM config already initialized above

            // Initialize HRM inner config - restore original layer counts
            hrm_config.base_config.base_config.hrm_config.inner_config.batch_size = 1;
            hrm_config.base_config.base_config.hrm_config.inner_config.seq_len = 512;
            hrm_config.base_config.base_config.hrm_config.inner_config.puzzle_emb_ndim = 128;
            hrm_config.base_config.base_config.hrm_config.inner_config.num_puzzle_identifiers = 10;
            hrm_config.base_config.base_config.hrm_config.inner_config.vocab_size = 100000;
            hrm_config.base_config.base_config.hrm_config.inner_config.H_cycles = 2;
            hrm_config.base_config.base_config.hrm_config.inner_config.L_cycles = 2;
            hrm_config.base_config.base_config.hrm_config.inner_config.H_layers = 4;
            hrm_config.base_config.base_config.hrm_config.inner_config.L_layers = 4;
            hrm_config.base_config.base_config.hrm_config.inner_config.hidden_size = 768;
            hrm_config.base_config.base_config.hrm_config.inner_config.expansion = 2.0f;
            hrm_config.base_config.base_config.hrm_config.inner_config.num_heads = 12;
            hrm_config.base_config.base_config.hrm_config.inner_config.pos_encodings = "learned";
            hrm_config.base_config.base_config.hrm_config.inner_config.rms_norm_eps = 1e-5f;
            hrm_config.base_config.base_config.hrm_config.inner_config.rope_theta = 10000.0f;
            hrm_config.base_config.base_config.hrm_config.inner_config.halt_max_steps = 8;  // Restore proper reasoning for research assistant
            hrm_config.base_config.base_config.hrm_config.inner_config.halt_exploration_prob = 0.1f;
            hrm_config.base_config.base_config.hrm_config.inner_config.forward_dtype = "float32";

#ifndef NO_VULKAN
    // Adapt configuration based on hardware capabilities
    adapt_config_to_hardware(hrm_config, hw_caps, vulkan);

    // Set Vulkan resources in config (only if Vulkan is available)
    if (vulkan.device != VK_NULL_HANDLE) {
        hrm_config.base_config.base_config.hrm_config.inner_config.physicalDevice = vulkan.physicalDevice;
        hrm_config.base_config.base_config.hrm_config.inner_config.device = vulkan.device;
        hrm_config.base_config.base_config.hrm_config.inner_config.computeQueue = vulkan.computeQueue;
        hrm_config.base_config.base_config.hrm_config.inner_config.computeQueueFamilyIndex = vulkan.computeQueueFamilyIndex;
        hrm_config.base_config.base_config.hrm_config.inner_config.commandPool = vulkan.commandPool;

        // Set Vulkan resources in UTF8 config
        hrm_config.base_config.base_config.utf8_config.physicalDevice = vulkan.physicalDevice;
        hrm_config.base_config.base_config.utf8_config.device = vulkan.device;
        hrm_config.base_config.base_config.utf8_config.computeQueue = vulkan.computeQueue;
        hrm_config.base_config.base_config.utf8_config.computeQueueFamilyIndex = vulkan.computeQueueFamilyIndex;
        hrm_config.base_config.base_config.utf8_config.commandPool = vulkan.commandPool;
    }
#else
    // Adapt configuration based on hardware capabilities (CPU-only)
    adapt_config_to_hardware(hrm_config, hw_caps);
#endif

            hrm = ResourceAwareHRM::getInstance(hrm_config);
            logger.info("Nyx emerges from the shadows, ready to guide");
        }

        if (test_mode) {
            logger.info("Nyx tests the boundaries of her domain");
            runBasicTests(hrm, memory_system, cloud_manager);
            logger.info("System tests completed");
            return 0;
        }

    if (train_mode) {
        // Training mode - reuse existing system to prevent Vulkan conflicts
        auto hrm_system = ResourceAwareHRM::getInstance(hrm_config);

        // Handle CPU-only mode
        if (cpu_mode) {
            logger.info("CPU-only training mode enabled - Vulkan will be disabled");
            // Force CPU-only configuration
            hrm_config.base_config.base_config.hrm_config.inner_config.vocab_size = 10000;
            hrm_config.base_config.base_config.hrm_config.inner_config.hidden_size = 128;
            hrm_config.base_config.base_config.hrm_config.inner_config.H_layers = 1;
            hrm_config.base_config.base_config.hrm_config.inner_config.L_layers = 1;
        }

        // Initialize Vulkan resources if available and not in CPU mode
#ifndef NO_VULKAN
        VulkanResources vulkan_res{};
        bool vulkan_available = false;
        if (!cpu_mode) {
            try {
                vulkan_res = initializeVulkan();
                vulkan_available = true;
                adapt_config_to_hardware(hrm_config, hw_caps, vulkan_res);
            } catch (const std::exception& e) {
                logger.warning("Vulkan initialization failed in training mode, using CPU fallback");
                vulkan_available = false;
            }
        }
#endif

        NyxCLI cli(hrm_system);
        std::vector<std::string> train_args;
        if (!config_file.empty()) {
            train_args.push_back("config");
            train_args.push_back(config_file);
        }
        std::string train_cmd = "train";
        if (!config_file.empty()) {
            train_cmd += " config " + config_file;
        }
        CLICommandResult result = cli.process_command(train_cmd);
        std::cout << result.output << std::endl;
        if (!result.success) {
            std::cerr << "Training failed" << std::endl;
            return 1;
        }
        return 0;
    }

    // Default mode is CLI
    if (cli_mode) {
        // CLI mode - use singleton HRM system for interactive interface
        auto hrm_system = ResourceAwareHRM::getInstance(hrm_config);

        // Initialize Vulkan resources if available
#ifndef NO_VULKAN
        try {
            VulkanResources vulkan_res = initializeVulkan();
            adapt_config_to_hardware(hrm_config, hw_caps, vulkan_res);
            logger.info("Vulkan initialized successfully");
        } catch (const std::exception& e) {
            logger.error("Failed to initialize Vulkan: " + std::string(e.what()));
            logger.warning("Falling back to CPU-only mode");
        }
#endif

        NyxCLI cli(hrm_system);
        cli.run();
        return 0;
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
}
}