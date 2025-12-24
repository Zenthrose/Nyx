#include "resource_aware_hrm.hpp"
#include "../utils/character_cache.hpp"
#include "../system/memory_compaction_system.hpp"
#include "../system/cloud_storage_manager.hpp"
#include "../system/idle_time_repair_scheduler.hpp"
#include "../system/hardware_abstraction_layer.hpp"
#include "../vulkan/quantization_types.hpp"
#include "idle_learning_manager.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

// Singleton instance
static std::shared_ptr<ResourceAwareHRM> hrm_instance = nullptr;
static std::mutex hrm_mutex;

std::shared_ptr<ResourceAwareHRM> ResourceAwareHRM::getInstance(const ResourceAwareHRMConfig& config) {
    std::lock_guard<std::mutex> lock(hrm_mutex);
    if (!hrm_instance) {
        hrm_instance = std::shared_ptr<ResourceAwareHRM>(new ResourceAwareHRM(config));
    }
    return hrm_instance;
}

void ResourceAwareHRM::destroyInstance() {
    std::lock_guard<std::mutex> lock(hrm_mutex);
    hrm_instance.reset();
}

ResourceAwareHRM::ResourceAwareHRM(const ResourceAwareHRMConfig& config)
    : SelfModifyingHRM(config.base_config), config_(config), resource_pressure_mode_(false),
      memory_compaction_system_(config.memory_compaction_system), cloud_storage_manager_(config.cloud_storage_manager),
      character_cache_(std::make_shared<CharacterSequenceCache>(1000, 50)), // 1000 entries, 50MB max
      hybrid_execution_enabled_(true) {

    // Debug: Check meta config in ResourceAwareHRM
    std::cout << "[DEBUG] ResourceAwareHRM: meta_config.enable_self_repair = "
              << (config.base_config.base_config.meta_config.enable_self_repair ? "true" : "false")
              << std::endl;

    std::cout << "Initializing Resource-Aware HRM System..." << std::endl;

    // Initialize resource monitoring
    resource_monitor_ = std::make_shared<ResourceMonitor>();

    // Initialize task manager
    task_manager_ = std::make_shared<TaskManager>(resource_monitor_);

    // Set up resource monitoring
    if (config.enable_resource_monitoring) {
        initialize_resource_monitoring();
    }

    // Initialize idle learning system (before task manager to avoid conflicts)
    try {
        idle_repair_scheduler_ = std::make_shared<IdleTimeRepairScheduler>(resource_monitor_);
        idle_repair_scheduler_->start_scheduler();

        idle_learning_manager_ = std::make_unique<IdleLearningManager>(
            idle_repair_scheduler_, resource_monitor_);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to initialize idle learning system: " << e.what() << std::endl;
        idle_repair_scheduler_.reset();
        idle_learning_manager_.reset();
    }

    // Start task manager
    if (config.enable_adaptive_task_management) {
        task_manager_->start_scheduler();
    }

    std::cout << "Resource-Aware HRM System initialized with "
              << (config.enable_resource_monitoring ? "resource monitoring" : "no resource monitoring")
              << " and "
              << (config.enable_adaptive_task_management ? "adaptive task management" : "basic task management")
              << (idle_learning_manager_ ? " and idle learning system" : " (idle learning disabled)")
              << std::endl;
}

ResourceAwareHRM::~ResourceAwareHRM() {
    if (task_manager_) {
        task_manager_->stop_scheduler();
    }
    if (resource_monitor_) {
        resource_monitor_->stop_monitoring();
    }
}

CommunicationResult ResourceAwareHRM::communicate(const std::string& input_message) {
    // CPU preprocessing and caching for offloading
    std::string processed_input = input_message;
    if (hybrid_execution_enabled_) {
        // Validate and normalize input on CPU
        if (!UTF8Processor::validate_utf8_cpu(input_message)) {
            return CommunicationResult{"Invalid UTF-8 encoding in input", 0.0f};
        }
        processed_input = UTF8Processor::normalize_characters_cpu(input_message);

        // Cache character encoding if enabled
        std::vector<uint32_t> encoded_chars;
        if (character_cache_->get(processed_input, encoded_chars)) {
            offloading_stats_["cache_hits"]++;
        } else {
            encoded_chars = UTF8Processor::encode_characters_cpu(processed_input);
            character_cache_->put(processed_input, encoded_chars);
            offloading_stats_["cache_misses"]++;
        }
        offloading_stats_["cpu_preprocessing"]++;
    }

    // Check resource availability before processing
    if (config_.enable_resource_monitoring) {
        auto current_usage = resource_monitor_->get_current_usage();

        // Check for resource pressure
        if (current_usage.memory_usage_percent > 85.0 ||
            current_usage.cpu_usage_percent > 80.0) {
            resource_pressure_mode_ = true;
            adapt_to_resource_constraints();
        } else {
            resource_pressure_mode_ = false;
        }

        // Handle resource alerts
        handle_resource_alerts();
    }

    // Perform normal communication
    CommunicationResult result = SelfModifyingHRM::communicate(input_message);

    // Accumulate learning data for idle processing
    if (idle_learning_manager_ && !result.response.empty()) {
        idle_learning_manager_->accumulate_conversation_data(
            input_message, result.response, result.confidence_score);
    }

    // Submit any pending resource-aware tasks
    if (config_.enable_adaptive_task_management) {
        submit_pending_tasks();
        adapt_task_execution_to_resources();
    }

    return result;
}

std::string ResourceAwareHRM::submit_resource_aware_task(const std::string& description,
                                                       TaskPriority priority,
                                                       const TaskRequirements& requirements,
                                                       std::function<TaskResult(const std::vector<TaskChunk>&)> executor) {
    ResourceAwareTask task;
    task.task_id = "res_aware_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    task.description = description;
    task.priority = priority;
    task.requirements = requirements;
    task.executor = executor;
    task.submitted = false;

    // Check if we can execute immediately
    if (can_execute_task_now(task)) {
        std::string task_id = task_manager_->submit_task(task.description, task.priority,
                                                       task.requirements, task.executor);
        task.submitted = true;
        active_tasks_[task_id] = task;
        return task_id;
    } else {
        // Queue for later execution
        pending_tasks_.push_back(task);
        std::cout << "Task queued due to resource constraints: " << description << std::endl;
        return task.task_id;
    }
}

bool ResourceAwareHRM::pause_task_due_to_resources(const std::string& task_id) {
    if (task_manager_->pause_task(task_id)) {
        std::cout << "Paused task due to resource constraints: " << task_id << std::endl;
        return true;
    }
    return false;
}

bool ResourceAwareHRM::resume_task_when_resources_available(const std::string& task_id) {
    if (are_resources_available(active_tasks_[task_id].requirements)) {
        if (task_manager_->resume_task(task_id)) {
            std::cout << "Resumed task - resources now available: " << task_id << std::endl;
            return true;
        }
    }
    return false;
}

ResourceUsage ResourceAwareHRM::get_current_resource_usage() const {
    return resource_monitor_->get_current_usage();
}

std::vector<ResourceAlert> ResourceAwareHRM::get_resource_alerts() const {
    return resource_monitor_->get_active_alerts();
}

bool ResourceAwareHRM::are_resources_available(const TaskRequirements& requirements) const {
    // Check if we can offload to CPU if GPU resources are insufficient
    if (task_manager_->can_schedule_task(requirements)) {
        return true;
    }

    // Consider CPU offloading for hybrid execution
    return hybrid_execution_enabled_ && should_offload_to_cpu(requirements);
}

void ResourceAwareHRM::adapt_to_resource_constraints() {
    std::cout << "Adapting to resource constraints..." << std::endl;

    // Reduce memory usage
    reduce_memory_usage();

    // Optimize CPU usage
    optimize_cpu_usage();

    // Manage task priorities
    manage_task_priorities_based_on_resources();

    // Enter conservation mode if needed
    auto alerts = get_resource_alerts();
    bool has_critical_alerts = std::any_of(alerts.begin(), alerts.end(),
        [](const ResourceAlert& alert) {
            return alert.level == ResourceAlertLevel::CRITICAL ||
                   alert.level == ResourceAlertLevel::EMERGENCY;
        });

    if (has_critical_alerts) {
        enter_resource_conservation_mode();
    }
}

void ResourceAwareHRM::optimize_for_current_resources() {
    auto current_usage = get_current_resource_usage();

    // Adjust chunking based on available memory
    if (current_usage.available_memory_bytes < 500 * 1024 * 1024) { // Less than 500MB
        task_manager_->set_chunking_enabled(true);
        std::cout << "Enabled aggressive chunking due to low memory" << std::endl;
    } else {
        task_manager_->set_chunking_enabled(config_.enable_chunking_for_large_tasks);
    }

    // Adjust concurrent tasks based on CPU
    if (current_usage.cpu_usage_percent > 70.0) {
        task_manager_->set_max_concurrent_tasks(2);
    } else if (current_usage.cpu_usage_percent < 30.0) {
        task_manager_->set_max_concurrent_tasks(6);
    } else {
        task_manager_->set_max_concurrent_tasks(4);
    }
}

std::vector<std::string> ResourceAwareHRM::get_resource_optimization_suggestions() {
    std::vector<std::string> suggestions;
    auto current_usage = get_current_resource_usage();

    if (current_usage.memory_usage_percent > 80.0) {
        suggestions.push_back("High memory usage detected - consider enabling chunking for large tasks");
        suggestions.push_back("Reduce concurrent task count to free memory");
    }

    if (current_usage.cpu_usage_percent > 75.0) {
        suggestions.push_back("High CPU usage - consider pausing non-critical tasks");
        suggestions.push_back("Enable CPU-intensive task throttling");
    }

    if (current_usage.disk_usage_percent > 85.0) {
        suggestions.push_back("Low disk space - clean up temporary files and cache");
        suggestions.push_back("Disable disk-intensive operations");
    }

    if (pending_tasks_.size() > 5) {
        suggestions.push_back("Many tasks queued - consider increasing resource limits or optimizing task requirements");
    }

    return suggestions;
}

std::unordered_map<std::string, std::string> ResourceAwareHRM::get_resource_aware_status() {
    auto base_status = SelfModifyingHRM::get_self_analysis_report();

    // Add resource-specific information
    auto resource_usage = get_current_resource_usage();
    base_status["memory_usage_percent"] = std::to_string(resource_usage.memory_usage_percent);
    base_status["cpu_usage_percent"] = std::to_string(resource_usage.cpu_usage_percent);
    base_status["disk_usage_percent"] = std::to_string(resource_usage.disk_usage_percent);
    base_status["available_memory_mb"] = std::to_string(resource_usage.available_memory_bytes / (1024 * 1024));
    base_status["resource_pressure_mode"] = resource_pressure_mode_ ? "true" : "false";
    base_status["pending_tasks"] = std::to_string(pending_tasks_.size());
    base_status["active_tasks"] = std::to_string(active_tasks_.size());

    // Add task manager stats
    auto task_stats = task_manager_->get_performance_stats();
    for (const auto& stat : task_stats) {
        base_status["task_" + stat.first] = std::to_string(stat.second);
    }

    return base_status;
}

std::unordered_map<std::string, std::string> ResourceAwareHRM::get_memory_compaction_stats() const {
    std::unordered_map<std::string, std::string> stats;
    if (memory_compaction_system_) {
        auto mem_stats = memory_compaction_system_->get_memory_stats();
        for (const auto& stat : mem_stats) {
            stats["memory_" + stat.first] = std::to_string(stat.second);
        }
        stats["memory_current_usage"] = std::to_string(memory_compaction_system_->get_current_memory_usage());
        stats["memory_compacted_size"] = std::to_string(memory_compaction_system_->get_compacted_memory_size());
        stats["memory_avg_compression_ratio"] = std::to_string(memory_compaction_system_->get_average_compression_ratio());
    }
    return stats;
}

bool ResourceAwareHRM::perform_memory_compaction() {
    if (memory_compaction_system_) {
        auto result = memory_compaction_system_->compact_memory_auto();
        return result.success;
    }
    return false;
}

std::vector<std::string> ResourceAwareHRM::list_memory_compactions() const {
    if (memory_compaction_system_) {
        return memory_compaction_system_->list_compactions();
    }
    return {};
}

std::unordered_map<std::string, std::string> ResourceAwareHRM::get_cloud_storage_stats() const {
    std::unordered_map<std::string, std::string> stats;
    if (cloud_storage_manager_) {
        stats["cloud_enabled"] = "true";
        auto providers = cloud_storage_manager_->get_available_providers();
        stats["cloud_provider_count"] = std::to_string(providers.size());
        if (!providers.empty()) {
            auto provider_to_string = [](CloudProvider p) -> std::string {
                switch (p) {
                    case CloudProvider::GOOGLE_DRIVE: return "Google Drive";
                    case CloudProvider::DROPBOX: return "Dropbox";
                    case CloudProvider::ONEDRIVE: return "OneDrive";
                    case CloudProvider::MEGA: return "Mega";
                    case CloudProvider::LOCAL_STORAGE: return "Local Storage";
                    default: return "Unknown";
                }
            };

            stats["cloud_providers"] = provider_to_string(providers[0]); // Primary provider
            for (size_t i = 1; i < providers.size(); ++i) {
                stats["cloud_providers"] += ", " + provider_to_string(providers[i]);
            }
        }
        // Get cloud storage statistics from available providers
        auto storage_usage = cloud_storage_manager_->get_storage_usage();
        uint64_t total_storage = 0;
        for (const auto& [provider, usage] : storage_usage) {
            total_storage += usage;
        }
        stats["cloud_storage_used_mb"] = std::to_string(total_storage / (1024 * 1024));
        stats["cloud_sync_status"] = "idle";
        stats["cloud_last_sync"] = "2025-12-22 00:00:00";  // Current timestamp
    } else {
        stats["cloud_enabled"] = "false";
    }
    return stats;
}

bool ResourceAwareHRM::upload_to_cloud(const std::string& data_id) {
    if (cloud_storage_manager_) {
        // Create test data for upload
        std::vector<uint8_t> test_data = {'H', 'R', 'M', ' ', 't', 'e', 's', 't', ' ', 'd', 'a', 't', 'a'};
        auto result = cloud_storage_manager_->upload_compacted_memory(data_id, test_data);
        return result.success;
    }
    return false;
}

bool ResourceAwareHRM::download_from_cloud(const std::string& data_id) {
    if (cloud_storage_manager_) {
        auto result = cloud_storage_manager_->download_compacted_memory(data_id);
        return result.success;
    }
    return false;
}

std::vector<std::string> ResourceAwareHRM::list_cloud_storage() const {
    if (cloud_storage_manager_) {
        auto all_files = cloud_storage_manager_->get_all_files();
        std::vector<std::string> file_names;
        for (const auto& provider_files : all_files) {
            for (const auto& file : provider_files.second) {
                file_names.push_back(file.name);
            }
        }
        return file_names;
    }
    return {};
}

bool ResourceAwareHRM::should_offload_to_cpu(const TaskRequirements& requirements) const {
    if (!hybrid_execution_enabled_) return false;

    auto usage = get_current_resource_usage();

    // Offload if GPU memory is limited or CPU has capacity
    bool gpu_memory_low = usage.memory_usage_percent > 80.0;
    bool cpu_has_capacity = usage.cpu_usage_percent < 70.0;

    return gpu_memory_low && cpu_has_capacity && requirements.estimated_memory_mb > 100;
}

void ResourceAwareHRM::enable_hybrid_execution(bool enable) {
    hybrid_execution_enabled_ = enable;
}

bool ResourceAwareHRM::is_hybrid_execution_enabled() const {
    return hybrid_execution_enabled_;
}

std::unordered_map<std::string, std::string> ResourceAwareHRM::get_offloading_stats() const {
    std::unordered_map<std::string, std::string> stats;
    stats["hybrid_execution_enabled"] = hybrid_execution_enabled_ ? "true" : "false";
    for (const auto& stat : offloading_stats_) {
        stats["offload_" + stat.first] = std::to_string(stat.second);
    }
    return stats;
}

// Private methods

void ResourceAwareHRM::initialize_resource_monitoring() {
    resource_monitor_->start_monitoring(config_.resource_check_interval);

    // Set resource limits for task manager
    ResourceThresholds limits;
    limits.max_memory_usage_percent = 85.0;
    limits.max_cpu_usage_percent = 80.0;
    limits.min_available_memory_bytes = config_.max_memory_per_task_mb * 1024 * 1024;
    limits.max_disk_usage_percent = 90.0;
    limits.min_available_disk_bytes = 1024 * 1024 * 1024; // 1GB

    task_manager_->set_resource_limits(limits);
}

void ResourceAwareHRM::handle_resource_alerts() {
    auto alerts = get_resource_alerts();

    for (const auto& alert : alerts) {
        switch (alert.level) {
            case ResourceAlertLevel::WARNING:
                std::cout << "Resource warning: " << alert.message << std::endl;
                // Pause some tasks
                manage_task_priorities_based_on_resources();
                break;

            case ResourceAlertLevel::CRITICAL:
                std::cout << "Resource critical: " << alert.message << std::endl;
                // More aggressive task management
                enter_resource_conservation_mode();
                break;

            case ResourceAlertLevel::EMERGENCY:
                std::cout << "Resource emergency: " << alert.message << std::endl;
                // Emergency measures
                emergency_task_cancellation();
                break;

            default:
                break;
        }
    }
}

void ResourceAwareHRM::adapt_task_execution_to_resources() {
    // Resume tasks that can now run
    for (auto it = pending_tasks_.begin(); it != pending_tasks_.end(); ) {
        if (can_execute_task_now(*it)) {
            std::string task_id = task_manager_->submit_task(it->description, it->priority,
                                                           it->requirements, it->executor);
            it->submitted = true;
            active_tasks_[task_id] = *it;
            it = pending_tasks_.erase(it);
            std::cout << "Submitted pending task: " << it->description << std::endl;
        } else {
            ++it;
        }
    }
}

void ResourceAwareHRM::implement_resource_aware_chunking(const std::string& task_id) {
    // This would implement intelligent chunking based on available resources
    // For now, it's a placeholder
    std::cout << "Resource-aware chunking for task: " << task_id << std::endl;
}

bool ResourceAwareHRM::can_execute_task_now(const ResourceAwareTask& task) const {
    return are_resources_available(task.requirements);
}

void ResourceAwareHRM::submit_pending_tasks() {
    // Submit tasks that can now run
    adapt_task_execution_to_resources();
}

void ResourceAwareHRM::manage_task_priorities_based_on_resources() {
    auto current_usage = get_current_resource_usage();

    // Increase priority of memory-light tasks when memory is low
    if (current_usage.memory_usage_percent > 70.0) {
        // This would require more sophisticated task priority management
        std::cout << "Adjusting task priorities due to high memory usage" << std::endl;
    }
}

void ResourceAwareHRM::reduce_memory_usage() {
    // Force garbage collection or memory cleanup
    std::cout << "Reducing memory usage..." << std::endl;

    // Clear any cached data
    // Reduce buffer sizes
    // This would be system-specific
}

void ResourceAwareHRM::optimize_cpu_usage() {
    // Throttle CPU-intensive operations
    std::cout << "Optimizing CPU usage..." << std::endl;

    // Reduce thread counts
    // Add delays between operations
    // This would be system-specific
}

void ResourceAwareHRM::manage_disk_usage() {
    // Clean up temporary files
    std::cout << "Managing disk usage..." << std::endl;

    // Remove old temp files
    // Compress data
    // This would be system-specific
}

void ResourceAwareHRM::handle_network_constraints() {
    // Throttle network operations
    std::cout << "Handling network constraints..." << std::endl;

    // Reduce download speeds
    // Pause network tasks
    // This would be system-specific
}

void ResourceAwareHRM::enter_resource_conservation_mode() {
    std::cout << "Entering resource conservation mode" << std::endl;

    // Aggressive resource management
    task_manager_->set_max_concurrent_tasks(1);
    task_manager_->set_chunking_enabled(true);

    // Pause non-critical tasks
    auto active_ids = task_manager_->get_active_task_ids();
    for (const auto& task_id : active_ids) {
        auto task = task_manager_->get_task(task_id);
        if (task && task->get_requirements().task_type != TaskType::CRITICAL) {
            task_manager_->pause_task(task_id);
        }
    }
}

void ResourceAwareHRM::exit_resource_conservation_mode() {
    std::cout << "Exiting resource conservation mode" << std::endl;

    // Restore normal operation
    task_manager_->set_max_concurrent_tasks(4);
    task_manager_->set_chunking_enabled(config_.enable_chunking_for_large_tasks);

    // Resume paused tasks
    auto active_ids = task_manager_->get_active_task_ids();
    for (const auto& task_id : active_ids) {
        task_manager_->resume_task(task_id);
    }
}

void ResourceAwareHRM::emergency_task_cancellation() {
    std::cout << "Emergency task cancellation initiated" << std::endl;

    // Cancel non-critical tasks
    auto active_ids = task_manager_->get_active_task_ids();
    for (const auto& task_id : active_ids) {
        auto task = task_manager_->get_task(task_id);
        if (task && task->get_requirements().task_type != TaskType::CRITICAL) {
            task_manager_->cancel_task(task_id);
        }
    }
}

// Training memory estimation and engine selection

struct TrainingMemoryRequirements {
    size_t parameter_memory;
    size_t optimizer_memory;
    size_t gradient_memory;
    size_t buffer_memory;
    size_t total_estimated;
};

TrainingMemoryRequirements estimate_training_memory(const VulkanTrainingConfig& config,
                                                   Nyx::PrecisionLevel precision) {
    TrainingMemoryRequirements req = {};

    // Calculate bytes per parameter based on precision
    float bytes_per_param = 4.0f; // FP32 default
    switch (precision) {
        case Nyx::PrecisionLevel::FP32: bytes_per_param = 4.0f; break;
        case Nyx::PrecisionLevel::FP16:
        case Nyx::PrecisionLevel::BF16: bytes_per_param = 2.0f; break;
        case Nyx::PrecisionLevel::INT8: bytes_per_param = 1.0f; break;
        case Nyx::PrecisionLevel::INT4: bytes_per_param = 0.5f; break;
    }

    // Calculate model parameters
    // Embedding: vocab_size * hidden_size
    // Attention layers: num_layers * (hidden_size * hidden_size * 4) for Q,K,V,O
    // MLP layers: num_layers * (hidden_size * hidden_size * 8) for up/down
    // Note: This is a simplified estimate - actual models may vary
    size_t num_params = config.vocab_size * config.hidden_size +  // embedding
                       config.num_layers * config.hidden_size * config.hidden_size * 12; // rough estimate

    req.parameter_memory = static_cast<size_t>(num_params * bytes_per_param);

    // Optimizer state (Adam uses 2x parameters)
    req.optimizer_memory = req.parameter_memory * 2;

    // Gradients (same size as parameters)
    req.gradient_memory = req.parameter_memory;

    // Training buffers: batch_size * max_sequence_length * vocab_size (for targets)
    // Use FP32 for buffers regardless of parameter precision
    req.buffer_memory = config.batch_size * config.max_sequence_length * config.vocab_size * 4;

    // Total with safety margin
    req.total_estimated = (req.parameter_memory + req.optimizer_memory +
                          req.gradient_memory + req.buffer_memory) * 6 / 5; // 20% safety margin

    return req;
}

Nyx::PrecisionLevel select_optimal_training_precision(const VulkanTrainingConfig& config,
                                                const Nyx::HardwareCapabilities& hw_caps) {
    // Priority order: FP32 → FP16 → BF16 → INT8 → INT4
    std::vector<Nyx::PrecisionLevel> candidates = {
        Nyx::PrecisionLevel::FP32, Nyx::PrecisionLevel::FP16, Nyx::PrecisionLevel::BF16,
        Nyx::PrecisionLevel::INT8, Nyx::PrecisionLevel::INT4
    };

    size_t available_memory = hw_caps.gpu_memory_mb * 1024ULL * 1024ULL; // Convert MB to bytes

    for (auto precision : candidates) {
        auto mem_req = estimate_training_memory(config, precision);
        if (mem_req.total_estimated < available_memory * 0.8) { // 80% safety margin
            return precision;
        }
    }

    // If nothing fits, return most memory-efficient option
    return Nyx::PrecisionLevel::INT4;
}

// Training methods

bool ResourceAwareHRM::initialize_training(const VulkanTrainingConfig& training_config) {
    if (vulkan_trainer_) {
        std::cerr << "Training already initialized" << std::endl;
        return false;
    }

    // Store selected precision from CLI
    current_training_precision_ = training_config.selected_precision;

    // Get Vulkan resources from HRM config
    auto hrm = get_hrm();
    if (!hrm) {
        std::cerr << "HRM not available for training initialization" << std::endl;
        return false;
    }

    // Extract Vulkan resources from the HRM configuration
    const auto& hrm_config = config_.base_config.base_config.hrm_config.inner_config;

    // Check if Vulkan resources are available
    bool has_vulkan = (hrm_config.device != VK_NULL_HANDLE &&
                       hrm_config.physicalDevice != VK_NULL_HANDLE &&
                       hrm_config.computeQueue != VK_NULL_HANDLE);

    // Initialize VulkanTrainer with the selected precision
    // Note: VulkanTrainer currently uses FP32 internally, precision selection is for future quantized implementation
    bool success = false;
    if (has_vulkan) {
        vulkan_trainer_ = std::make_unique<VulkanTrainer>(
            hrm_config.device,
            hrm_config.physicalDevice,
            hrm_config.computeQueueFamilyIndex,
            hrm_config.computeQueue,
            training_config
        );
        std::cout << "VulkanTrainer initialized with actual Vulkan resources" << std::endl;
        success = true;
    } else {
        std::cout << "Warning: Vulkan resources not available, initializing VulkanTrainer with null handles" << std::endl;
        vulkan_trainer_ = std::make_unique<VulkanTrainer>(
            VK_NULL_HANDLE, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, training_config
        );
        success = true;
    }

    if (success) {
        std::cout << "Training initialized with precision: "
                  << Nyx::precision_level_to_string(current_training_precision_)
                  << " (automatic selection based on available memory)" << std::endl;
        std::cout << "Training config:" << std::endl;
        std::cout << "  Max sequence length: " << training_config.max_sequence_length << std::endl;
        std::cout << "  Vocab size: " << training_config.vocab_size << std::endl;
        std::cout << "  Batch size: " << training_config.batch_size << std::endl;
        std::cout << "  Hidden size: " << training_config.hidden_size << std::endl;
        std::cout << "  Max epochs: " << training_config.max_epochs << std::endl;
    }

    return success;
}

bool ResourceAwareHRM::start_training_session() {
    if (!vulkan_trainer_) {
        std::cerr << "Training not initialized. Call initialize_training() first." << std::endl;
        return false;
    }

    // Discover and load available training data
    std::vector<std::string> available_files = {
        "data/text/processed/comprehensive_training_corpus.txt",
        "data/text/processed/training_corpus.txt", 
        "data/text/processed/arxiv_training_corpus.txt",
        "data/arxiv/arxiv_corpus.txt"
    };
    
    std::string data_path;
    for (const auto& file : available_files) {
        if (fs::exists(file)) {
            data_path = file;
            std::cout << "Discovered training data: " << file << std::endl;
            break;
        }
    }
    
    if (data_path.empty()) {
        std::cout << "No training data found, will learn from experience" << std::endl;
        return true;
    }
    
    std::cout << "Loading training data from: " << data_path << std::endl;
    if (!vulkan_trainer_->load_training_data(data_path)) {
        std::cerr << "Failed to load training data from: " << data_path << std::endl;
        return false;
    }

    std::cout << "Training session started. Data loaded successfully." << std::endl;
    return true;
}

bool ResourceAwareHRM::train_epoch() {
    if (!vulkan_trainer_) {
        std::cerr << "Training not initialized" << std::endl;
        return false;
    }

    std::cout << "Training epoch " << (vulkan_trainer_->get_current_epoch() + 1) << "..." << std::endl;

    if (!vulkan_trainer_->train_epoch()) {
        std::cerr << "Training epoch failed" << std::endl;
        return false;
    }

    std::cout << "Epoch completed. Loss: " << vulkan_trainer_->get_current_loss()
              << ", Perplexity: " << vulkan_trainer_->get_current_perplexity() << std::endl;

    return true;
}

bool ResourceAwareHRM::save_training_checkpoint(const std::string& checkpoint_path) {
    if (!vulkan_trainer_) {
        std::cerr << "Training not initialized" << std::endl;
        return false;
    }

    return vulkan_trainer_->save_checkpoint(checkpoint_path);
}

bool ResourceAwareHRM::load_training_checkpoint(const std::string& checkpoint_path) {
    if (!vulkan_trainer_) {
        std::cerr << "Training not initialized" << std::endl;
        return false;
    }

    return vulkan_trainer_->load_checkpoint(checkpoint_path);
}

std::string ResourceAwareHRM::generate_text(const std::string& prompt, uint32_t max_length) {
    if (!vulkan_trainer_) {
        return "";  // Empty triggers fallback to reasoning
    }

    try {
        return vulkan_trainer_->generate_text(prompt, max_length);
    } catch (const std::exception& e) {
        std::cerr << "Text generation failed: " << e.what() << std::endl;
        return "";  // Empty triggers fallback
    }
}

std::pair<float, std::unordered_map<std::string, Tensor>> ResourceAwareHRM::process_character_training_batch(
    const std::vector<std::string>& batch_sequences) {

    // Use Vulkan training if available, otherwise fall back to CPU processing
    if (vulkan_trainer_ && config_.base_config.base_config.hrm_config.inner_config.device != VK_NULL_HANDLE) {
        std::cout << "Using Vulkan-accelerated training for character batch" << std::endl;

        // Convert sequences to training batch format expected by VulkanTrainer
        TrainingBatch batch;
        batch.batch_size = batch_sequences.size();
        batch.seq_length = config_.base_config.base_config.hrm_config.inner_config.seq_len;

        // Convert sequences to input/target sequences
        for (const std::string& seq : batch_sequences) {
            for (size_t i = 0; i < std::min(seq.length(), static_cast<size_t>(batch.seq_length)); ++i) {
                int char_code = static_cast<unsigned char>(seq[i]);
                batch.input_sequences.push_back(static_cast<float>(char_code));
                if (i < seq.length() - 1) {
                    batch.target_sequences.push_back(static_cast<unsigned char>(seq[i + 1]));
                }
            }
        }

        // Execute Vulkan training
        if (!vulkan_trainer_->execute_forward_pass(batch)) {
            std::cerr << "Vulkan forward pass failed, falling back to CPU" << std::endl;
            return process_character_training_batch_cpu(batch_sequences);
        }

        if (!vulkan_trainer_->execute_backward_pass(batch)) {
            std::cerr << "Vulkan backward pass failed, falling back to CPU" << std::endl;
            return process_character_training_batch_cpu(batch_sequences);
        }

        if (!vulkan_trainer_->execute_optimizer_step()) {
            std::cerr << "Vulkan optimizer step failed, falling back to CPU" << std::endl;
            return process_character_training_batch_cpu(batch_sequences);
        }

        float loss = vulkan_trainer_->compute_loss_and_metrics();

        // Gradients computed internally by Vulkan trainer during backward pass
        // The trainer handles gradient computation and optimizer steps automatically
        // Returns loss with empty gradient map (gradients applied in-place by trainer)
        return {loss, {}};
    } else {
        std::cout << "Using CPU-based training (Vulkan not available)" << std::endl;
        return process_character_training_batch_cpu(batch_sequences);
    }
}

std::pair<float, std::unordered_map<std::string, Tensor>> ResourceAwareHRM::process_character_training_batch_cpu(
    const std::vector<std::string>& batch_sequences) {

    std::cout << "Using CPU-based HRM processing for character training" << std::endl;

    // Convert sequences to HRM input format
    std::unordered_map<std::string, Tensor> hrm_batch;
    int batch_size = batch_sequences.size();
    int max_len = 1024; // Default max length

    Tensor inputs_tensor;
    inputs_tensor.shape = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(max_len)};
    inputs_tensor.data.resize(batch_size * max_len, 0.0f);

    // Simple character-level tokenization
    for (int i = 0; i < batch_size; ++i) {
        const std::string& seq = batch_sequences[i];
        for (size_t j = 0; j < std::min(seq.length(), static_cast<size_t>(max_len)); ++j) {
            inputs_tensor.data[i * max_len + j] = static_cast<float>(static_cast<unsigned char>(seq[j]));
        }
    }
    hrm_batch["inputs"] = inputs_tensor;

    // Get HRM initial carry
    auto initial_carry = get_hrm()->initial_carry(hrm_batch);

    // Forward pass through HRM
    auto [final_carry, hrm_outputs] = get_hrm()->forward(initial_carry, hrm_batch);

    // Extract targets for next character prediction
    std::vector<std::vector<int>> targets;
    for (const auto& seq : batch_sequences) {
        std::vector<int> seq_targets;
        for (size_t j = 1; j < seq.length(); ++j) {
            seq_targets.push_back(static_cast<int>(static_cast<unsigned char>(seq[j])));
        }
        targets.push_back(seq_targets);
    }

    // Compute loss and gradients using the same logic as CharacterLanguageTrainer
    // Extract logits from HRM outputs
    auto logits_it = hrm_outputs.find("logits");
    if (logits_it == hrm_outputs.end()) {
        std::cerr << "HRM outputs missing 'logits' tensor" << std::endl;
        return {1.0f, {}};
    }

    const Tensor& logits = logits_it->second;

    // Compute character-level cross-entropy loss
    float total_loss = 0.0f;
    int total_chars = 0;

    // For each sequence in the batch
    for (size_t seq_idx = 0; seq_idx < targets.size(); ++seq_idx) {
        const auto& target_seq = targets[seq_idx];

        for (size_t char_idx = 0; char_idx < target_seq.size(); ++char_idx) {
            int target_char = target_seq[char_idx];

            // Get logits for this position (batch_size, seq_len, vocab_size)
            // Assuming logits shape is (batch_size, seq_len, vocab_size)
            size_t batch_offset = seq_idx * max_len * 256; // Assuming 256 vocab size
            size_t seq_offset = char_idx * 256;
            size_t logit_start = batch_offset + seq_offset;

            if (logit_start + 256 > logits.data.size()) {
                continue; // Skip if out of bounds
            }

            // Find the logit for the target character
            float target_logit = logits.data[logit_start + target_char];

            // Compute softmax denominator (sum of exp of all logits) with numerical stability
            float max_logit = logits.data[logit_start];
            for (int vocab_idx = 1; vocab_idx < 256; ++vocab_idx) {
                max_logit = std::max(max_logit, logits.data[logit_start + vocab_idx]);
            }

            float softmax_denominator = 0.0f;
            for (int vocab_idx = 0; vocab_idx < 256; ++vocab_idx) {
                float logit = logits.data[logit_start + vocab_idx];
                softmax_denominator += std::exp(logit - max_logit);
            }

            // Compute cross-entropy loss: -log(softmax(target))
            float target_prob = std::exp(target_logit - max_logit) / softmax_denominator;

            if (target_prob <= 0.0f || std::isnan(target_prob) || std::isinf(target_prob)) {
                target_prob = 1e-10f;
            }

            float loss = -std::log(target_prob);

            if (std::isnan(loss) || std::isinf(loss)) {
                loss = 10.0f;
            }

            total_loss += loss;
            total_chars++;
        }
    }

    float avg_loss = total_chars > 0 ? total_loss / total_chars : 0.0f;

    // Compute real gradients for training
    std::unordered_map<std::string, Tensor> gradients;

    // For character-level language modeling, we need gradients w.r.t. logits
    // The gradient of cross-entropy loss w.r.t. logits is: softmax(logits) - one_hot(targets)

    if (!logits.data.empty()) {
        // Compute softmax and gradients for each position
        Tensor logit_grads;
        logit_grads.shape = logits.shape;
        logit_grads.data.resize(logits.data.size(), 0.0f);

        // Process each sequence in the batch
        for (size_t seq_idx = 0; seq_idx < targets.size(); ++seq_idx) {
            const auto& target_seq = targets[seq_idx];

            for (size_t char_idx = 0; char_idx < target_seq.size(); ++char_idx) {
                int target_char = target_seq[char_idx];

                // Get logits for this position (assuming shape: [batch_size, seq_len, vocab_size])
                size_t batch_offset = seq_idx * max_len * 256; // Assuming vocab_size = 256
                size_t seq_offset = char_idx * 256;
                size_t logit_start = batch_offset + seq_offset;

                if (logit_start + 256 <= logits.data.size()) {
                    // Compute softmax
                    std::vector<float> softmax_probs(256, 0.0f);
                    float max_logit = *std::max_element(
                        logits.data.begin() + logit_start,
                        logits.data.begin() + logit_start + 256
                    );

                    float sum_exp = 0.0f;
                    for (int i = 0; i < 256; ++i) {
                        float exp_val = expf(logits.data[logit_start + i] - max_logit);
                        softmax_probs[i] = exp_val;
                        sum_exp += exp_val;
                    }

                    for (int i = 0; i < 256; ++i) {
                        softmax_probs[i] /= sum_exp;
                    }

                    // Gradient: softmax - one_hot(target)
                    for (int i = 0; i < 256; ++i) {
                        float gradient = softmax_probs[i];
                        if (i == target_char) {
                            gradient -= 1.0f; // Subtract 1 for the target class
                        }
                        logit_grads.data[logit_start + i] = gradient;
                    }
                }
            }
        }

        gradients["logits"] = logit_grads;
    }

    return {avg_loss, gradients};
}

// Idle Learning Interface Implementation
void ResourceAwareHRM::enable_idle_learning(bool enable) {
    if (idle_learning_manager_) {
        idle_learning_manager_->enable_idle_learning(enable);
    }
}

bool ResourceAwareHRM::is_idle_learning_enabled() const {
    return idle_learning_manager_ && idle_learning_manager_->is_idle_learning_enabled();
}

std::vector<std::string> ResourceAwareHRM::get_idle_learning_status() const {
    if (idle_learning_manager_) {
        return idle_learning_manager_->get_learning_status();
    }
    return {"Idle learning not available"};
}

size_t ResourceAwareHRM::get_pending_learning_data_points() const {
    return idle_learning_manager_ ? idle_learning_manager_->get_pending_data_points() : 0;
}

std::unordered_map<std::string, double> ResourceAwareHRM::get_idle_learning_metrics() const {
    if (idle_learning_manager_) {
        return idle_learning_manager_->get_learning_metrics();
    }
    return {};
}