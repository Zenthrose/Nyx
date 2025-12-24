#pragma once

#include "../self_mod/self_modifying_hrm.hpp"
#include "../system/resource_monitor.hpp"
#include "../system/task_manager.hpp"
#include "../vulkan/vulkan_trainer.hpp"
#include "../training/int4_training_engine.hpp"
#include "../vulkan/quantization_types.hpp"
#include "idle_learning_manager.hpp"

struct ResourceAwareHRMConfig {
    SelfModifyingHRMConfig base_config;
    bool enable_resource_monitoring;
    bool enable_adaptive_task_management;
    bool enable_chunking_for_large_tasks;
    std::chrono::milliseconds resource_check_interval;
    uint64_t max_memory_per_task_mb;
    double max_cpu_per_task_percent;
    std::shared_ptr<class MemoryCompactionSystem> memory_compaction_system;
    std::shared_ptr<class CloudStorageManager> cloud_storage_manager;
};

struct ResourceAwareTask {
    std::string task_id;
    std::string description;
    TaskPriority priority;
    TaskRequirements requirements;
    std::function<TaskResult(const std::vector<TaskChunk>&)> executor;
    bool submitted;
    TaskResult result;
};

class ResourceAwareHRM : public SelfModifyingHRM {
public:
    // Singleton pattern to prevent duplicate HRM instances
    static std::shared_ptr<ResourceAwareHRM> getInstance(const ResourceAwareHRMConfig& config = ResourceAwareHRMConfig{});
    static void destroyInstance();

    ~ResourceAwareHRM();

private:
    ResourceAwareHRM(const ResourceAwareHRMConfig& config);

public:
    // Enhanced communication with resource awareness
    CommunicationResult communicate(const std::string& input_message);

    // Resource-aware task management
    std::string submit_resource_aware_task(const std::string& description,
                                         TaskPriority priority,
                                         const TaskRequirements& requirements,
                                         std::function<TaskResult(const std::vector<TaskChunk>&)> executor);

    bool pause_task_due_to_resources(const std::string& task_id);
    bool resume_task_when_resources_available(const std::string& task_id);

    // Resource monitoring interface
    ResourceUsage get_current_resource_usage() const;
    std::vector<ResourceAlert> get_resource_alerts() const;
    bool are_resources_available(const TaskRequirements& requirements) const;

    // Memory compaction interface
    std::unordered_map<std::string, std::string> get_memory_compaction_stats() const;
    bool perform_memory_compaction();
    std::vector<std::string> list_memory_compactions() const;

    // Cloud storage interface
    std::unordered_map<std::string, std::string> get_cloud_storage_stats() const;
    bool upload_to_cloud(const std::string& data_id);
    bool download_from_cloud(const std::string& data_id);
    std::vector<std::string> list_cloud_storage() const;

    // CPU/RAM offloading interface
    bool should_offload_to_cpu(const TaskRequirements& requirements) const;
    void enable_hybrid_execution(bool enable);
    bool is_hybrid_execution_enabled() const;
    std::unordered_map<std::string, std::string> get_offloading_stats() const;

    // Adaptive behavior
    void adapt_to_resource_constraints();
    void optimize_for_current_resources();
    std::vector<std::string> get_resource_optimization_suggestions();

    // System status with resource information
    std::unordered_map<std::string, std::string> get_resource_aware_status();

    // Training capabilities
    bool initialize_training(const VulkanTrainingConfig& training_config);
    bool start_training_session();
    bool train_epoch();
    bool save_training_checkpoint(const std::string& checkpoint_path);
    bool load_training_checkpoint(const std::string& checkpoint_path);

    // Training status
    bool is_training_initialized() const { return vulkan_trainer_ != nullptr; }
    float get_training_loss() const {
        return vulkan_trainer_ ? vulkan_trainer_->get_current_loss() : 0.0f;
    }
    float get_training_perplexity() const {
        return vulkan_trainer_ ? vulkan_trainer_->get_current_perplexity() : 0.0f;
    }
    uint32_t get_current_training_epoch() const {
        return vulkan_trainer_ ? vulkan_trainer_->get_current_epoch() : 0;
    }

    // Batch processing for character language training
    std::pair<float, std::unordered_map<std::string, Tensor>> process_character_training_batch(
        const std::vector<std::string>& batch_sequences);

    // Access to core HRM model
    HRM* get_hrm() { return SelfEvolvingHRM::get_hrm(); }

    // Idle learning interface
    void enable_idle_learning(bool enable);
    bool is_idle_learning_enabled() const;
    std::vector<std::string> get_idle_learning_status() const;
    size_t get_pending_learning_data_points() const;
    std::unordered_map<std::string, double> get_idle_learning_metrics() const;

private:
    // Text generation (override)
    std::string generate_text(const std::string& prompt, uint32_t max_length = 100) override;
    // CPU fallback for character training batch processing
    std::pair<float, std::unordered_map<std::string, Tensor>> process_character_training_batch_cpu(
        const std::vector<std::string>& batch_sequences);
    ResourceAwareHRMConfig config_;
    std::shared_ptr<ResourceMonitor> resource_monitor_;
    std::shared_ptr<TaskManager> task_manager_;
    std::shared_ptr<class MemoryCompactionSystem> memory_compaction_system_;
    std::shared_ptr<class CloudStorageManager> cloud_storage_manager_;
    std::unique_ptr<VulkanTrainer> vulkan_trainer_;
    Nyx::PrecisionLevel current_training_precision_ = Nyx::PrecisionLevel::FP32;
    std::shared_ptr<class CharacterSequenceCache> character_cache_;
    bool hybrid_execution_enabled_;
    std::unordered_map<std::string, size_t> offloading_stats_;

    // Idle learning system
    std::shared_ptr<IdleTimeRepairScheduler> idle_repair_scheduler_;
    std::unique_ptr<IdleLearningManager> idle_learning_manager_;

    // Resource-aware state
    std::vector<ResourceAwareTask> pending_tasks_;
    std::unordered_map<std::string, ResourceAwareTask> active_tasks_;
    bool resource_pressure_mode_;

    // Resource-aware methods
    void initialize_resource_monitoring();
    void handle_resource_alerts();
    void adapt_task_execution_to_resources();
    void implement_resource_aware_chunking(const std::string& task_id);

    // Task lifecycle with resource awareness
    bool can_execute_task_now(const ResourceAwareTask& task) const;
    void submit_pending_tasks();
    void manage_task_priorities_based_on_resources();

    // Resource optimization
    void reduce_memory_usage();
    void optimize_cpu_usage();
    void manage_disk_usage();
    void handle_network_constraints();

    // Emergency resource management
    void enter_resource_conservation_mode();
    void exit_resource_conservation_mode();
    void emergency_task_cancellation();
};