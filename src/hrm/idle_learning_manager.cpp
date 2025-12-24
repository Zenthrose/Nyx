#include "idle_learning_manager.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "../utils/logger.hpp"

IdleLearningManager::IdleLearningManager(std::shared_ptr<IdleTimeRepairScheduler> idle_scheduler,
                                       std::shared_ptr<ResourceMonitor> resource_monitor)
    : idle_scheduler_(idle_scheduler), resource_monitor_(resource_monitor),
      max_data_points_(10000), learning_interval_(std::chrono::hours(4)),
      learning_enabled_(true), max_cpu_percent_(20.0), max_memory_mb_(512),
      session_in_progress_(false), state_file_path_("idle_learning_state.json") {

    // Initialize learning state
    learning_state_.total_learning_sessions = 0;
    learning_state_.total_data_points_processed = 0;
    learning_state_.last_learning_session = std::chrono::system_clock::now();
    learning_state_.average_learning_duration_seconds = 0.0;
    learning_state_.learning_active = false;
    learning_state_.insights_shared_from_communication = 0;
    learning_state_.last_coordination = std::chrono::system_clock::now();

    // Load previous state if exists
    load_learning_state(state_file_path_);

    // Schedule initial learning task if enabled and scheduler is available
    if (learning_enabled_ && idle_scheduler_) {
        schedule_learning_task();
    }

    auto& logger = Logger::getInstance();
    logger.info("Idle Learning Manager initialized with " +
               std::to_string(accumulated_data_.size()) + " queued data points");
}

IdleLearningManager::~IdleLearningManager() {
    // Save state before destruction
    save_learning_state(state_file_path_);
}

void IdleLearningManager::accumulate_learning_data(const LearningDataPoint& data_point) {
    if (!learning_enabled_) return;

    std::lock_guard<std::mutex> lock(data_mutex_);
    accumulated_data_.push(data_point);

    // Cleanup old data if we exceed limit
    if (accumulated_data_.size() > max_data_points_) {
        cleanup_old_data();
    }

    // Compress data periodically to save memory
    if (accumulated_data_.size() % 1000 == 0) {
        compress_data_if_needed();
    }
}

void IdleLearningManager::accumulate_conversation_data(const std::string& input,
                                                     const std::string& response,
                                                     float confidence) {
    LearningDataPoint data_point;
    data_point.input_text = input;
    data_point.response_text = response;
    data_point.confidence_score = confidence;
    data_point.timestamp = std::chrono::system_clock::now();
    data_point.metadata["type"] = "conversation";
    data_point.metadata["input_length"] = std::to_string(input.length());
    data_point.metadata["response_length"] = std::to_string(response.length());

    accumulate_learning_data(data_point);
}

void IdleLearningManager::enable_idle_learning(bool enable) {
    learning_enabled_ = enable;

    if (enable) {
        schedule_learning_task();
        auto& logger = Logger::getInstance();
        logger.info("Idle learning enabled");
    } else {
        // Note: We don't cancel existing scheduled tasks, they will just check the flag
        auto& logger = Logger::getInstance();
        logger.info("Idle learning disabled");
    }
}

bool IdleLearningManager::is_idle_learning_enabled() const {
    return learning_enabled_;
}

void IdleLearningManager::set_learning_parameters(size_t max_data_points,
                                                std::chrono::hours learning_interval) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    max_data_points_ = max_data_points;
    learning_interval_ = learning_interval;

    // Cleanup if new limit is lower
    while (accumulated_data_.size() > max_data_points_) {
        accumulated_data_.pop();
    }
}

bool IdleLearningManager::perform_idle_learning_session() {
    if (!learning_enabled_ || session_in_progress_) {
        return false;
    }

    // Check if enough time has passed since last session
    auto now = std::chrono::system_clock::now();
    auto time_since_last = now - last_session_time_;
    if (time_since_last < learning_interval_) {
        return false;  // Too soon for another session
    }

    // Check resource availability
    if (!check_resource_availability()) {
        return false;
    }

    session_in_progress_ = true;
    auto session_start = std::chrono::high_resolution_clock::now();

    bool success = false;
    try {
        // Mark learning as active
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            learning_state_.learning_active = true;
        }

        auto& logger = Logger::getInstance();
        logger.info("Starting idle learning session");

        // Coordinate with communication learning if needed
        if (should_coordinate_now()) {
            coordinate_with_communication_learning();
        }

        // Process accumulated data
        success = process_accumulated_data();

        if (success) {
            // Only update learning models if we have data
            if (!accumulated_data_.empty()) {
                success = update_learning_models();
            }
            // If no data, consider it successful (nothing to do)
        }

        // Update session time
        last_session_time_ = now;

        auto session_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(session_end - session_start);

        // Update metrics
        update_learning_metrics(duration, success);

        // Save state
        save_learning_state(state_file_path_);

        logger.info("Idle learning session completed in " + std::to_string(duration.count()) + "ms");

    } catch (const std::exception& e) {
        auto& logger = Logger::getInstance();
        logger.error("Idle learning session failed: " + std::string(e.what()));
        success = false;
    }

    // Mark learning as inactive
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        learning_state_.learning_active = false;
    }

    session_in_progress_ = false;
    return success;
}

bool IdleLearningManager::process_accumulated_data() {
    if (accumulated_data_.empty()) {
        return true;  // Nothing to process, consider success
    }

    size_t processed_count = 0;
    const size_t batch_size = 50;  // Process in small batches

    while (!accumulated_data_.empty() && check_resource_availability()) {
        auto batch = get_batch_for_processing(batch_size);

        if (batch.empty()) break;

        // Process batch (placeholder - would integrate with actual learning logic)
        for (const auto& data_point : batch) {
            // Here would be the actual learning processing
            // For now, just count and log
            processed_count++;
        }

        // Small delay to prevent overwhelming system
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Update processed count
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        learning_state_.total_data_points_processed += processed_count;
    }

    auto& logger = Logger::getInstance();
    logger.info("Processed " + std::to_string(processed_count) + " data points in idle learning session");

    return true;
}

bool IdleLearningManager::update_learning_models() {
    // Update learning models with accumulated idle time data
    auto& logger = Logger::getInstance();
    logger.info("Updating learning models with accumulated data");

    try {
        if (accumulated_data_.empty()) {
            logger.info("Insufficient data for model update (queue empty)");
            return true;
        }

        int update_count = 0;
        std::vector<std::string> batch_texts;
        
        // Collect text data from accumulated learning data (queue is FIFO)
        std::queue<LearningDataPoint> temp_queue = accumulated_data_;
        while (!temp_queue.empty()) {
            const auto& data = temp_queue.front();
            // Collect both input and response text for training
            if (!data.input_text.empty()) {
                batch_texts.push_back(data.input_text);
                update_count++;
            }
            if (!data.response_text.empty()) {
                batch_texts.push_back(data.response_text);
                update_count++;
            }
            temp_queue.pop();
        }

        // Train language model on new patterns if sufficient data
        if (!batch_texts.empty() && batch_texts.size() > 5) {
            logger.info("Training on " + std::to_string(batch_texts.size()) + " text samples");
            // Model update would occur here with trainer integration
            update_count += batch_texts.size();
        }

        // Clear accumulated data after processing
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            std::queue<LearningDataPoint> empty;
            std::swap(accumulated_data_, empty);
        }

        logger.info("Completed model updates (processed " + std::to_string(update_count) + " items)");
        return true;
    } catch (const std::exception& e) {
        logger.error("Error updating learning models: " + std::string(e.what()));
        return false;
    }

    return true;
}

bool IdleLearningManager::save_learning_state(const std::string& state_file) {
    try {
        std::lock_guard<std::mutex> lock(state_mutex_);
        std::ofstream file(state_file);
        if (!file.is_open()) return false;

        // Save basic state as text
        file << learning_state_.total_learning_sessions << "\n";
        file << learning_state_.total_data_points_processed << "\n";
        file << std::chrono::duration_cast<std::chrono::seconds>(
            learning_state_.last_learning_session.time_since_epoch()).count() << "\n";
        file << learning_state_.average_learning_duration_seconds << "\n";
        file << learning_state_.learning_metrics.size() << "\n";

        // Save metrics
        for (const auto& [key, value] : learning_state_.learning_metrics) {
            file << key << "=" << value << "\n";
        }

        // Save queued data count
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            file << accumulated_data_.size() << "\n";
        }

        file.close();
        return true;
    } catch (const std::exception& e) {
        auto& logger = Logger::getInstance();
        logger.error("Failed to save learning state: " + std::string(e.what()));
        return false;
    }
}

bool IdleLearningManager::load_learning_state(const std::string& state_file) {
    try {
        std::ifstream file(state_file);
        if (!file.is_open()) {
            return false;  // No existing state file
        }

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Load basic state from text
        std::string line;
        if (std::getline(file, line)) learning_state_.total_learning_sessions = std::stoull(line);
        if (std::getline(file, line)) learning_state_.total_data_points_processed = std::stoull(line);

        if (std::getline(file, line)) {
            auto last_session_seconds = std::stoll(line);
            learning_state_.last_learning_session = std::chrono::system_clock::time_point(
                std::chrono::seconds(last_session_seconds));
        }

        if (std::getline(file, line)) learning_state_.average_learning_duration_seconds = std::stod(line);

        // Load metrics count and metrics
        size_t metrics_count = 0;
        if (std::getline(file, line)) metrics_count = std::stoull(line);

        for (size_t i = 0; i < metrics_count && std::getline(file, line); ++i) {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                learning_state_.learning_metrics[key] = value;
            }
        }

        file.close();
        return true;
    } catch (const std::exception& e) {
        auto& logger = Logger::getInstance();
        logger.error("Failed to load learning state: " + std::string(e.what()));
        return false;
    }
}

IdleLearningState IdleLearningManager::get_learning_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return learning_state_;
}

std::vector<std::string> IdleLearningManager::get_learning_status() const {
    std::vector<std::string> status;

    auto state = get_learning_state();
    status.push_back("Idle learning: " + std::string(is_idle_learning_enabled() ? "enabled" : "disabled"));
    status.push_back("Learning sessions: " + std::to_string(state.total_learning_sessions));
    status.push_back("Data points processed: " + std::to_string(state.total_data_points_processed));
    status.push_back("Queued data points: " + std::to_string(get_pending_data_points()));

    if (state.learning_active) {
        status.push_back("Status: Learning session in progress");
    } else {
        auto time_since_last = std::chrono::system_clock::now() - state.last_learning_session;
        auto hours_since = std::chrono::duration_cast<std::chrono::hours>(time_since_last).count();
        status.push_back("Last session: " + std::to_string(hours_since) + " hours ago");
    }

    return status;
}

size_t IdleLearningManager::get_pending_data_points() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return accumulated_data_.size();
}

std::unordered_map<std::string, double> IdleLearningManager::get_learning_metrics() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::unordered_map<std::string, double> metrics;

    for (const auto& [key, value] : learning_state_.learning_metrics) {
        try {
            metrics[key] = std::stod(value);
        } catch (...) {
            metrics[key] = 0.0;
        }
    }

    return metrics;
}

void IdleLearningManager::set_resource_limits(double max_cpu_percent, size_t max_memory_mb) {
    max_cpu_percent_ = max_cpu_percent;
    max_memory_mb_ = max_memory_mb;
}

bool IdleLearningManager::check_resource_availability() const {
    if (!resource_monitor_) return true;  // Assume available if no monitor

    auto usage = resource_monitor_->get_current_usage();

    if (usage.cpu_usage_percent > max_cpu_percent_) {
        return false;
    }

    // Convert bytes to MB
    size_t used_memory_mb = usage.used_memory_bytes / (1024 * 1024);
    if (used_memory_mb > max_memory_mb_) {
        return false;
    }

    return true;
}

void IdleLearningManager::schedule_learning_task() {
    if (!idle_scheduler_) {
        auto& logger = Logger::getInstance();
        logger.warning("Idle scheduler not available - learning tasks will not be scheduled");
        return;
    }

    auto learning_task = [this]() -> bool {
        return perform_idle_learning_session();
    };

    // Schedule as low-priority task
    idle_scheduler_->schedule_repair_task(
        "Idle Learning Session",
        RepairPriority::LOW,
        learning_task,
        std::chrono::minutes(30),  // Estimated duration
        false  // No restart required
    );
}

bool IdleLearningManager::validate_learning_conditions() const {
    return learning_enabled_ && check_resource_availability() && !session_in_progress_;
}

void IdleLearningManager::update_learning_metrics(const std::chrono::milliseconds& duration, bool success) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    learning_state_.total_learning_sessions++;

    // Update average duration
    double session_duration_seconds = duration.count() / 1000.0;
    learning_state_.average_learning_duration_seconds =
        (learning_state_.average_learning_duration_seconds * (learning_state_.total_learning_sessions - 1) +
         session_duration_seconds) / learning_state_.total_learning_sessions;

    // Update metrics
    learning_state_.learning_metrics["last_session_duration_seconds"] = std::to_string(session_duration_seconds);
    learning_state_.learning_metrics["success_rate"] = std::to_string(
        (learning_state_.learning_metrics.count("success_count") ?
         std::stod(learning_state_.learning_metrics["success_count"]) : 0.0 + (success ? 1.0 : 0.0)) /
        learning_state_.total_learning_sessions);

    if (success) {
        learning_state_.learning_metrics["success_count"] =
            std::to_string(std::stod(learning_state_.learning_metrics["success_count"]) + 1.0);
    }
}

void IdleLearningManager::cleanup_old_data() {
    // Remove oldest data points when we exceed limit
    while (accumulated_data_.size() > max_data_points_) {
        accumulated_data_.pop();
    }
}

bool IdleLearningManager::compress_data_if_needed() {
    // Placeholder for data compression logic
    // In a real implementation, this might compress old data or summarize patterns
    return true;
}

std::vector<LearningDataPoint> IdleLearningManager::get_batch_for_processing(size_t batch_size) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<LearningDataPoint> batch;
    size_t count = 0;

    while (!accumulated_data_.empty() && count < batch_size) {
        batch.push_back(accumulated_data_.front());
        accumulated_data_.pop();
        count++;
    }

    return batch;
}

void IdleLearningManager::log_learning_activity(const std::string& activity, bool success) {
    auto& logger = Logger::getInstance();
    std::string message = "Idle Learning: " + activity;
    if (success) {
        logger.info(message);
    } else {
        logger.error(message + " - failed");
    }
}

// Coordination with communication learning
void IdleLearningManager::coordinate_with_communication_learning() {
    // Basic coordination: track that coordination occurred
    // In a full implementation, this would exchange insights between learning modes
    std::lock_guard<std::mutex> lock(state_mutex_);
    learning_state_.last_coordination = std::chrono::system_clock::now();

    // For now, just mark that coordination happened
    // Future: exchange model parameters, insights, or training data
    learning_state_.insights_shared_from_communication++;

    auto& logger = Logger::getInstance();
    logger.info("Coordinated with communication learning (insights: " +
               std::to_string(learning_state_.insights_shared_from_communication) + ")");
}

bool IdleLearningManager::should_coordinate_now() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto now = std::chrono::system_clock::now();
    auto time_since_coordination = now - learning_state_.last_coordination;

    // Coordinate every few learning sessions or every few hours
    return (learning_state_.total_learning_sessions % 3 == 0) ||
           (time_since_coordination > std::chrono::hours(2));
}

uint64_t IdleLearningManager::get_shared_insights_count() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return learning_state_.insights_shared_from_communication;
}