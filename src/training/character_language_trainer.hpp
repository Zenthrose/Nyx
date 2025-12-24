#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <atomic>
#include "../hrm/resource_aware_hrm.hpp"
#include "character_language_config.hpp"
#include "character_text_dataset.hpp"
#include "character_language_loss.hpp"
#include "character_language_evaluator.hpp"
#include "advanced_training_optimizations.hpp"
#include "character_language_config.hpp"

/**
 * Progressive Data Feeder for Curriculum Learning
 *
 * Manages stage-based data loading and feeding to prevent memory overload
 * and enable natural learning progression.
 */
class ProgressiveDataFeeder {
public:
    ProgressiveDataFeeder(const ProgressiveDataFeederConfig& config);
    ~ProgressiveDataFeeder();

    /**
     * Load data for a specific curriculum stage
     * @param stage_id The curriculum stage to load data for
     * @return Vector of training sequences for this stage
     */
    std::vector<std::string> load_stage_data(int stage_id);

    /**
     * Feed next batch of data based on current learning progress
     * @param current_stage Current curriculum stage
     * @param learning_metrics Current learning performance metrics
     * @return Next batch of sequences to train on
     */
    std::vector<std::string> feed_next_batch(int current_stage,
                                           const std::unordered_map<std::string, float>& learning_metrics);

    /**
     * Check if current stage data is exhausted
     * @param stage_id Stage to check
     * @return True if all stage data has been fed
     */
    bool is_stage_data_exhausted(int stage_id) const;

    /**
     * Get memory usage statistics
     * @return Memory usage information
     */
    std::unordered_map<std::string, size_t> get_memory_stats() const;

    /**
     * Clean up data from completed stages to free memory
     * @param completed_stage Stage that was just finished
     */
    void cleanup_completed_stage(int completed_stage);

private:
    ProgressiveDataFeederConfig config_;
    std::unordered_map<int, std::vector<std::string>> loaded_stage_data_;
    std::unordered_map<int, size_t> stage_feed_position_;
    std::unordered_map<int, bool> stage_data_exhausted_;

    /**
     * Load data segments for a stage from the corpus
     * @param stage_config Configuration for the stage
     * @return Loaded data segments
     */
    std::vector<std::string> load_data_segments(const DataStageConfig& stage_config);

    /**
     * Apply memory-aware data subsetting
     * @param full_data Complete stage data
     * @param stage_config Stage configuration
     * @return Memory-safe subset
     */
    std::vector<std::string> apply_memory_constraints(
        const std::vector<std::string>& full_data,
        const DataStageConfig& stage_config);
};

/**
 * Character-Level Language Trainer for HRM
 *
 * Integrates character-level language training with the HRM system,
 * enabling the AI to learn conversational abilities, reasoning skills,
 * and text generation capabilities.
 */
class CharacterLanguageTrainer {
public:
    CharacterLanguageTrainer(std::shared_ptr<ResourceAwareHRM> hrm_system,
                           const CharacterLanguageModelConfig& config);

    ~CharacterLanguageTrainer();

    /**
     * Train the HRM on character-level language tasks
     * @param dataset_path Path to the training dataset
     * @return Training statistics and final model performance
     */
    std::unordered_map<std::string, float> train_character_language_model(
        const std::string& dataset_path);

    /**
     * Fine-tune the model on a specific task
     * @param task_data_path Path to task-specific training data
     * @return Fine-tuning results
     */
    std::unordered_map<std::string, float> fine_tune_on_task(
        const std::string& task_data_path);

    /**
     * Generate text using the trained model
     * @param prompt Initial prompt text
     * @param max_length Maximum generation length
     * @return Generated text
     */
    std::string generate_text(const std::string& prompt, int max_length = 500);

    /**
     * Evaluate model performance on test data
     * @param test_data_path Path to test dataset
     * @return Evaluation metrics
     */
    std::unordered_map<std::string, float> evaluate_model(
        const std::string& test_data_path);

    /**
     * Save model checkpoint
     * @param checkpoint_path Path to save checkpoint
     * @return Success status
     */
    bool save_checkpoint(const std::string& checkpoint_path);

    /**
     * Load model checkpoint
     * @param checkpoint_path Path to load checkpoint
     * @return Success status
     */
    bool load_checkpoint(const std::string& checkpoint_path);

    /**
     * Get current training statistics
     * @return Training metrics
     */
    std::unordered_map<std::string, float> get_training_stats() const;

    /**
     * Stop training gracefully
     */
    void stop_training();

private:
    std::shared_ptr<ResourceAwareHRM> hrm_system_;
    CharacterLanguageModelConfig config_;

    // Training components
    std::unique_ptr<CharacterTextDataset> dataset_;
    std::unique_ptr<ProgressiveDataFeeder> data_feeder_;
    std::unique_ptr<CharacterLanguageLoss> loss_calculator_;
    std::unique_ptr<CharacterLanguageEvaluator> evaluator_;

    // Training state
    std::atomic<bool> training_active_;
    std::atomic<bool> should_stop_;
    int current_epoch_;
    int global_step_;
    float best_loss_;
    int epochs_without_improvement_;
    std::chrono::steady_clock::time_point training_start_time_;

    // Training statistics
    std::vector<float> epoch_losses_;
    std::vector<float> epoch_perplexities_;
    std::vector<float> epoch_accuracies_;
    std::vector<float> learning_rates_;

    /**
     * Initialize training components
     */
    void initialize_training_components();

    /**
     * Training loop for one epoch
     * @param train_sequences Training data sequences
     * @return Epoch training metrics
     */
    std::unordered_map<std::string, float> train_epoch(
        const std::vector<std::string>& train_sequences);

    /**
     * Validation loop
     * @param val_sequences Validation data sequences
     * @return Validation metrics
     */
    std::unordered_map<std::string, float> validate(
        const std::vector<std::string>& val_sequences);

/**
     * Process a single training batch
     * @param batch_sequences Character sequences in batch
     * @return Batch loss and gradients
     */
    std::pair<float, std::unordered_map<std::string, Tensor>> process_training_batch(
        const std::vector<std::string>& batch_sequences);

    /**
     * Process a single training batch with reused carry (prevents memory leak)
     * @param batch_sequences Character sequences in batch
     * @param reused_carry Reused HRM carry state
     * @return Batch loss and gradients
     */
    std::pair<float, std::unordered_map<std::string, Tensor>> process_training_batch_with_carry(
        const std::vector<std::string>& batch_sequences, const HRMCarry& reused_carry);

    /**
     * Convert character sequences to HRM input format
     * @param sequences Character sequences
     * @return HRM-compatible batch data
     */
    std::unordered_map<std::string, Tensor> sequences_to_hrm_batch(
        const std::vector<std::string>& sequences);

    /**
     * Extract next-character targets from sequences
     * @param sequences Input sequences
     * @return Target character indices
     */
    std::vector<std::vector<int>> extract_targets(
        const std::vector<std::string>& sequences);

    /**
     * Compute character-level loss and gradients
     * @param hrm_outputs HRM model outputs
     * @param targets Target character indices
     * @return Loss value and gradients
     */
    std::pair<float, std::unordered_map<std::string, Tensor>> compute_loss_and_gradients(
        const std::unordered_map<std::string, Tensor>& hrm_outputs,
        const std::vector<std::vector<int>>& targets);

    /**
     * Update model parameters using gradients
     * @param gradients Parameter gradients
     * @param learning_rate Current learning rate
     */
    void update_parameters(const std::unordered_map<std::string, Tensor>& gradients,
                          float learning_rate);

    /**
     * Compute current learning rate based on schedule
     * @param step Current training step
     * @return Learning rate
     */
    float compute_learning_rate(size_t step) const;

    /**
     * Apply learning rate scheduler
     * @param step Current step
     * @return Scheduled learning rate
     */
    float apply_lr_scheduler(size_t step) const;

    /**
     * Log training progress
     * @param epoch Current epoch
     * @param train_metrics Training metrics
     * @param val_metrics Validation metrics
     */
    void log_training_progress(int epoch, const std::unordered_map<std::string, float>& train_metrics,
                              const std::unordered_map<std::string, float>& val_metrics);

    /**
     * Check if training should stop early
     * @param current_loss Current validation loss
     * @return True if should stop
     */
    bool should_early_stop(float current_loss);

    /**
     * Save training statistics to file
     * @param stats_path Path to save statistics
     */
    void save_training_stats(const std::string& stats_path) const;

    /**
     * Save epoch results to text file
     * @param epoch Epoch number
     * @param train_metrics Training metrics
     * @param val_metrics Validation metrics
     */
    void save_epoch_results(int epoch, 
                          const std::unordered_map<std::string, float>& train_metrics,
                          const std::unordered_map<std::string, float>& val_metrics);

    /**
     * Generate intelligent contexts from training data
     * @param sequences Raw training sequences
     * @return Intelligent context sequences
     */
    std::vector<std::string> generate_intelligent_contexts(const std::vector<std::string>& sequences);

    /**
     * Extract semantic chunks from text
     * @param text Input text
     * @return Semantic chunks
     */
    std::vector<std::string> extract_semantic_chunks(const std::string& text);

    /**
     * Generate meta-contexts for deeper understanding
     * @param contexts Base contexts
     * @return Meta-contexts
     */
    std::vector<std::string> generate_meta_contexts(const std::vector<std::string>& contexts);

    /**
     * Load training data from file
     * @param data_path Path to training data
     * @return Loaded sequences
     */
    std::vector<std::string> load_training_data(const std::string& data_path, float data_percentage = 1.0f);
    void extract_sequences_from_text(const std::string& content, std::vector<std::string>& sequences, float data_percentage);
    std::vector<std::string> scan_directory_async(const std::string& base_dir);
    std::vector<std::string> generate_system_learning_sequences();

    /**
     * Calculate realistic batch accuracy based on loss
     * @param batch_sequences Sequences in the batch
     * @param batch_loss Batch loss value
     * @return Accuracy percentage (0-100)
     */
    float calculate_batch_accuracy(const std::vector<std::string>& batch_sequences, float batch_loss);
};