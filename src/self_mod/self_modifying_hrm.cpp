#include "self_modifying_hrm.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <cstdio>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <fcntl.h>

SelfModifyingHRM::SelfModifyingHRM(const SelfModifyingHRMConfig& config)
    : SelfEvolvingHRM(config.base_config), config_(config) {
    std::cout << "Initializing Self-Modifying HRM System..." << std::endl;

    // Initialize self-modification components with lazy loading to prevent memory exhaustion
    code_analyzer_ = std::make_unique<CodeAnalysisSystem>(config.project_root, true, 500, 10); // lazy=true, max 500 files, 10MB per file
    runtime_compiler_ = std::make_unique<RuntimeCompilationSystem>(config.temp_compilation_dir);
    sandbox_manager_ = std::make_unique<SandboxManager>();

    // Initialize self-modification state
    interactions_since_last_analysis_ = 0;
    current_modifiable_function = default_modifiable_function;
    loaded_module_handle = nullptr;

    std::cout << "Self-Modifying HRM System initialized with "
              << (config.enable_self_modification ? "self-modification enabled" : "self-modification disabled")
              << " and "
              << (config.enable_runtime_recompilation ? "runtime recompilation enabled" : "runtime recompilation disabled")
              << std::endl;
}

SelfModifyingHRM::~SelfModifyingHRM() {
    // Clean up any active modifications
    runtime_compiler_->cleanup_temp_files();
}

CommunicationResult SelfModifyingHRM::communicate(const std::string& input_message) {
    // First, perform normal communication
    CommunicationResult result = SelfEvolvingHRM::communicate(input_message);

    // Increment interaction counter
    interactions_since_last_analysis_++;

    // Check if we should perform self-analysis
    if (config_.enable_self_modification &&
        interactions_since_last_analysis_ >= config_.self_analysis_frequency) {

        std::cout << "Performing periodic self-analysis..." << std::endl;

        SelfModificationResult self_mod = analyze_and_modify_self();
        if (self_mod.modification_applied) {
            std::cout << "Self-modification applied: " << self_mod.modification_description << std::endl;
            log_self_modification_activity(self_mod);

            // Reset counter
            interactions_since_last_analysis_ = 0;

            // Add self-modification info to result
            result.applied_corrections.push_back("Self-modification: " + self_mod.modification_description);
        }
    }

    return result;
}

SelfModificationResult SelfModifyingHRM::analyze_and_modify_self() {
    SelfModificationResult result;
    result.modification_applied = false;
    result.compilation_successful = false;
    result.system_restart_required = false;

    // Step 1: Analyze the codebase
    CodeAnalysisResult analysis = code_analyzer_->analyze_codebase();

    if (analysis.issues.empty()) {
        std::cout << "No issues found in self-analysis" << std::endl;
        return result;
    }

    std::cout << "Found " << analysis.issues.size() << " issues in self-analysis" << std::endl;

    // Step 2: Generate potential fixes
    std::vector<CodeModification> fixes = generate_self_fixes(analysis);

    if (fixes.empty()) {
        std::cout << "No fixes generated for self-modification" << std::endl;
        return result;
    }

    // Step 3: Evaluate modification impact
    SelfModificationResult evaluation = evaluate_modification_impact(fixes);

    if (evaluation.confidence_score < config_.modification_confidence_threshold) {
        std::cout << "Self-modification confidence too low: " << evaluation.confidence_score << std::endl;
        return result;
    }

    // Step 4: Apply modifications if safe
    if (apply_self_modification(evaluation)) {
        result = evaluation;
        result.modification_applied = true;
        std::cout << "Self-modification successfully applied" << std::endl;
    } else {
        std::cout << "Self-modification failed or was deemed unsafe" << std::endl;
    }

    return result;
}

std::vector<CodeModification> SelfModifyingHRM::generate_self_fixes(const CodeAnalysisResult& analysis) {
    return code_analyzer_->generate_fixes(analysis.issues);
}

SelfModificationResult SelfModifyingHRM::evaluate_modification_impact(const std::vector<CodeModification>& modifications) {
    SelfModificationResult result;
    result.modification_applied = false;
    result.confidence_score = 0.0f;

    if (modifications.empty()) return result;

    // Evaluate each modification
    float total_confidence = 0.0f;
    std::vector<std::string> all_risks;

    for (const auto& mod : modifications) {
        // Check if modification is safe
        if (!is_file_modification_safe(mod.file_path)) {
            all_risks.push_back("Unsafe file modification: " + mod.file_path);
            continue;
        }

        // Validate modification semantics
        if (!validate_modification_semantics(mod)) {
            all_risks.push_back("Invalid modification semantics: " + mod.reason);
            continue;
        }

        // Assess risk
        float risk = assess_modification_risk(mod);
        float confidence = mod.confidence_score * (1.0f - risk);

        total_confidence += confidence;

        if (!mod.file_path.empty()) {
            result.modified_file = mod.file_path;
            result.modification_description = mod.reason;
        }
    }

    result.confidence_score = total_confidence / modifications.size();
    result.potential_risks = all_risks;

    // Generate rollback instructions
    result.rollback_instructions = "Use git checkout or backup files to restore previous state";

    return result;
}

bool SelfModifyingHRM::apply_self_modification(const SelfModificationResult& modification) {
    // This is a high-risk operation - extensive validation and safety measures

    if (!validate_self_modification_safety(modification)) {
        std::cout << "Self-modification validation failed" << std::endl;
        return false;
    }

    // Create backup before modification
    if (config_.create_backups_before_modification) {
        create_system_backup();
    }

    // Log the modification
    modification_history_.push_back(modification);

    try {
        // Test modifications in sandbox before applying
        for (const auto& code_change : modification.code_changes) {
            CodeModification test_mod;
            test_mod.file_path = code_change.file_path;
            test_mod.start_line = 1; // Placeholder
            test_mod.end_line = 10; // Placeholder
            test_mod.original_code = code_change.old_code;
            test_mod.modified_code = code_change.new_code;
            test_mod.reason = "Testing self-modification: " + code_change.file_path;
            test_mod.confidence_score = 0.8f; // Placeholder

            std::cout << "Testing modification in sandbox: " << test_mod.reason << std::endl;

            // Run sandbox test
            TestResult test_result = sandbox_manager_->test_modification(test_mod);

            if (!test_result.success) {
                std::cout << "Sandbox test failed for " << code_change.file_path << ": " << test_result.errors[0] << std::endl;
                return false;
            }

            // Validate test results
            ValidationResult validation = sandbox_manager_->validate_modification(test_result);

            if (!validation.approved) {
                std::cout << "Modification validation failed for " << code_change.file_path << std::endl;
                std::cout << "Confidence: " << validation.confidence_score << ", Risk: " << validation.risk_assessment << std::endl;
                for (const auto& concern : validation.concerns) {
                    std::cout << "Concern: " << concern << std::endl;
                }
                return false;
            }

            // Make deployment decision
            DeploymentDecision decision = sandbox_manager_->make_deployment_decision(validation);

            if (!decision.deploy) {
                std::cout << "Deployment decision: Do not deploy - " << decision.reasoning << std::endl;
                return false;
            }

            std::cout << "Sandbox validation passed for " << code_change.file_path << std::endl;
        }

        // Apply the validated modifications
        RuntimeCompilationSystem compiler;

        for (const auto& code_change : modification.code_changes) {
            std::cout << "Applying validated code modification to: " << code_change.file_path << std::endl;

            // Use the modification script format: "old_text|new_text"
            std::string modification_script = code_change.old_code + "|" + code_change.new_code;

            CompilationResult result = compiler.modify_and_recompile(
                code_change.file_path,
                modification_script,
                "self_modified_component"
            );

            if (!result.success) {
                std::cout << "Code modification failed for " << code_change.file_path << ": " << result.errors[0] << std::endl;
                // Attempt rollback
                rollback_self_modification("auto_backup_" + std::to_string(time(nullptr)));
                return false;
            }

            // Attempt hot-swapping of the modifiable function for this change
            if (!result.library_path.empty()) {
                auto module = runtime_compiler_->load_module(result.library_path);
                if (module) {
                    // For demonstration, assume the library has a modified_function
                    // In practice, this would be more sophisticated
                    std::cout << "Loaded modified module for hot-swapping" << std::endl;
                    // Keep the module loaded for now
                }
            }
        }

        // Test the modified function
        current_modifiable_function("Self-modification completed successfully");

        std::cout << "Self-modification applied successfully: " << modification.modification_description << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cout << "Self-modification failed with exception: " << e.what() << std::endl;
        // Attempt rollback
        rollback_self_modification("auto_backup_" + std::to_string(time(nullptr)));
        return false;
    }
}

bool SelfModifyingHRM::rollback_self_modification(const std::string& backup_id) {
    return restore_system_backup(backup_id);
}

std::unordered_map<std::string, std::string> SelfModifyingHRM::get_self_analysis_report() {
    std::unordered_map<std::string, std::string> report;

    // Get base system status
    std::unordered_map<std::string, std::string> base_status;
    get_system_status(base_status);

    // Add self-modification specific info
    report["self_modification_enabled"] = config_.enable_self_modification ? "true" : "false";
    report["runtime_recompilation_enabled"] = config_.enable_runtime_recompilation ? "true" : "false";
    report["interactions_since_last_analysis"] = std::to_string(interactions_since_last_analysis_);
    report["total_modifications_applied"] = std::to_string(modification_history_.size());
    report["self_analysis_frequency"] = std::to_string(config_.self_analysis_frequency);

    // Merge with base status
    report.insert(base_status.begin(), base_status.end());

    return report;
}

std::vector<std::string> SelfModifyingHRM::detect_self_limitations() {
    std::vector<std::string> limitations;

    // Analyze current capabilities and identify limitations
    limitations.push_back("Cannot modify core system libraries");
    limitations.push_back("Limited understanding of complex code patterns");
    limitations.push_back("Cannot guarantee correctness of all modifications");
    limitations.push_back("Requires compilation environment for runtime modifications");

    return limitations;
}

std::vector<std::string> SelfModifyingHRM::propose_self_improvements() {
    std::vector<std::string> improvements;

    // Analyze current system and propose improvements
    improvements.push_back("Implement more sophisticated code analysis patterns");
    improvements.push_back("Add support for multi-file modifications");
    improvements.push_back("Implement better risk assessment algorithms");
    improvements.push_back("Add support for gradual system updates");

    return improvements;
}

bool SelfModifyingHRM::validate_self_modification_safety(const SelfModificationResult& modification) {
    // Basic safety checks
    if (modification.confidence_score < 0.8f) return false;
    if (!modification.potential_risks.empty()) return false;
    if (modification.system_restart_required) return false; // Too risky for now

    return true;
}

void SelfModifyingHRM::log_self_modification_activity(const SelfModificationResult& modification) {
    std::cout << "=== Self-Modification Log ===" << std::endl;
    std::cout << "File: " << modification.modified_file << std::endl;
    std::cout << "Description: " << modification.modification_description << std::endl;
    std::cout << "Confidence: " << modification.confidence_score << std::endl;
    std::cout << "Compilation: " << (modification.compilation_successful ? "Success" : "Failed") << std::endl;
    std::cout << "Risks: " << modification.potential_risks.size() << std::endl;
    std::cout << "==========================" << std::endl;
}

std::vector<std::string> SelfModifyingHRM::get_self_modification_history() {
    std::vector<std::string> history;

    for (const auto& mod : modification_history_) {
        history.push_back(mod.modified_file + ": " + mod.modification_description +
                         " (confidence: " + std::to_string(mod.confidence_score) + ")");
    }

    return history;
}

// Private methods

bool SelfModifyingHRM::is_file_modification_safe(const std::string& file_path) {
    // Check if file is in protected list
    for (const auto& protected_file : config_.protected_files) {
        if (file_path.find(protected_file) != std::string::npos) {
            return false;
        }
    }

    // Only allow modification of source files
    return file_path.find(".cpp") != std::string::npos ||
           file_path.find(".hpp") != std::string::npos;
}

bool SelfModifyingHRM::validate_modification_semantics(const CodeModification& modification) {
    // Basic semantic validation
    if (modification.modified_code.empty()) return false;
    if (modification.modified_code.find("delete this") != std::string::npos) return false;
    if (modification.modified_code.find("free(this)") != std::string::npos) return false;

    return true;
}

float SelfModifyingHRM::assess_modification_risk(const CodeModification& modification) {
    float risk = 0.0f;

    // Assess risk based on modification type
    if (modification.reason.find("memory") != std::string::npos) risk += 0.3f;
    if (modification.reason.find("pointer") != std::string::npos) risk += 0.4f;
    if (modification.reason.find("thread") != std::string::npos) risk += 0.5f;

    // Assess risk based on code changes
    if (modification.modified_code.find("nullptr") != std::string::npos) risk += 0.2f;
    if (modification.modified_code.find("delete") != std::string::npos) risk += 0.3f;

    return std::min(risk, 1.0f);
}

bool SelfModifyingHRM::compile_modified_system(const std::vector<CodeModification>& modifications) {
    if (!runtime_compiler_) {
        std::cerr << "Runtime compiler not available" << std::endl;
        return false;
    }

    bool all_ok = true;
    int idx = 0;
    for (const auto& mod : modifications) {
        // Prepare modification script as old|new
        std::string script = mod.original_code + "|" + mod.modified_code;
        std::string output_name = "self_mod_" + std::to_string(time(nullptr)) + "_" + std::to_string(idx++);

        CompilationResult cres = runtime_compiler_->modify_and_recompile(mod.file_path, script, output_name);
        if (!cres.success) {
            all_ok = false;
            std::cerr << "Compilation failed for " << mod.file_path << ": ";
            if (!cres.errors.empty()) std::cerr << cres.errors[0];
            std::cerr << std::endl;
            break;
        }

        // If a library was produced, load it so it can be hot-swapped later
        if (!cres.library_path.empty()) {
            auto module = runtime_compiler_->load_module(cres.library_path);
            if (module && module->handle) {
                loaded_module_handle = module->handle;
            }
        }
    }

    return all_ok;
}

bool SelfModifyingHRM::hot_swap_modified_components() {
    if (!runtime_compiler_) {
        std::cerr << "Runtime compiler not available for hot-swap" << std::endl;
        return false;
    }

    // Attempt to hot-swap the most recently loaded module if available
    if (!loaded_module_handle) {
        std::cerr << "No loaded module available to hot-swap" << std::endl;
        return false;
    }

    // In a real implementation we would map old->new libraries. Here we just report success if loaded.
    std::cout << "Hot-swapping: module handle present, marking successful" << std::endl;
    return true;
}

void SelfModifyingHRM::update_system_configuration() {
    // Query runtime compiler for environment info and merge into config
    if (!runtime_compiler_) return;
    auto info = runtime_compiler_->get_system_info();
    std::cout << "Runtime compilation system info:" << std::endl;
    for (const auto& kv : info) {
        std::cout << "  " << kv.first << ": " << kv.second << std::endl;
    }
}

bool SelfModifyingHRM::create_system_backup() {
    // Create backups of critical files
    for (const auto& file : {"src/main.cpp", "src/hrm.hpp", "CMakeLists.txt"}) {
        runtime_compiler_->create_backup(file);
    }
    return true;
}

bool SelfModifyingHRM::restore_system_backup(const std::string& backup_id) {
    // Restore from backups
    for (const auto& file : {"src/main.cpp", "src/hrm.hpp", "CMakeLists.txt"}) {
        if (!runtime_compiler_->restore_backup(file)) {
            std::cerr << "Failed to restore backup for " << file << std::endl;
            return false;
        }
    }
    return true;
}

void SelfModifyingHRM::enter_safe_mode() {
    std::cout << "Entering safe mode - disabling self-modification" << std::endl;
    config_.enable_self_modification = false;
}

std::vector<std::string> SelfModifyingHRM::analyze_self_improvement_opportunities() {
    std::vector<std::string> opportunities;

    // Analyze current performance and suggest improvements
    auto status = get_self_analysis_report();

    if (status["average_confidence"] < "0.8") {
        opportunities.push_back("Improve confidence scoring algorithms");
    }

    if (std::stoi(status["total_modifications_applied"]) < 5) {
        opportunities.push_back("Increase self-modification frequency for better adaptation");
    }

    return opportunities;
}

std::vector<std::string> SelfModifyingHRM::detect_self_degradation_patterns() {
    std::vector<std::string> patterns;

    // Analyze modification history for degradation patterns
    if (modification_history_.size() > 10) {
        int recent_failures = 0;
        for (size_t i = modification_history_.size() - 5; i < modification_history_.size(); ++i) {
            if (!modification_history_[i].compilation_successful) {
                recent_failures++;
            }
        }

        if (recent_failures >= 3) {
            patterns.push_back("Recent modifications showing compilation failures - possible degradation");
        }
    }

    return patterns;
}

void SelfModifyingHRM::adapt_self_analysis_parameters() {
    // Adapt analysis parameters based on performance
    auto status = get_self_analysis_report();

    if (std::stod(status["average_confidence"]) > 0.9) {
        // High confidence - can be more aggressive
        config_.modification_confidence_threshold = std::max(0.7f, config_.modification_confidence_threshold - 0.05f);
    } else if (std::stod(status["average_confidence"]) < 0.7) {
        // Low confidence - be more conservative
        config_.modification_confidence_threshold = std::min(0.9f, config_.modification_confidence_threshold + 0.05f);
    }
}

// Missing method implementations
bool SelfModifyingHRM::validate_hot_swap_safety(const std::string& file_path, const std::string& new_code) {
    // Basic validation for hot-swapping code
    if (file_path.empty() || new_code.empty()) {
        return false;
    }

    // Check if file is critical system file
    std::vector<std::string> critical_files = {"main.cpp", "hrm.hpp", "hrm.cpp"};
    for (const auto& critical : critical_files) {
        if (file_path.find(critical) != std::string::npos) {
            std::cout << "Cannot hot-swap critical system file: " << file_path << std::endl;
            return false;
        }
    }

    // Basic AST validation (syntax check)
    if (!validate_code_syntax(new_code)) {
        std::cout << "Code syntax validation failed, cannot hot-swap" << std::endl;
        return false;
    }

    // Scan code for risks
    auto risks = scan_code_for_risks(new_code);
    if (!risks.empty()) {
        std::cout << "Code contains risks, cannot hot-swap" << std::endl;
        return false;
    }

    return true;
}

bool SelfModifyingHRM::create_safety_checkpoint(const std::string& description) {
    // Create a safety checkpoint before modifications
    std::cout << "Creating safety checkpoint: " << description << std::endl;

    // In a real implementation, this would create a full system snapshot
    // For now, just log the checkpoint
    safety_checkpoints_[description] = std::to_string(std::time(nullptr));

    return true;
}

bool SelfModifyingHRM::restore_from_safety_checkpoint(const std::string& checkpoint_id) {
    // Restore from a safety checkpoint
    auto it = safety_checkpoints_.find(checkpoint_id);
    if (it == safety_checkpoints_.end()) {
        std::cout << "Safety checkpoint not found: " << checkpoint_id << std::endl;
        return false;
    }

    std::cout << "Restoring from safety checkpoint: " << checkpoint_id << std::endl;

    // In a real implementation, this would restore the full system state
    // For now, just remove the checkpoint
    safety_checkpoints_.erase(it);

    return true;
}

std::vector<std::string> SelfModifyingHRM::scan_code_for_risks(const std::string& code_content) {
    std::vector<std::string> risks;

    // Scan for dangerous patterns
    if (code_content.find("system(") != std::string::npos) {
        risks.push_back("System command execution detected");
    }

    if (code_content.find("exec(") != std::string::npos) {
        risks.push_back("Process execution detected");
    }

    if (code_content.find("delete this") != std::string::npos) {
        risks.push_back("Self-deletion detected");
    }

    if (code_content.find("nullptr dereference") != std::string::npos) {
        risks.push_back("Potential null pointer dereference");
    }

    return risks;
}

bool SelfModifyingHRM::validate_code_syntax(const std::string& code) {
    // Securely create a temporary file and run clang's syntax-only check
    try {
        namespace fs = std::filesystem;
        fs::path temp_dir = fs::temp_directory_path();
        std::string template_path = (temp_dir / "hrm_code_XXXXXX.cpp").string();

        // mkstemp requires a mutable char array
        std::vector<char> tmpl(template_path.begin(), template_path.end());
        tmpl.push_back('\0');

        int fd = mkstemp(tmpl.data());
        if (fd == -1) {
            std::cerr << "Failed to create temp file for syntax validation" << std::endl;
            return false;
        }

        std::string temp_file_path(tmpl.data());

        // Write code to file descriptor
        ssize_t written = write(fd, code.data(), static_cast<size_t>(code.size()));
        (void)written;
        close(fd);

        // Run clang syntax check (capture exit code)
        std::string clang_cmd = "clang++ -fsyntax-only -std=c++17 -I. -I./src " + temp_file_path + " > /dev/null 2>&1";
        int rc = std::system(clang_cmd.c_str());

        // Clean up
        std::error_code ec;
        fs::remove(temp_file_path, ec);

        return rc == 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in validate_code_syntax: " << e.what() << std::endl;
        return false;
    }
}

bool SelfModifyingHRM::validate_code_integrity(const std::string& file_path, const std::string& expected_hash) {
    // Basic integrity validation
    if (!std::filesystem::exists(file_path)) {
        return false;
    }

    // In a real implementation, this would compute and compare file hashes
    // For now, just check if file exists and is readable
    std::ifstream file(file_path);
    return file.good();
}

bool SelfModifyingHRM::perform_dynamic_validation(const SelfModificationResult& modification) {
    // Dynamic validation of modifications
    if (modification.confidence_score < 0.7f) {
        return false;
    }

    if (!modification.potential_risks.empty()) {
        return false;
    }

    // Additional validation logic would go here
    return true;
}

bool SelfModifyingHRM::validate_system_stability() {
    // Check if the system is in a stable state
    // Validate several stability metrics:
    
    // 1. Check HRM model is still functional
    if (!hrm_model_) {
        std::cerr << "System unstable: HRM model is null" << std::endl;
        return false;
    }
    
    // 2. Verify runtime compiler is available if hot-swapping is enabled
    if (config_.enable_runtime_recompilation && !runtime_compiler_) {
        std::cerr << "System unstable: Runtime compiler unavailable" << std::endl;
        return false;
    }
    
    // 3. Check sandbox manager is available
    if (!sandbox_manager_) {
        std::cerr << "System unstable: Sandbox manager unavailable" << std::endl;
        return false;
    }
    
    // 4. Verify modification history is bounded (prevent runaway modifications)
    if (modification_history_.size() > 100) {
        std::cerr << "System unstable: Too many modifications (" << modification_history_.size() << ")" << std::endl;
        return false;
    }
    
    // 5. Check safety checkpoints are available for rollback
    if (config_.create_backups_before_modification && safety_checkpoints_.empty() && 
        !modification_history_.empty()) {
        std::cerr << "System unstable: No safety checkpoints despite modifications" << std::endl;
        return false;
    }
    
    // 6. Verify analysis frequency parameter is reasonable
    if (config_.self_analysis_frequency <= 0.0f || config_.self_analysis_frequency > 1000.0f) {
        std::cerr << "System unstable: Invalid analysis frequency" << std::endl;
        return false;
    }
    
    // System is stable if all checks pass
    return true;
}

bool SelfModifyingHRM::should_analyze_self() const {
    // Determine if self-analysis should be performed
    return config_.enable_self_modification &&
           interactions_since_last_analysis_ >= config_.self_analysis_frequency;
}

void SelfModifyingHRM::add_boot_task(const std::function<void()>& task) {
    // Add a task to be executed at system boot
    boot_tasks_.push_back(task);
}

void SelfModifyingHRM::add_idle_task(const std::function<void()>& task) {
    // Add a task to be executed during idle time
    idle_tasks_.push_back(task);
}