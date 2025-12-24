#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

struct CodeIssue {
    std::string file_path;
    int line_number;
    std::string issue_type;
    std::string description;
    std::string severity; // "low", "medium", "high", "critical"
    std::string suggested_fix;
    std::vector<std::string> code_context;
};

struct CodeAnalysisResult {
    std::vector<CodeIssue> issues;
    std::unordered_map<std::string, int> issue_counts_by_type;
    std::unordered_map<std::string, int> issue_counts_by_severity;
    float overall_quality_score; // 0.0 to 1.0
    std::vector<std::string> recommendations;
};

struct CodeModification {
    std::string file_path;
    int start_line;
    int end_line;
    std::string original_code;
    std::string modified_code;
    std::string reason;
    float confidence_score; // 0.0 to 1.0
};

class CodeAnalysisSystem {
public:
    CodeAnalysisSystem(const std::string& project_root = ".", bool lazy_load = true, size_t max_files = 500, size_t max_file_size_mb = 10);
    ~CodeAnalysisSystem() = default;

    // Core analysis functions
    CodeAnalysisResult analyze_codebase();
    CodeAnalysisResult analyze_file(const std::string& file_path);
    CodeAnalysisResult analyze_code_snippet(const std::string& code, const std::string& file_path = "");

    // Bug detection
    std::vector<CodeIssue> detect_memory_issues(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_logic_errors(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_performance_issues(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_security_vulnerabilities(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_style_issues(const std::string& code, const std::string& file_path);

    // Code modification
    std::vector<CodeModification> generate_fixes(const std::vector<CodeIssue>& issues);
    bool apply_modification(const CodeModification& modification);
    bool validate_modification(const CodeModification& modification);

    // Compilation and testing
    bool test_compilation(const std::vector<CodeModification>& modifications);
    bool run_unit_tests(const std::vector<CodeModification>& modifications);

private:
    std::string project_root_;
    std::vector<std::string> source_files_;
    std::unordered_map<std::string, std::regex> bug_patterns_;
    bool lazy_load_;
    size_t max_files_;
    size_t max_file_size_mb_;
    bool files_discovered_;

    // Helper functions
    void discover_source_files();
    std::vector<std::string> read_file_lines(const std::string& file_path);
    std::string extract_code_context(const std::vector<std::string>& lines, int line_number, int context_lines = 3);

    // Pattern initialization
    void initialize_bug_patterns();

    // Specific bug detection methods
    std::vector<CodeIssue> detect_null_pointer_dereferences(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_memory_leaks(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_buffer_overflows(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_uninitialized_variables(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_infinite_loops(const std::string& code, const std::string& file_path);
    std::vector<CodeIssue> detect_race_conditions(const std::string& code, const std::string& file_path);

    // Code quality metrics
    float calculate_code_quality_score(const std::vector<CodeIssue>& issues);
    std::vector<std::string> generate_recommendations(const CodeAnalysisResult& result);

    // Modification helpers
    CodeModification create_memory_fix(const CodeIssue& issue);
    CodeModification create_logic_fix(const CodeIssue& issue);
    CodeModification create_performance_fix(const CodeIssue& issue);
    CodeModification create_security_fix(const CodeIssue& issue);
    CodeModification create_style_fix(const CodeIssue& issue);
};