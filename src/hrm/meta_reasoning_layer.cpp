#include "meta_reasoning_layer.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <regex>
#include <cmath>
#include <unordered_set>
#include <filesystem>
#include <atomic>

// Evolutionary guard: Prevent duplicate HRM initializations
static std::atomic<int> hrm_instance_count(0);

MetaReasoningLayer::MetaReasoningLayer(const MetaReasoningConfig& config) : config_(config) {
    // Evolutionary guard: Detect and warn about duplicate HRM initializations
    int current_count = hrm_instance_count.fetch_add(1);
    if (current_count > 0) {
        std::cerr << "WARNING: Evolutionary guard detected duplicate HRM initialization (#" << (current_count + 1) << ")" << std::endl;
        std::cerr << "This may indicate architectural issues causing memory conflicts and std::bad_alloc" << std::endl;
        std::cerr << "Consider reviewing singleton pattern implementation in ResourceAwareHRM" << std::endl;
    }

    std::cout << "Initializing Meta-Reasoning Layer..." << std::endl;
    std::cout << "Meta-Reasoning Layer initialized with self-repair "
              << (config_.enable_self_repair ? "enabled" : "disabled")
              << " (config values: depth=" << config_.analysis_depth
              << ", threshold=" << config_.confidence_threshold
              << ", attempts=" << config_.max_correction_attempts << ")" << std::endl;
}

AnalysisResult MetaReasoningLayer::analyze_output(const std::string& input,
                                                 const std::string& output,
                                                 const HRMCarry& model_state) {
    AnalysisResult result;
    result.confidence_score = 1.0f;
    result.requires_repair = false;

    if (!config_.enable_self_repair) {
        return result;
    }

    // Analyze semantic coherence
    auto semantic_issues = analyze_semantic_coherence(output);
    result.detected_issues.insert(result.detected_issues.end(),
                                 semantic_issues.begin(), semantic_issues.end());

    // Analyze syntactic correctness
    auto syntactic_issues = analyze_syntactic_correctness(output);
    result.detected_issues.insert(result.detected_issues.end(),
                                 syntactic_issues.begin(), syntactic_issues.end());

    // Analyze logical soundness
    auto logical_issues = analyze_logical_soundness(input, output);
    result.detected_issues.insert(result.detected_issues.end(),
                                 logical_issues.begin(), logical_issues.end());

    // Detect contradictions
    auto contradictions = detect_contradictions(output);
    result.detected_issues.insert(result.detected_issues.end(),
                                 contradictions.begin(), contradictions.end());

    // Compute confidence score
    if (config_.enable_confidence_scoring) {
        // Simplified confidence calculation based on issue count
        float issue_penalty = std::min(1.0f, result.detected_issues.size() * 0.1f);
        result.confidence_score = 1.0f - issue_penalty;
    }

    // Determine if repair is needed
    result.requires_repair = (result.confidence_score < config_.confidence_threshold) ||
                            !result.detected_issues.empty();

    // Generate suggested corrections
    if (result.requires_repair) {
        for (const auto& issue : result.detected_issues) {
            if (issue.find("contradiction") != std::string::npos) {
                result.suggested_corrections.push_back("Resolve logical contradiction");
            } else if (issue.find("grammar") != std::string::npos) {
                result.suggested_corrections.push_back("Fix grammatical errors");
            } else if (issue.find("semantic") != std::string::npos) {
                result.suggested_corrections.push_back("Improve semantic coherence");
            }
        }
    }

    return result;
}

RepairResult MetaReasoningLayer::attempt_repair(const std::string& input,
                                               const std::string& flawed_output,
                                               const AnalysisResult& analysis,
                                               const HRMCarry& model_state) {
    RepairResult result;
    result.repair_successful = false;
    result.attempts_used = 0;
    result.improvement_score = 0.0f;

    if (!config_.enable_self_repair || !analysis.requires_repair) {
        result.repaired_output = flawed_output;
        return result;
    }

    std::string current_output = flawed_output;
    float best_confidence = analysis.confidence_score;

    for (int attempt = 0; attempt < config_.max_correction_attempts; ++attempt) {
        result.attempts_used++;

        // Try different repair strategies
        std::vector<std::string> candidates;

        // Strategy 1: Local corrections
        std::string local_corrected = apply_local_corrections(current_output, analysis.detected_issues);
        if (local_corrected != current_output) {
            candidates.push_back(local_corrected);
        }

        // Strategy 2: Regenerate problematic sections
        std::vector<size_t> problem_positions;
        for (size_t i = 0; i < analysis.detected_issues.size(); ++i) {
            problem_positions.push_back(i * (current_output.length() / std::max(size_t(1), analysis.detected_issues.size())));
        }

        std::string regenerated = regenerate_problematic_sections(current_output, problem_positions);
        if (regenerated != current_output) {
            candidates.push_back(regenerated);
        }

        // Strategy 3: Use learned correction patterns
        for (const auto& pattern : correction_patterns_) {
            std::regex pattern_regex(pattern.first); // pattern.first is the key (string)
            std::string pattern_corrected = std::regex_replace(current_output, pattern_regex, pattern.second.first); // pattern.second is the pair
            if (pattern_corrected != current_output) {
                candidates.push_back(pattern_corrected);
            }
        }

        // Evaluate candidates
        std::string best_candidate = current_output;
        float best_candidate_confidence = best_confidence;

        for (const auto& candidate : candidates) {
            // Analyze the candidate
            auto candidate_analysis = analyze_output(input, candidate, model_state);

            if (candidate_analysis.confidence_score > best_candidate_confidence) {
                best_candidate = candidate;
                best_candidate_confidence = candidate_analysis.confidence_score;
            }
        }

        // Check for improvement
        if (best_candidate_confidence > best_confidence) {
            current_output = best_candidate;
            best_confidence = best_candidate_confidence;
            result.improvement_score = best_candidate_confidence - analysis.confidence_score;
        } else {
            // No improvement, stop trying
            break;
        }
    }

    result.repaired_output = current_output;
    result.repair_successful = (best_confidence > analysis.confidence_score);

    return result;
}

float MetaReasoningLayer::compute_output_confidence(const Tensor& logits, const std::string& output) {
    // Simplified confidence calculation based on logit distribution
    if (logits.data.empty()) return 0.5f;

    // Calculate entropy as a measure of uncertainty
    double entropy = 0.0;
    double max_logit = *std::max_element(logits.data.begin(), logits.data.end());

    // Softmax calculation
    double sum_exp = 0.0;
    for (float logit : logits.data) {
        sum_exp += std::exp(logit - max_logit);
    }

    for (float logit : logits.data) {
        double prob = std::exp(logit - max_logit) / sum_exp;
        if (prob > 0) {
            entropy -= prob * std::log(prob);
        }
    }

    // Convert entropy to confidence (lower entropy = higher confidence)
    double max_entropy = std::log(logits.data.size());
    double confidence = 1.0 - (entropy / max_entropy);

    return static_cast<float>(std::max(0.0, std::min(1.0, confidence)));
}

bool MetaReasoningLayer::check_logical_consistency(const std::string& input, const std::string& output) {
    // Simple logical consistency checks
    auto contradictions = detect_contradictions(output);
    return contradictions.empty();
}

std::vector<std::string> MetaReasoningLayer::detect_contradictions(const std::string& text) {
    std::vector<std::string> contradictions;

    // Simple contradiction detection (can be extended)
    std::vector<std::string> contradiction_patterns = {
        "is true.*is false",
        "cannot.*can",
        "never.*always",
        "impossible.*possible"
    };

    for (const auto& pattern : contradiction_patterns) {
        std::regex regex_pattern(pattern, std::regex_constants::icase);
        if (std::regex_search(text, regex_pattern)) {
            contradictions.push_back("Detected contradiction: " + pattern);
        }
    }

    return contradictions;
}

void MetaReasoningLayer::learn_from_correction(const std::string& original_output,
                                             const std::string& corrected_output,
                                             const RepairResult& repair_result) {
    if (!repair_result.repair_successful) return;

    // Extract patterns from successful corrections
    auto original_tokens = tokenize_text(original_output);
    auto corrected_tokens = tokenize_text(corrected_output);

    // Find differences and learn patterns
    for (size_t i = 0; i < std::min(original_tokens.size(), corrected_tokens.size()); ++i) {
        if (original_tokens[i] != corrected_tokens[i]) {
            std::string pattern = original_tokens[i];
            std::string correction = corrected_tokens[i];
            update_correction_patterns(pattern, correction, repair_result.improvement_score);
            break; // Learn one pattern per correction for now
        }
    }
}

// Private helper methods

std::vector<std::string> MetaReasoningLayer::analyze_semantic_coherence(const std::string& text) {
    std::vector<std::string> issues;

    // Simple semantic checks
    if (text.empty()) {
        issues.push_back("Empty output");
        return issues;
    }

    // Check for repetitive patterns
    std::vector<std::string> words = tokenize_text(text);
    std::unordered_map<std::string, int> word_counts;
    for (const auto& word : words) {
        word_counts[word]++;
    }

    for (const auto& pair : word_counts) {
        if (pair.second > words.size() * 0.3) { // More than 30% of words are the same
            issues.push_back("Excessive repetition of word: " + pair.first);
        }
    }

    return issues;
}

std::vector<std::string> MetaReasoningLayer::analyze_syntactic_correctness(const std::string& text) {
    std::vector<std::string> issues;

    // Basic grammar checks
    if (!is_grammatically_correct(text)) {
        issues.push_back("Potential grammatical errors detected");
    }

    // Check for incomplete sentences
    size_t sentence_count = 0;
    for (char c : text) {
        if (c == '.' || c == '!' || c == '?') sentence_count++;
    }

    if (sentence_count == 0 && text.length() > 50) {
        issues.push_back("Missing sentence endings");
    }

    return issues;
}

std::vector<std::string> MetaReasoningLayer::analyze_logical_soundness(const std::string& input, const std::string& output) {
    std::vector<std::string> issues;

    // Check if output addresses the input
    float similarity = calculate_semantic_similarity(input, output);
    if (similarity < 0.3f) {
        issues.push_back("Output may not adequately address the input query");
    }

    return issues;
}

std::string MetaReasoningLayer::apply_local_corrections(const std::string& text,
                                                       const std::vector<std::string>& issues) {
    std::string corrected = text;

    for (const auto& issue : issues) {
        if (issue.find("repetition") != std::string::npos) {
            // Remove excessive repetition (simplified)
            std::regex repeat_pattern("(\\b\\w+\\b)(\\s+\\1)+");
            corrected = std::regex_replace(corrected, repeat_pattern, "$1");
        }
    }

    return corrected;
}

std::string MetaReasoningLayer::regenerate_problematic_sections(const std::string& text,
                                                               const std::vector<size_t>& problem_positions) {
    // Simplified: just remove problematic sections
    std::string corrected = text;

    for (auto pos : problem_positions) {
        if (pos < corrected.length()) {
            // Remove a few characters around the problem position
            size_t remove_start = std::max(size_t(0), pos - 5);
            size_t remove_end = std::min(corrected.length(), pos + 5);
            corrected.erase(remove_start, remove_end - remove_start);
        }
    }

    return corrected;
}

void MetaReasoningLayer::update_correction_patterns(const std::string& pattern,
                                                   const std::string& correction,
                                                   float success_rate) {
    auto key = pattern;
    auto& existing = correction_patterns_[key];

    // Update with exponential moving average
    float alpha = 0.1f;
    existing.second = alpha * success_rate + (1 - alpha) * existing.second;
    if (existing.first.empty() || existing.second < success_rate) {
        existing.first = correction;
    }
}

std::vector<std::string> MetaReasoningLayer::tokenize_text(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token) {
        // Remove punctuation
        token.erase(std::remove_if(token.begin(), token.end(),
                                  [](char c) { return std::ispunct(c); }), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

float MetaReasoningLayer::calculate_semantic_similarity(const std::string& text1, const std::string& text2) {
    // Simplified semantic similarity using Jaccard similarity
    auto tokens1 = tokenize_text(text1);
    auto tokens2 = tokenize_text(text2);

    std::unordered_set<std::string> set1(tokens1.begin(), tokens1.end());
    std::unordered_set<std::string> set2(tokens2.begin(), tokens2.end());

    std::vector<std::string> intersection;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                         std::back_inserter(intersection));

    std::vector<std::string> union_set;
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
                  std::back_inserter(union_set));

    if (union_set.empty()) return 0.0f;

    return static_cast<float>(intersection.size()) / union_set.size();
}

bool MetaReasoningLayer::is_grammatically_correct(const std::string& sentence) {
    // Very basic grammar check
    if (sentence.empty()) return false;

    // Check for basic sentence structure
    bool has_subject = false;
    bool has_verb = false;

    // Simplified: just check if it starts with a capital letter and ends with punctuation
    if (!std::isupper(sentence[0])) return false;

    char last_char = sentence.back();
    if (last_char != '.' && last_char != '!' && last_char != '?') return false;

    return true;
}