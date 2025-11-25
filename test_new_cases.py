# -*- coding: utf-8 -*-
# Test với các trường hợp mới

def ipa_compare(real_ipa, recorded_ipa):
    # Handle underscores in recorded IPA by treating them as gaps
    recorded_list = []
    for char in recorded_ipa:
        if char == '_':
            recorded_list.append('_')  # Keep underscore to mark gap
        elif char not in [' ', '\t']:  # Skip spaces
            recorded_list.append(char)
    
    ipa_units_recorded = recorded_list
    ipa_units_real = list(real_ipa)
    
    print(f"\nReal IPA: '{real_ipa}' -> {ipa_units_real}")
    print(f"Recorded IPA: '{recorded_ipa}' -> {ipa_units_recorded}")
    
    # Compare position by position - each position in real IPA gets compared
    result = []
    real_idx = 0
    recorded_idx = 0
    
    while real_idx < len(ipa_units_real):
        if recorded_idx >= len(ipa_units_recorded):
            result.append('0')
            real_idx += 1
        elif ipa_units_recorded[recorded_idx] == '_':
            result.append('0')
            real_idx += 1
            recorded_idx += 1
        elif ipa_units_real[real_idx] == ipa_units_recorded[recorded_idx]:
            result.append('1')
            real_idx += 1
            recorded_idx += 1
        else:
            result.append('0')
            real_idx += 1
            recorded_idx += 1
    
    return ''.join(result)

def compare_ipa_pairs(real_and_transcribed_words_ipa):
    results = []
    for real, recorded in real_and_transcribed_words_ipa:
        results.append(ipa_compare(real, recorded))
    return results

# Test với dữ liệu mới
test_cases = [('hələʊ', 'meɪloʊ'), ('wɔːtə', '_wɔ_zʌ')]
result = compare_ipa_pairs(test_cases)
print(f"\nFinal Result: {result}")

