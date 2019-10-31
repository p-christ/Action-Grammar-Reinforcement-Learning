from collections import Counter, defaultdict
import numpy as np

class IGGI(object):
    """Implements the Information-Greedy Grammar Inference (IGGI) method for extracting context-free grammars from sequential data"""
    def __init__(self, end_of_episode_symbol="/", divider_symbol="|"):
        self.end_of_episode_symbol = end_of_episode_symbol
        self.next_rule_name_ix = 0
        self.divider_symbol = divider_symbol
        self.new_symbol_to_use = "A"
        self.rules_created = {}

    def increment_new_symbol_to_use(self):
        """Update the next symbol to use"""
        self.new_symbol_to_use = chr(ord(self.new_symbol_to_use) + 1)
        if self.new_symbol_to_use == "\\":
            self.new_symbol_to_use = "]"
        print("New symbol ", self.new_symbol_to_use)
        assert self.new_symbol_to_use != "|", self.new_symbol_to_use

    def generate_action_grammar(self, actions, new_symbol_to_use, append_old_rules=True, max_rule_length=float("inf")):
        """Generates a grammar given a list of actions"""
        assert isinstance(actions, list), actions
        assert not isinstance(actions[0], list), "Should be 1 long list of actions - {}".format(actions[0])
        assert len(actions) > 0, "Need to provide a list of at least 1 action"
        assert isinstance(actions[0], int), "The actions should be integers"
        assert self.new_symbol_to_use not in actions
        self.max_rule_length = max_rule_length
        if append_old_rules:
            for rule, values in self.rules_created.items():
                print("Rule: {} -- {}".format(rule, values))
                extra_info = [self.divider_symbol] + list(values)
                actions.extend(extra_info)
                print("New actions ", actions)
        self.new_symbol_to_use = new_symbol_to_use
        rules = {}
        min_change, min_change_substring, min_change_new_string = self.calculate_max_info_reduction_rule(actions)
        print("First min change {} -- {}".format(min_change, min_change_substring))
        while min_change < 0.0:
            print("Min change {} -- Min change substring {} -- Min change new string".format(min_change, min_change_substring, min_change_new_string))
            actions = min_change_new_string
            rules[self.new_symbol_to_use] = min_change_substring
            self.new_symbol_to_use += 1
            min_change, min_change_substring, min_change_new_string = self.calculate_max_info_reduction_rule(actions)
            print("New Min change {} -- Min change substring {} -- Min change new string".format(min_change,
                                                                                             min_change_substring,
                                                                                             min_change_new_string))
        print("{} Rules: {}".format(len(rules), rules))
        self.rules_created.update(rules)
        return rules

    def calculate_max_info_reduction_rule(self, string_list):
        """Calculates the rule that will lead to biggest negative reduction in inofrmation content I"""
        if self.divider_symbol in string_list:
            divider_index = string_list.index(self.divider_symbol)
        else:
            divider_index = len(string_list)
        main_string = string_list[:divider_index]

        substring_counts = self.count_all_repeating_substrings(main_string)
        min_change = float("inf")
        min_change_substring = ""
        min_change_new_string = None
        for substring, count in substring_counts.items():
            if count >= 2 and len(substring) <= self.max_rule_length:
                change_I, new_string = self.calculate_change_in_information_content(string_list, substring)
                # print("Change I {} -- Substring {} -- Count {}".format(change_I, substring, count))
                if change_I < min_change:
                    min_change = change_I
                    min_change_substring = substring
                    min_change_new_string = new_string

        return min_change, min_change_substring, min_change_new_string

    def calculate_change_in_information_content(self, string, substring):
        """Calculates the change in information content from replacing the substring with a new symbol"""
        old_info = self.calculate_information_content(string)
        new_string = self.calculate_new_string(string, substring)
        new_info = self.calculate_information_content(new_string)
        change_in_info = new_info - old_info
        return change_in_info, new_string

    def calculate_new_string(self, string_list, substring):
        """Takes in a string list and a substring and replaces the substring with a new symbol in the part of the string to
        the left hand side of the first self.divider_symbol. Then it adds a self.divider_symbol and the substring to the end
        e.g. string = "abab|cde" and substring "ab" -->  "AA|cde|ab"""
        assert isinstance(substring, tuple)
        assert isinstance(string_list, list)
        new_string = []
        length_substring = len(substring)
        skip = 0
        seen_divider = False
        for ix in range(len(string_list)):
            if skip > 0:
                skip -= 1
                continue
            if string_list[ix] == self.divider_symbol:
                seen_divider = True
            if ix + length_substring <= len(string_list) and tuple(string_list[ix:ix+length_substring]) == substring and not seen_divider:
                new_string.append(self.new_symbol_to_use)
                skip = length_substring - 1
            else:
                new_string.append(string_list[ix])
        new_string.append(self.divider_symbol)
        new_string.extend(substring)
        return new_string

    def calculate_information_content(self, string):
        """Calculates the total information content in a string which is calculated as:
        I = n log_2(n) - sum(n_i log_2(n_i)). The higher the number the higher the entropy"""
        symbol_count = Counter(string)
        string_length = len(string)
        symbol_score = [count * np.log2(count)  for symbol, count in symbol_count.items()]
        I = string_length * np.log2(string_length) - np.sum(symbol_score)
        assert I >= 0.0
        return I

    def count_all_repeating_substrings(self, string_list):
        """Takes in a list of strings and counts how many times each possible substring appears. It doesn't include counts of the
        self.divider_symbol and also substrings cannot go across that symbol"""
        substring_count = defaultdict(lambda: 0)
        last_substring_ix_range = {} #to help make sure there isn't overlap in substrings
        min_start_ix = -1
        for end_ix in range(len(string_list)):
            if string_list[end_ix] == self.divider_symbol:
                break
            if string_list[end_ix] == self.end_of_episode_symbol:
                min_start_ix = end_ix
                continue
            for start_ix in range(min_start_ix + 1, end_ix + 1):
                substring = tuple(string_list[start_ix:end_ix+1])
                if substring in last_substring_ix_range.keys():
                    last_range = last_substring_ix_range[substring]
                    last_range = range(last_range[0], last_range[1])
                    new_range = range(start_ix, end_ix+1)
                    if len(set(last_range).intersection(new_range)) > 0:
                        # There is overlap so we cannot count this occurance
                        continue
                substring_count[substring] += 1
                last_substring_ix_range[substring] = [start_ix, end_ix + 1]
        return substring_count




