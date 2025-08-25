#!/usr/bin/env python3
"""
Test script using real ReasoningGym datasets.

This script demonstrates how to use the ReasoningGymDataset class
with actual ReasoningGym procedurally generated data.
"""
from dataset import ReasoningGymDataset, make_reasoning_gym_dataset
import reasoning_gym

def test_spiral_matrix():
    dataset = reasoning_gym.create_dataset('spiral_matrix', size=5, seed=42)
    reasoning_gym_dataset = ReasoningGymDataset(
        procedural_dataset=dataset,
        developer_prompt="You are a helpful assistant. Solve the problem step by step and provide the final answer inside <answer> tags."
    )
    example = reasoning_gym_dataset[0]
    print("Question:", example["extra_info"]["question"])
    print("Ground Truth:", example["reward_spec"]["ground_truth"])

    partial_output = "Following spiral order: 3 1 3 9 8 0 1 2 4. Final output: <answer>3 1 3 9 8 0 1 2 4</answer>"

    print("Note: score_generation method has been moved to the environment class")

def test_leg_counting():
    dataset = reasoning_gym.create_dataset('leg_counting', size=10, seed=42)
    reasoning_gym_dataset = ReasoningGymDataset(
        procedural_dataset=dataset,
        developer_prompt="You are a helpful assistant. Solve the problem step by step and provide the final answer inside <answer> tags."
    )
    example = reasoning_gym_dataset[0]
    print("Question:", example["extra_info"]["question"])
    print("Ground Truth:", example["reward_spec"]["ground_truth"])
    partial_output = "The total number of legs is 99. Final answer: <answer>99</answer>"
    print("score_generation: ", reasoning_gym_dataset.score_generation(0, partial_output))

def test_letter_jumble():
    dataset = reasoning_gym.create_dataset('letter_jumble', size=3, seed=42)
    reasoning_gym_dataset = ReasoningGymDataset(
        procedural_dataset=dataset,
        developer_prompt="Please unscramble the words into a readable sentence and wrap your final answer in <answer> tags."
    )
    
    example = reasoning_gym_dataset[0]
    print("Question:", example["extra_info"]["question"])
    print("Ground Truth:", example["reward_spec"]["ground_truth"])
    
    partial_output = "Here is the unscrambled sentence: <answer>we are be able to produce</answer>"
    score = reasoning_gym_dataset.score_generation(0, partial_output)
    print("Score Generation:", score)


if __name__ == "__main__":
    test_leg_counting()
    test_letter_jumble()
    test_spiral_matrix()