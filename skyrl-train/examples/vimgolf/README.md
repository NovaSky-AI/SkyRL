# VimGolf single-turn, multi-turn and gym env examples

This example shows how to train a LLM to play VimGolf.

It registers three different environments:

- `vimgolf-single-turn`
- `vimgolf-multi-turn`
- `vimgolf-gym`

## How the VimGolf environment works

### Single-turn example

The model can only produce one command per turn, for one chance. The command is evaluated and the reward is binary (1 if the command is correct, 0 otherwise).

### Multi-turn example

The model can produce multiple commands per turn, for multiple chances (up to `max_turns`). The commands are evaluated and the reward in every turn is computed with the following process:

- If the command is correct, the reward is 1.
- If the command is not given, the reward is 0.
- If the command is incorrect, the buffer in the Vim editor is extracted, and the reward is computed using the Levenshtein distance between the current buffer, the input buffer and the target buffer, using the following code:
    ```python
    d_io = levenshtein.distance(input_text, output_text)
    d_ib = levenshtein.distance(input_text, buffer_str)
    d_bo = levenshtein.distance(buffer_str, output_text)
    input_score = 1 - (d_io - d_ib) / d_io
    input_score = clip_value(input_score, 0, 1)
    output_score = (d_io - d_bo) / d_io
    output_score = clip_value(output_score, 0, 1)
    score = input_score * output_score
    ```

### Gym env example

The model can interact with VimGolf using TermExec, a terminal environment for AI.

The model is required to produce exactly one UTF-8 string with ANSI escape sequences, wrapped within `<termexec>` and `</termexec>` tags.

The environment will provide feedback in different circumstances:

- If the model does not produce a valid command, some error message will be given with role `system`.
- If the model produces a valid command, the environment will provide feedback in the following format (in which `edit_distance_score` is similar to the one used in multi-turn environment):
  ```python
  terminal_feedback = f"""
  You have performed a termexec action, now you have the feedback from the terminal:

  Terminal Screen:

  {terminal_screen}

  Estimated edit distance score (1 for perfect match, 0 for worst match): {edit_distance_score}
  """
  ```

The reward is calculated as follows:

- If the model produces a valid command, the reward is 0.5
- If the model solves the challenge, the reward is 1
- If the model does not produce a valid command, the reward is 0

## How to run the examples

### Generate the dataset

This converts the builtin VimGolf public challenges into parquet files with input text, output text, detail and challenge id.

```bash
# Adjust output_dir, train_ratio and random_seed as needed
uv run examples/vimgolf/vimgolf_dataset.py \
  --output_dir "$HOME/data/vimgolf" \
  --train_ratio 0.8 \
  --random_seed 42
```

Outputs:
- `$HOME/data/vimgolf/train.parquet`
- `$HOME/data/vimgolf/validation.parquet`

If you change `--output_dir`, update the `DATA_DIR` variable in the run script below accordingly.

### Launch training

Modify training config parameters in the training script as needed. Commonly modified parameters are: `NUM_GPUS`, `LOGGER`, and `INFERENCE_BACKEND`.

Then run the training script:

```bash
bash examples/vimgolf/run_vimgolf_single_turn.sh
bash examples/vimgolf/run_vimgolf_multi_turn.sh
bash examples/vimgolf/run_vimgolf_gym.sh
```