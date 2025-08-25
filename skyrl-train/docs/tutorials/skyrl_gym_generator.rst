SkyRLGymGenerator: Multi-turn Tokenization and Token-in-Token-out
=================================================================

Last updated: 2025-08-22

This document explains how ``SkyRLGymGenerator`` manages the chat history and tokens for both
single-turn and multi-turn rollouts, and how token-in-token-out (TI/TO) is enforced.

Overview
--------

``SkyRLGymGenerator`` is an implementation of the ``GeneratorInterface``, where we use SkyRL Gym for
the environment of the rollouts. If you would like to use other environments, you can write your
own Generator by extending ``GeneratorInterface``.

A ``SkyRLGymGenerator`` uses an ``InferenceEngineClient`` like an LLM endpoint to generate response,
ultimately returning ``GeneratorOutput`` (including ``response_ids`` and ``loss_masks``) to
the training loop for updating the model.

``SkyRLGymGenerator`` is implemented to enforce token-in-token-out (TI/TO) in most cases. To see
what TI/TO is and why it is important, please refer to `issue #123 <https://github.com/NovaSky-AI/SkyRL/issues/123>`_.

To implement ``GeneratorInterface.generate()``, ``SkyRLGymGenerator`` implements ``generate_batched()``
for single-turn generation, and ``agent_loop()`` for multi-turn generation.

Single-turn generation
----------------------

``SkyRLGymGenerator.generate_batched()`` is used when ``config.generator.batched`` is set to ``True``.
In this case, only a single assistant message is generated for each prompt. It is used for tasks
such as math problems, where the model is expected to generate a single response without interacting
with the environment. We pass a list of prompts to each invocation of the underlying LLM engine's
``.generate()`` method (hence "batched").

Multi-turn generation
---------------------

``SkyRLGymGenerator.agent_loop()`` is used when ``config.generator.batched`` is set to ``False``, where
the model is expected to interact with the environment for multiple turns (though ``agent_loop()`` can
also be used for single-turn generation). We pass a single prompt to each invocation of the underlying
LLM engine's ``.generate()`` method.

There are three distinct codepaths in ``agent_loop()``, each managing the chat history and tokens
differently. The codepaths are determined by:

- Config's ``generator.use_conversation_multi_turn``: If ``False``, all turns' observations and assistant
  generations are stored in the same assistant message. If ``True``, each observation from the
  environment's ``step()`` is a message.
- Whether ``get_custom_chat_template(model_name)`` returns a template (currently used for Qwen3).

The three codepaths are:

1) Multi-turn conversation with chat template (the more conventional way)

   - Enabled when ``use_conversation_multi_turn == True`` and ``custom_chat_template is None``.
   - Each observation is a turn of message following the model's chat template, appending
     LLM-generated response and observations as raw tokens to maintain TI/TO, but requires the
     fixed-base approach to ensure the raw tokens follow the model's chat template correctly.
   - TI/TO: enforced.
   - Example with Qwen2.5 chat template:

.. code-block:: python

  <|im_start|>system
  System prompt here<|im_end|>
  <|im_start|>user
  Question here<|im_end|>
  <|im_start|>assistant
  Response1<|im_end|>
  <|im_start|>user
  Observation1<|im_end|>
  <|im_start|>assistant
  Response2<|im_end|>
  <|im_start|>user
  Observation2<|im_end|>
  ...

2) Single assistant-message for all turns (single-turn chat template)

   - Enabled when ``use_conversation_multi_turn == False``.
   - Keep an entire multi-step interaction inside a single assistant message, appending
     LLM-generated response and observations as raw tokens to maintain TI/TO.
   - TI/TO: enforced.
   - Example with Qwen2.5 chat template:

.. code-block:: python

  <|im_start|>system
  System prompt here<|im_end|>
  <|im_start|>user
  Question here<|im_end|>
  <|im_start|>assistant
  Response1
  <observation>Observation1</observation>
  Response2
  <observation>Observation2</observation>
  ...<|im_end|>

3) Always re-tokenize full chat history (no TI/TO)

   - Enabled when ``use_conversation_multi_turn == True`` and a ``custom_chat_template`` is present.
   - Serve models like Qwen3 that require special handling (e.g., strip non-last-turn thinking
     tokens). We can also get ``[assistant_masks]`` and ``[input_ids]`` from the final tokenized chat
     history with the help of ``{% generation %}`` and ``{% endgeneration %}`` tags in the jinja template.
   - Chat history is maintained as string messages and re-tokenized every turn and
     at the end to obtain ``assistant_masks`` and final ``response_ids``.
   - TI/TO: NOT enforced

.. note::

  Currently, the Qwen3 model by default follows the 3rd codepath (i.e. TI/TO is not enforced). That
  is, this codepath rolls out Qwen3 by following the inference chat template, and returns only the
  last-turn thinking tokens to Generator for the training pipeline.

  It is debatable whether this is the best method to train Qwen3. We will soon add a configuration flag
  that only directs to codepath 3 when turned on. Otherwise when false, Qwen3 will follow the
  first two codepaths and save all thinking tokens to the training pipeline.

Multi-turn Tokenization and TI/TO
---------------------------------

In this section, we elaborate how TI/TO is enforced in the multi-turn generation, specifically
for the first codepath. TI/TO for the second codepath is simple since we keep appending the
generated tokens to the same message and hence do not need to worry about the chat templating
between messages. The third codepath does not enforce TI/TO.

In codepath 1, the agent loop does the following:
  1. Tokenize dataset's prompt to initialize ``input_ids``
  2. Feed ``input_ids`` to LLM engine, get ``output_ids`` out
  3. ``input_ids += output_ids`` (a.k.a. token-in-token-out) -- the next turn's input IDs are precisely what the LLM generated
  4. Tokenize observations got from SkyRL-Gym's environment output (i.e. ``env.step()``), and append to ``input_ids``
  5. Repeat 2-4 until ``env.step()`` marks done

To correctly tokenize the observations in step 4, we follow the fixed-base approach described in
`this blog <https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach>`_.

Specifically, we instantiate a ``base_conversation`` that we never change ("fixed base"):

.. code-block:: python
  
  self.base_conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "I am an assistant."},
  ]
  self.base_conversation_token_ids = tokenizer.apply_chat_template(
    self.base_conversation,
    add_generation_prompt=False,
    tokenize=True,
  )

When we get new observations ``new_obs``, which is a list of ``{role: str, content: str}``, we
convert them to token IDs while following the model's chat template by:

.. code-block:: python

  observation_ids = self.tokenizer.apply_chat_template(
    [*self.base_conversation, *new_obs],
    add_generation_prompt=True,
    tokenize=True,
  )[len(self.base_conversation_token_ids) :]
  input_ids += observation_ids
  loss_mask += [0] * len(observation_ids)

One tricky part is that, for some models, there are tokens after the last EOS token for a turn of
message. For instance, in Qwen2.5 and Qwen3, the ``base_conversation_token_ids`` are equivalent to:

.. code-block:: python

  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
  <|im_start|>user\nI am a user.<|im_end|>\n
  <|im_start|>assistant\nI am an assistant.<|im_end|>\n

Note that there is a ``\n`` in the assistant's message before the next user's message starts.
If we do token-in-token-out, there is no way for the LLM engine to generate ``\n`` since the
EOS token is ``<|im_end|>``. Therefore, we need to add the ``\n`` back when creating ``observation_ids``.
In order to do this, we cut the ``\n`` out in ``base_conversation_token_ids``:

.. code-block:: python

  if self.tokenizer.eos_token_id in self.base_conversation_token_ids:
      last_eos_token_index = (
          len(self.base_conversation_token_ids)
          - 1
          - self.base_conversation_token_ids[::-1].index(self.tokenizer.eos_token_id)
      )
      self.base_conversation_token_ids = self.base_conversation_token_ids[: last_eos_token_index + 1]


This way, ``observation_ids`` will be ``\n<|im_start|>user\nObservation here<|im_end|>\n`` (note the
very first ``\n`` that makes up the former assistant's ``\n``). The ``\n`` at the **final** assistant
turn will still be missing, but this is fine.
You can see ``tests/cpu/generators/test_skyrl_gym_generator_chat_templating.py`` for more details.


References
----------

- https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
