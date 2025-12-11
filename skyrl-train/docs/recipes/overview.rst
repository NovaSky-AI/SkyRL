E2E Recipes with SkyRL
======================

We provide a collection of end-to-end recipes for single and multi-turn RL training with SkyRL.

We provide reproduction runs for the following recipes:

1. Simple training on GSM8K 
2. `Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) <https://dapo-sia.github.io/>`_ 
3. `SkyRL-SQL <https://novasky-ai.notion.site/skyrl-sql>`_ 
4. `SearchR1 <https://arxiv.org/abs/2503.09516>`_ 



Simple training on GSM8K
~~~~~~~~~~~~~~~~~~~~~~~~~~

The scripts for training on GSM8K are available at :code_link:`examples/gsm8k/`.


.. raw:: html

   <style>
     table.skytable {
       border-collapse: collapse;
       margin-bottom: 20px;
     }
     table.skytable th, table.skytable td {
       border: 1px solid #ccc;
       padding: 6px 10px;
     }
     table.skytable th {
       background: #f2f2f2;
     }
     table.skytable tr:nth-child(even) {
       background: #fafafa;
     }
   </style>

   <table class="skytable">
     <thead>
       <tr>
         <th>Backend</th>
         <th>Model</th>
         <th>Eval Accuracy</th>
         <th>Hardware</th>
         <th>Training Steps</th>
         <th>Commit</th>
         <th>WandB</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td>FSDP2</td>
         <td>Qwen/Qwen2.5-1.5B-Instruct</td>
         <td>0.796</td>
         <td>4xH100</td>
         <td>140</td>
         <td><a href="https://github.com/novasky-ai/SkyRL/commit/a95b699">a95b699</a></td>
         <td><a href="https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k_recipes?nw=nwusersumanthrh99">Link</a></td>
       </tr>
       <tr>
         <td>DeepSpeed</td>
         <td>Qwen/Qwen2.5-1.5B-Instruct</td>
         <td>0.791</td>
         <td>4xH100</td>
         <td>140</td>
         <td><a href="https://github.com/novasky-ai/SkyRL/commit/a95b699">a95b699</a></td>
         <td><a href="https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k_recipes?nw=nwusersumanthrh99">Link</a></td>
       </tr>
     </tbody>
   </table>
  

DAPO Recipes
~~~~~~~~~~~~

The code for the DAPO recipe is available at :code_link:`examples/algorithms/dapo/`.


.. raw:: html

   <style>
     table.skytable {
       border-collapse: collapse;
       margin-bottom: 20px;
     }
     table.skytable th, table.skytable td {
       border: 1px solid #ccc;
       padding: 6px 10px;
     }
     table.skytable th {
       background: #f2f2f2;
     }
     table.skytable tr:nth-child(even) {
       background: #fafafa;
     }
   </style>

   <table class="skytable">
     <thead>
       <tr>
         <th>Recipe</th>
         <th>Model</th>
         <th>AIME24 Pass@32</th>
         <th>AIME24 Avg Score</th>
         <th>Hardware</th>
         <th>Training Steps</th>
         <th>Commit</th>
         <th>WandB</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td>DAPO (w/o Dynamic Sampling)</td>
         <td>Qwen/Qwen-2.5-7B-Math</td>
         <td>0.633</td>
         <td>-0.304</td>
         <td>8xH100</td>
         <td>320</td>
         <td><a href="https://github.com/novasky-ai/SkyRL/commit/a95b699">a95b699</a></td>
         <td><a href="https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q">Link</a></td>
       </tr>
       <tr>
         <td>DAPO (w/o Dynamic Sampling)</td>
         <td>Qwen/Qwen3-1.7B</td>
         <td>0.4</td>
         <td>-0.702</td>
         <td>8xH100</td>
         <td>225</td>
         <td><a href="https://github.com/novasky-ai/SkyRL/commit/a95b699">a95b699</a></td>
         <td><a href="https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q">Link</a></td>
       </tr>
       <tr>
         <td>DAPO (w/o Dynamic Sampling)</td>
         <td>Qwen/Qwen3-4B</td>
         <td>0.6</td>
         <td>-0.51</td>
         <td>8xH100</td>
         <td>90</td>
         <td><a href="https://github.com/novasky-ai/SkyRL/commit/a95b699">a95b699</a></td>
         <td><a href="https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q">Link</a></td>
       </tr>
     </tbody>
   </table>

SkyRL-SQL Recipes
~~~~~~~~~~~~~~~~~


For more details, please refer to :doc:`../recipes/skyrl-sql`.

We provide two reference runs: single-turn and multi-training for `Qwen/Qwen2.5-Coder-7B-Instruct <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct>`_ (run on 1 8xH100 node up to convergence), with the WandB report `here <https://wandb.ai/sky-posttraining-uc-berkeley/skyrl-sql/reports/SkyRL-SQL---VmlldzoxMzM0MTAyMw?accessToken=vrqncoa32qcobvvpuo672yji4gweguk6tjxvaflk1zh73fn70j6l5rj8j619uvry>`_.

The evaluation results are shown below (using the evaluation code `here <https://github.com/lynnliu030/OmniSQL/tree/main/evaluate_skysql>`_):

.. raw:: html

   <style>
     table.skytable2 {
       border-collapse: collapse;
       margin-bottom: 20px;
     }
     table.skytable2 th, table.skytable2 td {
       border: 1px solid #ccc;
       padding: 6px 10px;
     }
     table.skytable2 th {
       background: #f2f2f2;
     }
     table.skytable2 tr:nth-child(even) {
       background: #fafafa;
     }
   </style>
   <div style="width: 80%;">
   <table class="skytable2">
     <thead>
       <tr>
         <th>Eval Turns (Train)</th>
         <th>Training Method</th>
         <th>Spider-Dev</th>
         <th>Spider-Test</th>
         <th>Spider-Realistic</th>
         <th>Spider-DK</th>
         <th>Spider-Syn</th>
         <th>Avg</th>
         <th>WandB</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td>1</td>
         <td>Single-Turn</td>
         <td>81.2</td>
         <td>83.8</td>
         <td>76.8</td>
         <td>67.9</td>
         <td>70.1</td>
         <td>76.0</td>
         <td><a href="https://wandb.ai/sky-posttraining-uc-berkeley/skyrl-sql/reports/SkyRL-SQL---VmlldzoxMzM0MTAyMw">Link</a></td>
       </tr>
       <tr>
         <td>1</td>
         <td>Multi-Turn</td>
         <td>82.4 (+1.2%)</td>
         <td>83.7 (-0.1%)</td>
         <td>80.3 (+3.5%)</td>
         <td>70.5 (+2.6%)</td>
         <td>71.2 (+1.1%)</td>
         <td>77.6 (+1.6%)</td>
         <td><a href="https://wandb.ai/sky-posttraining-uc-berkeley/skyrl-sql/reports/SkyRL-SQL---VmlldzoxMzM0MTAyMw">Link</a></td>
       </tr>
       <tr>
         <td>5</td>
         <td>Single-Turn</td>
         <td>79.5</td>
         <td>82.2</td>
         <td>77.6</td>
         <td>65.6</td>
         <td>68.4</td>
         <td>74.7</td>
         <td><a href="https://wandb.ai/sky-posttraining-uc-berkeley/skyrl-sql/reports/SkyRL-SQL---VmlldzoxMzM0MTAyMw">Link</a></td>
       </tr>
       <tr>
         <td>5</td>
         <td>Multi-Turn</td>
         <td>83.9 (+4.4%)</td>
         <td>85.2 (+3%)</td>
         <td>81.1 (+3.5%)</td>
         <td>72.0 (+6.4%)</td>
         <td>73.7 (+5.3%)</td>
         <td>79.2 (+4.5%)</td>
         <td><a href="https://wandb.ai/sky-posttraining-uc-berkeley/skyrl-sql/reports/SkyRL-SQL---VmlldzoxMzM0MTAyMw">Link</a></td>
       </tr>
     </tbody>
   </table>
   </div>


SearchR1 Recipes
~~~~~~~~~~~~~~~~

For more details, please refer to :doc:`../recipes/searchr1`.


The WandB report is available `here <https://api.wandb.ai/links/sky-posttraining-uc-berkeley/5kvkzdzr>`_.

Qwen/Qwen2.5-3B-Instruct
^^^^^^^^^^^^^^^^^^^^^^^^
The evaluation results are shown below for `Qwen/Qwen2.5-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>`_, with all experiments run on 1 8xH100 node up to convergence (330 training steps).

.. raw:: html

  <table class="eval-table" style="border-collapse: collapse; width: 80%; margin: 20px auto;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th style="border: 1px solid #ccc; padding: 8px;">Dataset</th>
        <th style="border: 1px solid #ccc; padding: 8px;">Search-R1<br>(3 turns)</th>
        <th style="border: 1px solid #ccc; padding: 8px;">SkyRL + SearchR1<br>(2 turns)</th>
        <th style="border: 1px solid #ccc; padding: 8px;">SkyRL + SearchR1<br>(3 turns)</th>
        <th style="border: 1px solid #ccc; padding: 8px;">SkyRL + SearchR1<br>(4 turns)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">NQ†</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.397</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.455</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.449</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.449</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">TriviaQA†</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.565</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.613</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.616</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.611</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">PopQA†</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.391</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.447</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.444</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.435</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">HotpotQA*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.331</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.334</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.417</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.407</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">2wiki*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.310</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.313</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.396</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.403</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">Musique*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.124</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.086</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.179</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.163</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">Bamboogle*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.232</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.242</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.448</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.352</td>
      </tr>
      <tr style="background-color: #f2f2f2; font-weight: bold;">
        <td style="border: 1px solid #ccc; padding: 8px;">Average</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.336</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.356</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.421</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.403</td>
      </tr>
    </tbody>
  </table>

Qwen/Qwen3-30B-A3B
^^^^^^^^^^^^^^^^^^^
Evaluation results for `Qwen3-30B-A3B <https://huggingface.co/Qwen/Qwen3-30B-A3B>`_ on SearchR1, with experiments using 4 8xH100 nodes using the :doc:`Megatron <../examples/megatron>` backend, are shown below. These results can be reproduced with commit `9b878cd <https://github.com/NovaSky-AI/SkyRL/commit/9b878cdfe133b0ffe8a827f0ef91c63341f99cf6>`_.

.. raw:: html

  <table class="eval-table" style="border-collapse: collapse; width: 80%; margin: 20px auto;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th style="border: 1px solid #ccc; padding: 8px;">Dataset</th>
        <th style="border: 1px solid #ccc; padding: 8px;">SkyRL + SearchR1<br>(4 turns)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">NQ†</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.463</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">TriviaQA†</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.664</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">PopQA†</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.448</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">HotpotQA*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.412</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">2wiki*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.361</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">Musique*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.178</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ccc; padding: 8px;">Bamboogle*</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.488</td>
      </tr>
      <tr style="background-color: #f2f2f2; font-weight: bold;">
        <td style="border: 1px solid #ccc; padding: 8px;">Average</td>
        <td style="border: 1px solid #ccc; padding: 8px;">0.457</td>
      </tr>
    </tbody>
  </table>

