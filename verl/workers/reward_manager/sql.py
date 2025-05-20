
import os 
from verl import DataProto
import torch
import statistics
from verl.utils.reward_score.synsql_verifier import sql_compute_score

class SQLRewardManager:
    """The SQL reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, compute_score) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score or sql_compute_score
        self.config = config
        
        # Path to store the databases to interact with 
        self.db_path = self.config.actor_rollout_ref.rollout.sql.db_path
        
    def get_db_files(self, data):
        db_ids = data.non_tensor_batch['db_id']
        data_srcs = data.non_tensor_batch['data']
        
        # iterate through db_ids and data_srcs together (1-1 correspondence) to get db_files 
        db_files = []
        for db_id, data_src in zip(db_ids, data_srcs):
            if data_src == 'synsql':
                db_files.append(os.path.join(
                    self.db_path,
                    "SynSQL-2.5M/databases",
                    db_id,
                    db_id + ".sqlite"
                ))
            elif data_src == 'spider':
                db_files.append(os.path.join(
                    self.db_path,
                    "spider/database",
                    db_id, 
                    db_id + ".sqlite"
                ))
            elif data_src == 'bird':
                db_files.append(os.path.join(
                    self.db_path,
                    "bird/train/train_databases",
                    db_id,
                    db_id + ".sqlite"
                ))
            else:
                raise NotImplementedError
            
        return db_files
    
    def verify(self, data):
        response_ids = data.batch['responses']
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        prompt_ids = data.batch["prompts"]
        prompts = self.tokenizer.batch_decode(prompt_ids)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        
        db_files = self.get_db_files(data)
        n_agent = data.meta_info.get("n_agent")
        log_dir = data.meta_info.get("log_dir")
        score = self.compute_score(completions=response_str, references=ground_truth,
                              tasks=data.non_tensor_batch['ability'], db_files=db_files, questions=prompts, n_agent=n_agent, log_dir=log_dir)
        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        
        reward_metrics = {}
        for ability in list(set(data.non_tensor_batch['ability'])):
            score_ = [data.batch['acc'][i].item() for i in range(len(data.batch['acc'])) if
                      data.non_tensor_batch['ability'][i] == ability]
            reward_metrics[f'{ability}'] = statistics.mean(score_)
        reward_metrics['all'] = data.batch['acc'].mean().item()

        for i, response_str_ in enumerate(response_str):
            if i >= self.num_examine:
                break
            example = data.batch[i]['input_ids']
            print(self.tokenizer.decode(example, skip_special_tokens=True))
        
        print(f"Score: {score}")
        return score, reward_metrics

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        reward_metrics={}
        verifier_reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
        
        # if the batch already contains evaluation results, the verification is skipped here.
        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)

        for i in range(verifier_reward.shape[0]):
            verifier_reward[i, valid_response_length[i]-1] += verifier_score[i]

        score_dict = {"all": verifier_reward}
        
        return score_dict, reward_metrics
