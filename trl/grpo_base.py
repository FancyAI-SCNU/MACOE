import inspect
import os
import time
import typing
import warnings
from typing import Callable, List, Optional, Union

import torch


from .core import (
    GRPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)

from . import AdaptiveKLController, BaseTrainer, FixedKLController, GRPOConfig



class GRPOTrainer(BaseTrainer):
    """
    The GRPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`GRPOConfig`) -- Configuration object for GRPOTrainer. Check the documentation of `GRPOConfig` for more
         details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(self, config: GRPOConfig):
        """
        Initialize GRPOTrainer.

        Args:
            config (`GRPOConfig`):
                Configuration object for GRPOTrainer. Check the documentation of `GRPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty

        """
        super().__init__(config)

        set_seed(config.seed)
        self.grpo_config = config

        if config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(config.init_kl_coef,
                                               config.target, config.horizon)
        else:
            self.kl_ctl = FixedKLController(config.init_kl_coef)

       

    def compute_rewards(
            self,
            scores: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            ref_logprobs: torch.FloatTensor,
            diff_pos: torch.FloatTensor,
            plent: torch.FloatTensor
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`, `future_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `future_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `future_length`)
        """
        
        kl = self._kl_penalty(logprobs, ref_logprobs)
        kl = torch.sum(kl, dim=-1)
        
        non_score_rewards = -self.kl_ctl.value * kl
        
        scores_min = min(scores)
        scores_max = max(scores)
    
        scores =  2*(scores - scores_min)/(scores_max - scores_min+(1e-8))-1
        
        non_score_rewards_min = min(non_score_rewards)
        non_score_rewards_max = max(non_score_rewards)
        non_score_rewards = 2*(non_score_rewards-non_score_rewards_min)/(non_score_rewards_max-non_score_rewards_min+(1e-8))-1
        
        diff_pos_min = min(diff_pos)
        diff_pos_max = max(diff_pos)
        diff_pos = (diff_pos-diff_pos_min)/(diff_pos_max-diff_pos_min+(1e-8))

        
        rewards = 10.0 * scores + 5* non_score_rewards - 5 * diff_pos-4*plent
        
        
        return rewards, non_score_rewards

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        raise NotImplementedError

    def loss_diffusion(
            self,
            old_logprobs: torch.FloatTensor,
            rewards: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            logprobs_ref: torch.FloatTensor,
            ret_stat: bool = False
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            vpreds (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        kl = self._kl_penalty(logprobs, logprobs_ref)
         
        kl = torch.sum(kl, dim=-1).unsqueeze(0)
        kl = self.kl_ctl.value * kl
         
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        # adv:
        
        mean = rewards.mean()
        std = rewards.std()
        if std!=0.:
            adv = (rewards-mean)/std
        else:
            adv = rewards-mean
        if torch.isnan(adv).any():
            adv = rewards
            advantages = adv
            mask = None
             
            advantages = advantages.detach()
            ratio = torch.exp(logprobs - old_logprobs).sum(-1)
            
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * \
                torch.clamp(ratio, 1.0 - self.config.cliprange,
                            1.0 + self.config.cliprange)
            
            pg_loss = torch.max(pg_losses, pg_losses2)+kl
            loss = pg_loss*0.2
            return pg_loss,loss

        mask = None
         
       
        advantages = adv 
        
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        
        ratio = torch.exp(logprobs - old_logprobs).sum(-1)  # .transpose(0, 1)
        
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * \
            torch.clamp(ratio, 1.0 - self.config.cliprange,
                        1.0 + self.config.cliprange)
        
        pg_loss = torch.max(pg_losses, pg_losses2)+kl
        pg_loss = masked_mean(pg_loss, mask)
       
        loss = (pg_loss+1e-5)
         
        
        return pg_loss,loss
