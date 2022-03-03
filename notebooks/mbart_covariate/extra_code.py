class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    :class:`~transformers.LogitsProcessor` that enforces the specified token as the first generated token.

    Args:
        bos_token_id (:obj:`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if isinstance(self.bos_token_id, int):
            self.bos_token_id = torch.tensor([self.bos_token_id]).to(input_ids.device)
        
        num_repeat = int(input_ids.shape[0]/self.bos_token_id.shape[0])

        expanded_return_idx = (
            torch.arange(self.bos_token_id.shape[0]).view(-1, 1).repeat(1, num_repeat).view(-1).to(self.bos_token_id.device)
        )
        self.bos_token_id = self.bos_token_id.index_select(0, expanded_return_idx)
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, :] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


@staticmethod
def _expand_inputs_for_generation(
    input_ids: torch.LongTensor,
    expand_size: int = 1,
    is_encoder_decoder: bool = False,
    attention_mask: torch.LongTensor = None,
    encoder_outputs: ModelOutput = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    if "covariate_ids" in model_kwargs:
        model_kwargs["covariate_ids"] = model_kwargs["covariate_ids"].index_select(0, expanded_return_idx)

    if is_encoder_decoder:
        assert encoder_outputs is not None
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
    return input_ids, model_kwargs
        
        
        