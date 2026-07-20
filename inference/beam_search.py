import torch

class BeamSearch:
    """
    Implements beam search for generation.
    """
    def __init__(self, model, beam_width, max_length):
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length

    @torch.no_grad()
    def search(self, input_ids):
        """
        Runs beam search starting from input_ids.
        """
        # beams is a list of tuples: (score, sequence)
        beams = [(0.0, input_ids)]
        
        for _ in range(self.max_length):
            all_candidates = []
            
            for score, seq in beams:
                # Get logits for the last token
                logits = self.model(seq)
                next_token_logits = logits[..., -1, :]
                
                # Get log probabilities
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                
                # Get top k candidates for this beam
                topk_log_probs, topk_indices = torch.topk(log_probs, self.beam_width, dim=-1)
                
                for i in range(self.beam_width):
                    candidate_score = score + topk_log_probs[..., i].item()
                    candidate_seq = torch.cat([seq, topk_indices[..., i:i+1]], dim=-1)
                    all_candidates.append((candidate_score, candidate_seq))
            
            # Order all candidates by score
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            
            # Select top k
            beams = ordered[:self.beam_width]
            
        return beams[0][1] # Return the best sequence
