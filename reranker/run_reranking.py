import argparse
import jsonlines
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

def read_jsonl(path):
    data = []
    with jsonlines.open(path, mode='r') as reader:
        for line in reader:
            data.append(line)
    return data

def write_jsonl(path, data):
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(data)
    print(f"Written output to {path}")

class RerankerModel(torch.nn.Module):
    def __init__(self, model_name, device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name.lower()

        # Some known token IDs used by monot5/rankt5
        self.monot5_true_false_tokens = [1176, 6136]  # "▁true", "▁false"
        self.rankt5_extra_id_10 = 32089              # <extra_id_10>

        if 'rankt5' in self.model_name:
            self.model_type = 'rankt5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'monot5' in self.model_name:
            self.model_type = 'monot5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model_type = 'sequence_classifier'
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, batch):
        if self.model_type == 'monot5':
            output = self.model.generate(**batch, max_length=2, return_dict_in_generate=True, output_scores=True)
            scores_tensor = torch.stack(output.scores)
            log_probs = torch.nn.functional.log_softmax(scores_tensor[0][:, self.monot5_true_false_tokens], dim=1)
            scores = log_probs[:, 0].tolist() 
        
        elif self.model_type == 'rankt5':
            output = self.model.generate(**batch, max_length=2, return_dict_in_generate=True, output_scores=True)
            scores_tensor = torch.stack(output.scores)
            scores = scores_tensor[0][:, self.rankt5_extra_id_10].tolist()
        
        else:
            logits = self.model(**batch).logits
            if logits.shape[1] == 1:
                scores = logits.squeeze(dim=1).tolist()
            else:
                scores = logits[:, 0].tolist()
        
        return scores

    def score_pairs(self, pairs, batch_size, max_length):
        if self.model_type == 'monot5':
            input_texts = [f"Query: {q} Document: {d} Relevant:" for (q, d) in pairs]
            encodings = self.tokenizer(
                input_texts, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length
            ).to(self.device)
        elif self.model_type == 'rankt5':
            input_texts = [f"Query: {q} Document: {d}" for (q, d) in pairs]
            encodings = self.tokenizer(
                input_texts, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length
            ).to(self.device)
        else:
            q_texts = [p[0] for p in pairs]
            d_texts = [p[1] for p in pairs]
            encodings = self.tokenizer(
                q_texts, d_texts, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length
            ).to(self.device)

        scores = []
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device=='cuda')):
                for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring"):
                    end = start + batch_size
                    batch_encodings = {k: v[start:end] for k, v in encodings.items()}
                    batch_scores = self.forward(batch_encodings)
                    scores.extend(batch_scores)

        return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True,
                        help="Reranker model name or path (e.g. 'castorini/monot5-base-msmarco-10k', 'Soyoung97/RankT5-base', or 'cross-encoder/ms-marco-MiniLM-L-12-v2').")
    parser.add_argument('--input_path', required=True, help="Path to input JSONL.")
    parser.add_argument('--output_path', required=True, help="Path to output JSONL.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for scoring.")
    parser.add_argument('--topk', type=int, default=100, help="Number of top passages to re-rank.")
    parser.add_argument('--max_input_length', type=int, default=512, help="Maximum input length for tokenization.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (e.g., 'cuda', 'cpu').")
    args = parser.parse_args()

    query_key = "query"
    passages_key = "passages"
    text_key = "text"
    score_key = "score"

    data = read_jsonl(args.input_path)

    reranker = RerankerModel(model_name=args.model_name, device=args.device)

    output_data = []
    for entry in tqdm(data, desc="Re-ranking"):
        query = entry[query_key]
        passages = entry[passages_key]
        passages = passages[:args.topk]
        pairs = [(query, p[text_key]) for p in passages]
        scores = reranker.score_pairs(pairs, batch_size=args.batch_size, max_length=args.max_input_length)

        for p, s in zip(passages, scores):
            p[score_key] = s
        passages = sorted(passages, key=lambda x: x[score_key], reverse=True)

        entry[passages_key] = passages
        output_data.append(entry)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_path, output_data)
    print("Done!")

if __name__ == "__main__":
    main()
