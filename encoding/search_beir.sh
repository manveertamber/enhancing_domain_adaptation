model=scifact_bge_embedding_model
dataset=scifact

python -m pyserini.search.faiss \
  --threads 16 --batch-size 512 \
  --encoder-class auto \
  --encoder models/${model} --l2-norm --query-prefix "Represent this sentence for searching relevant passages:" \
  --index indices/models_${model}_${dataset}_index \
  --topics beir-v1.0.0-${dataset}-test \
  --output run.${model}.${dataset}.txt \
  --hits 1000  --remove-query

python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 beir-v1.0.0-${dataset}-test \
  run.${model}.${dataset}.txt

python -m pyserini.eval.trec_eval \
  -c -m recall.100 beir-v1.0.0-${dataset}-test \
  run.${model}.${dataset}.txt