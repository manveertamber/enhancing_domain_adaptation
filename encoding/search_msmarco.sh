model=msmarco_snowflake_embedding_model

python -m pyserini.search.faiss \
  --threads 16 --batch-size 512 \
  --encoder-class auto \
  --encoder models/${model} --l2-norm --query-prefix "Represent this sentence for searching relevant passages:" \
  --index indices/models_${model}_msmarco_index \
  --topics dl19-passage \
  --output run.${model}.dl19.txt \
  --hits 1000

python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage \
  run.${model}.dl19.txt 
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
  run.${model}.dl19.txt
python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
  run.${model}.dl19.txt  
python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
  run.${model}.dl19.txt 

python -m pyserini.search.faiss \
  --threads 16 --batch-size 512 \
  --encoder-class auto \
  --encoder models/${model} --l2-norm --query-prefix "Represent this sentence for searching relevant passages:" \
  --index indices/models_${model}_msmarco_index  \
  --topics dl20 \
  --output run.${model}.dl20.txt  \
  --hits 1000

python -m pyserini.eval.trec_eval -c -l 2 -m map dl20-passage \
  run.${model}.dl20.txt 
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
  run.${model}.dl20.txt
python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
  run.${model}.dl20.txt  
python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl20-passage \
  run.${model}.dl20.txt 
