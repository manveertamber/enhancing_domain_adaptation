dataset='msmarco'
for query_type in 'keywords' 'titles' 'claims' 'questions' 'random' 'msmarco'; do
    python3 -m pyserini.search.faiss \
      --threads 16 --batch-size 8192 \
      --encoder-class auto --encoder Alibaba-NLP/gte-base-en-v1.5 --l2-norm --pooling cls \
      --query-prefix "" \
      --index general_indices/gte-base-en-v1.5_${dataset}_index \
      --topics generated_queries/${dataset}_generated_queries_${query_type}.tsv \
      --output retrieval_runs/run.gte-base-en-v1.5.${dataset}.generated-queries-${query_type}.txt \
      --hits 100
done

for dataset in 'fiqa' 'scifact' 'trec-covid' 'nfcorpus' 'arguana' 'webis-touche2020' 'dbpedia-entity' 'scidocs' 'climate-fever'; do
  for query_type in 'keywords' 'titles' 'claims' 'questions' 'random' 'msmarco'; do

    python3 -m pyserini.search.faiss \
      --threads 16 --batch-size 8192 \
      --encoder-class auto --encoder Alibaba-NLP/gte-base-en-v1.5 --l2-norm --pooling cls \
      --query-prefix "" \
      --index  general_indices/gte-base-en-v1.5_${dataset}_index  \
      --topics generated_queries/${dataset}_generated_queries_${query_type}.tsv \
      --output retrieval_runs/run.gte-base-en-v1.5.${dataset}.generated-queries-${query_type}.txt \
      --hits 100
  done
done