#!/usr/bin/env bash
java -jar tools/RankLib-2.1-patched.jar -train data/output/raw_query_less_train.txt -ranker 3 -kcv 3 -kcvmd models/raw_query_less_adarank_model -kcvmn raw_query_less_adarank_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/raw_query_less_train.txt -ranker 5 -kcv 3 -kcvmd models/raw_query_less_lambda_model -kcvmn raw_query_less_lambda_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/raw_query_less_train.txt -ranker 7 -kcv 3 -kcvmd models/raw_query_less_listnet_model -kcvmn raw_query_less_listnet_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/raw_query_full_train.txt -ranker 3 -kcv 3 -kcvmd models/raw_query_full_adarank_model -kcvmn raw_query_full_adarank_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/raw_query_full_train.txt -ranker 5 -kcv 3 -kcvmd models/raw_query_full_lambda_model -kcvmn raw_query_full_lambda_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/raw_query_full_train.txt -ranker 7 -kcv 3 -kcvmd models/raw_query_full_listnet_model -kcvmn raw_query_full_listnet_model -metric2t NDCG@10 -metric2T ERR@10



java -jar tools/RankLib-2.1-patched.jar -train data/output/opt_query_less_train.txt -ranker 3 -kcv 3 -kcvmd models/opt_query_less_adarank_model -kcvmn opt_query_less_adarank_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/opt_query_less_train.txt -ranker 5 -kcv 3 -kcvmd models/opt_query_less_lambda_model -kcvmn opt_query_less_lambda_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/opt_query_less_train.txt -ranker 7 -kcv 3 -kcvmd models/opt_query_less_listnet_model -kcvmn opt_query_less_listnet_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/opt_query_full_train.txt -ranker 3 -kcv 3 -kcvmd models/opt_query_full_adarank_model -kcvmn opt_query_full_adarank_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/opt_query_full_train.txt -ranker 5 -kcv 3 -kcvmd models/opt_query_full_lambda_model -kcvmn opt_query_full_lambda_model -metric2t NDCG@10 -metric2T ERR@10


java -jar tools/RankLib-2.1-patched.jar -train data/output/opt_query_full_train.txt -ranker 7 -kcv 3 -kcvmd models/opt_query_full_listnet_model -kcvmn opt_query_full_listnet_model -metric2t NDCG@10 -metric2T ERR@10


