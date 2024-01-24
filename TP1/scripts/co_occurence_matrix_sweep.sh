python main.py --multirun \
    similarity=cosine_similarity,ip_norms_similarity,intersection_similarity \
    feature_extractor=co_occurrence_matrix \
    feature_extractor.angles=0,0.785398163,1.57079633,2.35619449 \
    feature_extractor.levels=4,6,8 \
    feature_extractor.channel=grey,red,green,blue,every \
    descriptor=matrix_flattener


