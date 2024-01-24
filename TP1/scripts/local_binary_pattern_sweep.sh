python main.py --multirun \
    similarity=cosine_similarity,ip_norms_similarity,intersection_similarity \
    feature_extractor=local_binary_pattern \
    feature_extractor.n_points=8,24,40,60 \
    feature_extractor.radius=1,5,10,15 \
    feature_extractor.method=uniform,default \
    descriptor=histogram


